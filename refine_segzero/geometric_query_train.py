import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from training_scripts.refine_segzero.common import append_metric, build_prediction_record, dtype_from_config, save_metrics_history
from training_scripts.refine_segzero.data import (
    build_direct_reject_sft_dataset,
    build_eval_dataset,
    build_stage1_cache_dataset,
    build_train_dataset,
    direct_reject_sft_collate_fn,
    geometric_collate_fn,
)

from training_scripts.refine_segzero.geometric_query_export import export_geometric_model
from training_scripts.refine_segzero.geometric_query_model import DEFAULT_SAM2_CONFIG, RefineSegZeroModel


def branch_exists_global(local_has_branch: bool, device: torch.device) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return bool(local_has_branch)
    flag = torch.tensor([1 if local_has_branch else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def zero_surrogate_loss(parameters: Any, device: torch.device) -> torch.Tensor:
    params = [param for param in parameters if param.requires_grad]
    if not params:
        return torch.zeros((), device=device, dtype=torch.float32)
    zero = None
    for param in params:
        term = param.sum() * 0.0
        zero = term if zero is None else zero + term
    return zero


def save_rank_eval_part(parts_dir: Path, rank: int, results: Dict[str, Any]) -> Path:
    parts_dir.mkdir(parents=True, exist_ok=True)
    final_path = parts_dir / f"rank_{rank}.json"
    tmp_path = parts_dir / f"rank_{rank}.json.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    tmp_path.replace(final_path)
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Any:
    return OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))

def is_direct_reject_sft_mode(cfg: Any) -> bool:
    return str(cfg.get("train_mode", "")).strip().lower() == "direct_reject_sft"


def cfg_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "ListConfig":
        return list(value)
    return [value]


def resolve_aligned_prompt_jitter_ratios(cfg: Any, training_progress: float) -> tuple[float, float]:
    """Return aligned-only box/point jitter limits for the current training progress."""
    if not bool(cfg.get("aligned_prompt_jitter_enabled", False)):
        return 0.0, 0.0

    phase_ends = [float(value) for value in cfg_list(cfg.get("aligned_prompt_jitter_phase_ends", [1.0]))]
    box_ratios = [float(value) for value in cfg_list(cfg.get("aligned_box_jitter_ratios", [0.0]))]
    point_ratios = [float(value) for value in cfg_list(cfg.get("aligned_point_jitter_ratios", [0.0]))]
    if not phase_ends or len(phase_ends) != len(box_ratios) or len(phase_ends) != len(point_ratios):
        raise ValueError(
            "aligned prompt jitter schedule requires equally sized non-empty "
            "phase_ends, box_jitter_ratios, and point_jitter_ratios lists."
        )
    if any(end <= 0.0 or end > 1.0 for end in phase_ends) or any(
        later <= earlier for earlier, later in zip(phase_ends, phase_ends[1:])
    ):
        raise ValueError("aligned_prompt_jitter_phase_ends must be strictly increasing values in (0, 1].")
    if any(ratio < 0.0 for ratio in [*box_ratios, *point_ratios]):
        raise ValueError("aligned prompt jitter ratios must be non-negative.")

    progress = min(max(float(training_progress), 0.0), 1.0)
    for phase_end, box_ratio, point_ratio in zip(phase_ends, box_ratios, point_ratios):
        if progress < phase_end:
            return box_ratio, point_ratio
    return box_ratios[-1], point_ratios[-1]

def moving_average(values: list, window: int) -> list:
    if window <= 1 or not values:
        return list(values)
    smoothed = []
    running_sum = 0.0
    queue = []
    for value in values:
        numeric = float(value)
        queue.append(numeric)
        running_sum += numeric
        if len(queue) > window:
            running_sum -= queue.pop(0)
        smoothed.append(running_sum / max(len(queue), 1))
    return smoothed


def save_plots(output_dir: Path, history: Dict[str, list]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_smooth_window = 20
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(
        axes.flatten(),
        ["aligned/loss", "aligned/mask_loss", "direct/loss", "direct/mask_loss"],
    ):
        points = history.get(key, [])
        steps = [item["step"] for item in points]
        raw_values = [item["value"] for item in points]
        smooth_values = moving_average(raw_values, loss_smooth_window)
        ax.plot(steps, raw_values, color="tab:blue", alpha=0.18, linewidth=0.8, label="raw")
        ax.plot(steps, smooth_values, color="tab:orange", linewidth=1.8, label=f"ma{loss_smooth_window}")
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
        if points:
            ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["eval/aligned_mean_iou", "eval/direct_mean_iou"]):
        points = history.get(key, [])
        ax.plot([item["step"] for item in points], [item["value"] for item in points])
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "eval_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)



def evaluate(
    accelerator: Accelerator,
    model: RefineSegZeroModel,
    cfg: Any,
    global_step: int,
    output_dir: Path,
    train_branch: str = "both",
) -> Optional[Dict[str, float]]:
    branch_key = str(train_branch or "both").strip().lower()

    if branch_key == "direct":
        direct_eval_path = str(cfg.get("direct_eval_json_path", "")).strip()
        if not direct_eval_path:
            return None

        dataset = build_direct_reject_sft_dataset(
            reflect_json_path=direct_eval_path,
            max_sample_ratio=cfg.get("eval_max_sample_ratio"),
            sample_seed=int(cfg.get("sample_seed", cfg.get("seed", 42))),
        )
        eval_collate_fn = direct_reject_sft_collate_fn

    else:
        data_mode = str(cfg.get("data_mode", "refcoco")).strip().lower()
        if data_mode == "stage1_cache":
            stage1_eval_path = str(cfg.get("stage1_eval_cache_path", "")).strip()
            if not stage1_eval_path:
                return None
            dataset = build_stage1_cache_dataset(
                [stage1_eval_path],
                max_sample_ratio_per_file=cfg.get("eval_max_sample_ratio"),
                sample_seed=int(cfg.get("sample_seed", cfg.get("seed", 42))),
            )
        else:
            if not str(cfg.get("eval_json_path", "")).strip():
                return None
            dataset = build_eval_dataset(
                str(cfg.eval_json_path),
                image_root=str(cfg.image_root),
                max_sample_ratio=cfg.get("eval_max_sample_ratio"),
                sample_seed=int(cfg.get("sample_seed", cfg.get("seed", 42))),
            )
        eval_collate_fn = geometric_collate_fn
    if len(dataset) == 0:
        return None
    eval_dir = output_dir / "eval"
    step_dir = eval_dir / f"step_{global_step}"
    parts_dir = step_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index in tqdm(range(accelerator.process_index, len(dataset), accelerator.num_processes), disable=not accelerator.is_local_main_process, desc="eval"):
        sample = dataset[index]
        try:
            raw_model = accelerator.unwrap_model(model)
            was_training = raw_model.training
            raw_model.eval()
            batch = eval_collate_fn(
                [sample],
                processor=raw_model.processor,
                resize_size=int(cfg.resize_size),
                sam_image_size=int(cfg.sam_image_size),
                max_pixels=int(cfg.max_pixels),
                min_pixels=int(cfg.min_pixels),
                **(
                    {"direct_prompt_mode": str(cfg.get("direct_prompt_mode", "query_reflect"))}
                    if branch_key == "direct"
                    else {}
                ),
            )
            eval_forward_branch = branch_key
            with torch.no_grad():
                if eval_forward_branch == "direct" and bool(getattr(raw_model, "has_direct_lora_adapter", False)):
                    with raw_model.direct_lora_enabled():
                        outputs = raw_model.stage1_forward(batch=batch, train_branch=eval_forward_branch)
                else:
                    outputs = raw_model.stage1_forward(batch=batch, train_branch=eval_forward_branch)
            if was_training:
                raw_model.train()
            aligned_pred = outputs["aligned_predictions"][0]
            direct_pred = outputs["direct_predictions"][0]
            record = {}

            if branch_key != "direct":
                record["aligned"] = build_prediction_record(
                    image_id=sample.image_id,
                    sample_id=sample.sample_id,
                    question=sample.refexp,
                    output_text="",
                    think="",
                    pred_bbox=aligned_pred["bbox"],
                    pred_points=aligned_pred["points"],
                    pred_mask=aligned_pred["pred_mask"],
                    gt_mask=sample.gt_mask,
                    branch_type="aligned",
                    error=str(aligned_pred.get("skip_reason", "")),
                )

            if branch_key != "aligned":
                record["direct"] = build_prediction_record(
                    image_id=sample.image_id,
                    sample_id=sample.sample_id,
                    question=sample.refexp,
                    output_text="",
                    think="",
                    pred_bbox=direct_pred["bbox"],
                    pred_points=direct_pred["points"],
                    pred_mask=direct_pred["pred_mask"],
                    gt_mask=sample.gt_mask,
                    branch_type="direct",
                    error=str(direct_pred.get("skip_reason", "")),
                )

            results.append(record)
        except Exception as exc:
            zero_mask = np.zeros_like(sample.gt_mask, dtype=np.uint8)
            record = {}

            if branch_key != "direct":
                record["aligned"] = build_prediction_record(
                    sample.image_id,
                    sample.sample_id,
                    sample.refexp,
                    "",
                    "",
                    [0, 0, 0, 0],
                    [[0, 0], [0, 0]],
                    zero_mask,
                    sample.gt_mask,
                    branch_type="aligned",
                    error=str(exc),
                )

            if branch_key != "aligned":
                record["direct"] = build_prediction_record(
                    sample.image_id,
                    sample.sample_id,
                    sample.refexp,
                    "",
                    "",
                    [0, 0, 0, 0],
                    [[0, 0], [0, 0]],
                    zero_mask,
                    sample.gt_mask,
                    branch_type="direct",
                    error=str(exc),
                )

            results.append(record)
    save_rank_eval_part(parts_dir, accelerator.process_index, results)
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return None
    merged = []
    for rank in range(accelerator.num_processes):
        with open(parts_dir / f"rank_{rank}.json", "r", encoding="utf-8") as f:
            merged.extend(json.load(f))
    aligned_items = [item["aligned"] for item in merged if "aligned" in item]
    direct_items = [item["direct"] for item in merged if "direct" in item]

    summary = {}
    if aligned_items:
        summary.update(
            {
                "aligned_mean_iou": sum(float(item["iou"]) for item in aligned_items) / max(len(aligned_items), 1),
                "aligned_mean_bbox_iou": sum(float(item["bbox_iou"]) for item in aligned_items) / max(len(aligned_items), 1),
            }
        )
    if direct_items:
        summary.update(
            {
                "direct_mean_iou": sum(float(item["iou"]) for item in direct_items) / max(len(direct_items), 1),
                "direct_mean_bbox_iou": sum(float(item["bbox_iou"]) for item in direct_items) / max(len(direct_items), 1),
            }
        )
    with open(eval_dir / f"eval_step_{global_step}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": merged}, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(gradient_accumulation_steps=int(cfg.gradient_accumulation_steps))
    set_seed(int(cfg.seed))

    train_branch = str(cfg.get("train_branch", "both")).strip().lower()
    if train_branch not in {"both", "aligned", "direct"}:
        raise ValueError(f"Unsupported train_branch: {train_branch}")


    model = RefineSegZeroModel(
        base_model_path=str(cfg.base_model_path),
        processor_path=str(cfg.get("processor_path", "") or str(cfg.base_model_path)),
        sam_model_path=str(cfg.sam_checkpoint_path),
        sam_model_cfg=str(cfg.get("sam_model_cfg", DEFAULT_SAM2_CONFIG)),
        attn_implementation=str(cfg.attn_implementation),
        torch_dtype=dtype_from_config(cfg),
        aligned_output_tokens=int(cfg.get("aligned_output_tokens", 4)),
        direct_output_tokens=int(cfg.get("direct_output_tokens", 4)),
        num_learnable_queries=int(cfg.get("num_learnable_queries", 64)),
        prompt_space=str(cfg.get("prompt_space", "sparse")),
        hidden_state_layer=int(cfg.get("hidden_state_layer", -2)),
        connector_qformer_dim=int(cfg.get("connector_qformer_dim", 1024)),
        freeze_mllm=True,
        freeze_sam=True,
    )
    model.aligned_prompt_source = str(cfg.get("aligned_prompt_source", "stage1")).strip().lower()
    if model.aligned_prompt_source not in {"stage1", "gt_mask"}:
        raise ValueError(
            "aligned_prompt_source must be either 'stage1' or 'gt_mask', "
            f"got {model.aligned_prompt_source!r}."
        )
    model.aligned_prompt_sampling_enabled = bool(cfg.get("aligned_prompt_sampling_enabled", False))
    model.aligned_prompt_variants = [
        str(value).strip().lower()
        for value in cfg_list(cfg.get("aligned_prompt_variants", ["box", "points", "box_points", "embedding"]))
        if str(value).strip()
    ]
    model.aligned_prompt_jitter_enabled = bool(cfg.get("aligned_prompt_jitter_enabled", False))
    model.aligned_prompt_variant_probs = [
        float(value)
        for value in cfg_list(cfg.get("aligned_prompt_variant_probs", [0.25, 0.25, 0.25, 0.25]))
    ]
    model.aligned_box_jitter_ratio = float(cfg.get("aligned_box_jitter_ratio", 0.0))
    model.aligned_point_jitter_ratio = float(cfg.get("aligned_point_jitter_ratio", 0.0))
    model.aligned_soft_bbox_loss_weight = float(cfg.get("aligned_soft_bbox_loss_weight", 0.0))
    model.aligned_box_gap_loss_weight = float(cfg.get("aligned_box_gap_loss_weight", 0.0))
    model.aligned_prompt_embedding_loss_weight = float(cfg.get("aligned_prompt_embedding_loss_weight", 0.0))
    print(f"[geometric-query-train] aligned prompt source={model.aligned_prompt_source}")
    if model.aligned_prompt_embedding_loss_weight > 0.0 and model.prompt_space != "two_stage_sparse":
        raise ValueError("aligned_prompt_embedding_loss_weight requires prompt_space='two_stage_sparse'.")
    if model.aligned_prompt_sampling_enabled:
        print(
            "[geometric-query-train] aligned prompt sampling enabled | "
            f"variants={model.aligned_prompt_variants} "
            f"probs={model.aligned_prompt_variant_probs} "
            f"box_jitter={model.aligned_box_jitter_ratio} "
            f"point_jitter={model.aligned_point_jitter_ratio}"
        )
    if model.aligned_prompt_jitter_enabled:
        print(
            "[geometric-query-train] aligned prompt jitter curriculum enabled | "
            f"phase_ends={cfg_list(cfg.get('aligned_prompt_jitter_phase_ends', [1.0]))} "
            f"box_ratios={cfg_list(cfg.get('aligned_box_jitter_ratios', [0.0]))} "
            f"point_ratios={cfg_list(cfg.get('aligned_point_jitter_ratios', [0.0]))}"
        )
    if any(
        value > 0.0
        for value in [
            model.aligned_soft_bbox_loss_weight,
            model.aligned_box_gap_loss_weight,
            model.aligned_prompt_embedding_loss_weight,
        ]
    ):
        print(
            "[geometric-query-train] aligned auxiliary losses | "
            f"soft_bbox={model.aligned_soft_bbox_loss_weight} "
            f"box_gap={model.aligned_box_gap_loss_weight} "
            f"prompt_embedding={model.aligned_prompt_embedding_loss_weight}"
        )
    init_geometric_export_dir = str(cfg.get("init_geometric_export_dir", "")).strip()
    if init_geometric_export_dir:
        trainable_state_path = Path(init_geometric_export_dir) / "trainable" / "trainable_state_dict.pt"
        if not trainable_state_path.is_file():
            raise FileNotFoundError(f"init_geometric_export_dir is missing trainable state: {trainable_state_path}")

        state_dict = torch.load(trainable_state_path, map_location="cpu")
        model.load_custom_state_dict(state_dict, strict=False)

        aligned_query_loaded = "aligned_learnable_query" in state_dict or "learnable_query" in state_dict
        direct_query_loaded = "direct_learnable_query" in state_dict or "learnable_query" in state_dict
        aligned_connector_count = sum(1 for key in state_dict if str(key).startswith("aligned_connector."))
        direct_connector_count = sum(1 for key in state_dict if str(key).startswith("direct_connector."))

        print(
            "[geometric-query-train] Loaded pretrained query/connector state from "
            f"{trainable_state_path} | "
            f"aligned_query={aligned_query_loaded} direct_query={direct_query_loaded} "
            f"aligned_connector_keys={aligned_connector_count} direct_connector_keys={direct_connector_count}"
        )
    direct_enable_lora = bool(cfg.get("direct_enable_lora", False))
    if direct_enable_lora and train_branch != "direct":
        raise ValueError("direct_enable_lora is only supported when train_branch='direct'.")
    if direct_enable_lora:
        direct_lora_adapter_name = str(cfg.get("direct_lora_adapter_name", "direct"))
        direct_lora_resume_path = str(cfg.get("direct_lora_resume_path", "") or "").strip()
        lora_path = Path(direct_lora_resume_path) if direct_lora_resume_path else None
        if lora_path is None and init_geometric_export_dir:
            candidate_lora_path = Path(init_geometric_export_dir) / "mllm_direct_lora"
            if candidate_lora_path.is_dir() and (
                (candidate_lora_path / "adapter_config.json").is_file()
                or (candidate_lora_path / direct_lora_adapter_name / "adapter_config.json").is_file()
                or any(child.is_dir() and (child / "adapter_config.json").is_file() for child in candidate_lora_path.iterdir())
            ):
                lora_path = candidate_lora_path
        if lora_path is not None and lora_path.is_dir() and (
            (lora_path / "adapter_config.json").is_file()
            or (lora_path / direct_lora_adapter_name / "adapter_config.json").is_file()
            or any(child.is_dir() and (child / "adapter_config.json").is_file() for child in lora_path.iterdir())
        ):
            model.load_direct_lora_adapter(
                str(lora_path),
                adapter_name=direct_lora_adapter_name,
                is_trainable=True,
            )
            model.direct_lora_r = int(cfg.get("direct_lora_r", getattr(model, "direct_lora_r", 0)))
            model.direct_lora_alpha = int(cfg.get("direct_lora_alpha", getattr(model, "direct_lora_alpha", 0)))
            model.direct_lora_dropout = float(cfg.get("direct_lora_dropout", getattr(model, "direct_lora_dropout", 0.0)))
            model.direct_lora_target_modules = cfg_list(
                cfg.get("direct_lora_target_modules", getattr(model, "direct_lora_target_modules", []))
            )
            print(
                "[geometric-query-train] resumed direct LoRA adapter | "
                f"path={lora_path} adapter={model.direct_lora_adapter_name}"
            )
        else:
            model.attach_direct_lora(
                r=int(cfg.get("direct_lora_r", 16)),
                alpha=int(cfg.get("direct_lora_alpha", 64)),
                dropout=float(cfg.get("direct_lora_dropout", 0.05)),
                target_modules=cfg_list(cfg.get("direct_lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
                adapter_name=direct_lora_adapter_name,
            )
            print(
                "[geometric-query-train] initialized direct LoRA adapter | "
                f"r={model.direct_lora_r} alpha={model.direct_lora_alpha} "
                f"dropout={model.direct_lora_dropout} targets={model.direct_lora_target_modules}"
            )
    if train_branch == "aligned":
        model.freeze_direct_branch()
    elif train_branch == "direct":
        model.freeze_aligned_branch()

    if accelerator.is_main_process:
        trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
        aligned_trainable = [name for name in trainable_names if name == "aligned_learnable_query" or name.startswith("aligned_connector.")]
        print(
            "[geometric-query-train] trainable parameter summary | "
            f"count={len(trainable_names)} "
            f"has_direct_query={'direct_learnable_query' in trainable_names} "
            f"direct_connector={sum(1 for name in trainable_names if name.startswith('direct_connector.'))} "
            f"lora={sum(1 for name in trainable_names if 'lora_' in name)} "
            f"aligned_trainable={len(aligned_trainable)} "
            f"first={trainable_names[:20]}"
        )
    data_mode = str(cfg.get("data_mode", "refcoco")).strip().lower()
    if train_branch == "direct":
        train_dataset = build_direct_reject_sft_dataset(
            reflect_json_path=str(cfg.get("direct_reflect_json_path", "")),
            max_sample_ratio=cfg.get("train_max_sample_ratio_per_file"),
            sample_seed=int(cfg.get("sample_seed", cfg.get("seed", 42))),
        )
    elif data_mode == "stage1_cache":
        train_dataset = build_stage1_cache_dataset(
            cfg_list(cfg.get("stage1_train_cache_paths", [])),
            max_sample_ratio_per_file=cfg.get("train_max_sample_ratio_per_file"),
            sample_seed=int(cfg.get("sample_seed", cfg.get("seed", 42))),
        )
    else:
        train_dataset = build_train_dataset(
            list(cfg.train_json_paths),
            image_root=str(cfg.image_root),
            max_sample_ratio_per_file=cfg.get("train_max_sample_ratio_per_file"),
            sample_seed=int(cfg.get("sample_seed", cfg.get("seed", 42))),
        )

    print("train_dataset", len(train_dataset))

    collate_fn = direct_reject_sft_collate_fn if train_branch == "direct" else geometric_collate_fn

    collate_kwargs = {}
    if train_branch == "direct":
        collate_kwargs["direct_prompt_mode"] = str(cfg.get("direct_prompt_mode", "query_reflect"))

    collate = partial(
        collate_fn,
        processor=model.processor,
        resize_size=int(cfg.resize_size),
        sam_image_size=int(cfg.sam_image_size),
        max_pixels=int(cfg.max_pixels),
        min_pixels=int(cfg.min_pixels),
        **collate_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=bool(cfg.get("shuffle", False)),
        num_workers=int(cfg.num_workers),
        collate_fn=collate,
        pin_memory=True,
    )
    connector_learning_rate = float(cfg.get("connector_learning_rate", cfg.learning_rate))
    aligned_param_groups = list(model.aligned_connector.parameters()) + [model.aligned_learnable_query]
    if train_branch == "direct":
        direct_param_groups = list(model.direct_connector.parameters()) + [model.direct_learnable_query] + model.direct_lora_parameters()
    else:
        direct_param_groups = list(model.direct_connector.parameters())
    aligned_optimizer = torch.optim.AdamW(
        aligned_param_groups,
        lr=connector_learning_rate,
        weight_decay=float(cfg.weight_decay),
    )
    direct_optimizer = torch.optim.AdamW(
        direct_param_groups,
        lr=connector_learning_rate,
        weight_decay=float(cfg.weight_decay),
    )
    updates_per_epoch = max(int(math.ceil(len(train_loader) / max(int(cfg.gradient_accumulation_steps), 1))), 1)
    total_steps = max(updates_per_epoch * int(cfg.num_epochs), 1)
    warmup_steps = min(int(cfg.warmup_steps), total_steps)
    aligned_scheduler = get_scheduler(
        str(cfg.scheduler_type),
        optimizer=aligned_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    direct_scheduler = get_scheduler(
        str(cfg.scheduler_type),
        optimizer=direct_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    model, aligned_optimizer, direct_optimizer, train_loader, aligned_scheduler, direct_scheduler = accelerator.prepare(
        model, aligned_optimizer, direct_optimizer, train_loader, aligned_scheduler, direct_scheduler
    )
    curriculum_updates_per_epoch = max(
        int(math.ceil(len(train_loader) / max(int(cfg.gradient_accumulation_steps), 1))),
        1,
    )
    curriculum_total_steps = max(curriculum_updates_per_epoch * int(cfg.num_epochs), 1)
    history: Dict[str, list] = {}
    best_metric = float("-inf")
    global_step = 0
    update_step = 0
    eval_every_steps = int(cfg.get("eval_every_steps", 100))
    save_best_every_steps = int(cfg.get("save_best_every_steps", 500))
    current_aligned_box_jitter_ratio = 0.0
    current_aligned_point_jitter_ratio = 0.0
    aligned_optimizer.zero_grad(set_to_none=True)
    direct_optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(cfg.num_epochs)):
        progress = tqdm(train_loader, disable=not accelerator.is_local_main_process, desc=f"epoch {epoch + 1}")
        for batch in progress:
            with accelerator.accumulate(model):
                raw_model = accelerator.unwrap_model(model)
                if train_branch == "aligned":
                    current_aligned_box_jitter_ratio, current_aligned_point_jitter_ratio = resolve_aligned_prompt_jitter_ratios(
                        cfg,
                        training_progress=float(update_step) / float(curriculum_total_steps),
                    )
                    raw_model.aligned_box_jitter_ratio = current_aligned_box_jitter_ratio
                    raw_model.aligned_point_jitter_ratio = current_aligned_point_jitter_ratio
                aligned_backward_params = list(raw_model.aligned_connector.parameters()) + [raw_model.aligned_learnable_query]
                direct_backward_params = list(raw_model.direct_connector.parameters())
                if train_branch == "direct":
                    direct_backward_params = direct_backward_params + [raw_model.direct_learnable_query] + raw_model.direct_lora_parameters()
                outputs = model(batch=batch, train_branch=train_branch)
                aligned_loss = outputs["aligned_loss"]
                direct_loss = outputs["direct_loss"]
                aligned_valid_local = aligned_loss is not None and int(outputs["aligned_valid_count"]) > 0
                direct_valid_local = direct_loss is not None and int(outputs["direct_valid_count"]) > 0
                aligned_has_global = branch_exists_global(aligned_valid_local, accelerator.device)
                direct_has_global = branch_exists_global(direct_valid_local, accelerator.device)
                total_backward_loss = None
                if aligned_has_global:
                    aligned_backward_loss = aligned_loss if aligned_valid_local else zero_surrogate_loss(aligned_backward_params, accelerator.device)
                    total_backward_loss = aligned_backward_loss if total_backward_loss is None else total_backward_loss + aligned_backward_loss
                if direct_has_global:
                    direct_backward_loss = direct_loss if direct_valid_local else zero_surrogate_loss(direct_backward_params, accelerator.device)
                    total_backward_loss = direct_backward_loss if total_backward_loss is None else total_backward_loss + direct_backward_loss
                if total_backward_loss is not None:
                    accelerator.backward(total_backward_loss)

                if accelerator.sync_gradients:
                    if aligned_has_global:
                        accelerator.clip_grad_norm_(aligned_backward_params, float(cfg.max_grad_norm))
                        aligned_optimizer.step()
                        aligned_scheduler.step()
                        aligned_optimizer.zero_grad(set_to_none=True)
                    if direct_has_global:
                        accelerator.clip_grad_norm_(direct_backward_params, float(cfg.max_grad_norm))
                        direct_optimizer.step()
                        direct_scheduler.step()
                        direct_optimizer.zero_grad(set_to_none=True)
                    update_step += 1

            global_step += 1
            if accelerator.is_main_process:
                aligned_metrics = outputs["aligned_metrics"]
                direct_metrics = outputs["direct_metrics"]
                append_metric(history, "aligned/loss", global_step, float(aligned_metrics["loss"]))
                append_metric(history, "aligned/mask_loss", global_step, float(aligned_metrics["mask_loss"]))
                append_metric(history, "aligned/bbox_loss", global_step, float(aligned_metrics["bbox_loss"]))
                append_metric(history, "aligned/bbox_iou", global_step, float(aligned_metrics["bbox_iou"]))
                append_metric(history, "aligned/soft_bbox_loss", global_step, float(aligned_metrics.get("soft_bbox_loss", 0.0)))
                append_metric(history, "aligned/soft_bbox_iou", global_step, float(aligned_metrics.get("soft_bbox_iou", 0.0)))
                append_metric(history, "aligned/box_gap_loss", global_step, float(aligned_metrics.get("box_gap_loss", 0.0)))
                append_metric(history, "aligned/prompt_embedding_loss", global_step, float(aligned_metrics.get("prompt_embedding_loss", 0.0)))
                append_metric(history, "aligned/prompt_embedding_valid", global_step, float(aligned_metrics.get("prompt_embedding_valid", 0.0)))
                append_metric(history, "aligned/box_jitter_ratio", global_step, current_aligned_box_jitter_ratio)
                append_metric(history, "aligned/point_jitter_ratio", global_step, current_aligned_point_jitter_ratio)
                append_metric(history, "direct/loss", global_step, float(direct_metrics["loss"]))
                append_metric(history, "direct/mask_loss", global_step, float(direct_metrics["mask_loss"]))
                append_metric(history, "direct/bbox_loss", global_step, float(direct_metrics["bbox_loss"]))
                append_metric(history, "direct/bbox_iou", global_step, float(direct_metrics["bbox_iou"]))
                append_metric(history, "aligned/valid_count", global_step, float(outputs["aligned_valid_count"]))
                append_metric(history, "direct/valid_count", global_step, float(outputs["direct_valid_count"]))
                append_metric(history, "aligned/skipped_count", global_step, float(outputs.get("aligned_skipped_count", 0)))
                append_metric(history, "direct/skipped_count", global_step, float(outputs.get("direct_skipped_count", 0)))
                append_metric(history, "aligned/valid_count_local", global_step, float(int(aligned_valid_local)))
                append_metric(history, "aligned/has_global", global_step, float(int(aligned_has_global)))
                append_metric(history, "direct/valid_count_local", global_step, float(int(direct_valid_local)))
                append_metric(history, "direct/has_global", global_step, float(int(direct_has_global)))
                append_metric(history, "train/update_step", global_step, float(update_step))
                append_metric(history, "aligned/lr", global_step, float(aligned_optimizer.param_groups[0]["lr"]))
                append_metric(history, "direct/lr", global_step, float(direct_optimizer.param_groups[0]["lr"]))
                save_metrics_history(output_dir, history)
                save_plots(output_dir, history)
                display_lr = direct_optimizer.param_groups[0]["lr"] if train_branch == "direct" else aligned_optimizer.param_groups[0]["lr"]
                progress.set_postfix(
                    aligned=f"{aligned_metrics['loss']:.4f}",
                    direct=f"{direct_metrics['loss']:.4f}",
                    lr=f"{display_lr:.2e}",
                    a_global=int(aligned_has_global),
                    d_global=int(direct_has_global),
                )

            if eval_every_steps > 0 and global_step % eval_every_steps == 0:
                eval_summary = evaluate(
                    accelerator=accelerator,
                    model=model,
                    cfg=cfg,
                    global_step=global_step,
                    output_dir=output_dir,
                    train_branch=train_branch,
                )
                if accelerator.is_main_process and eval_summary is not None:
                    if "aligned_mean_iou" in eval_summary:
                        append_metric(history, "eval/aligned_mean_iou", global_step, float(eval_summary["aligned_mean_iou"]))
                    if "aligned_mean_bbox_iou" in eval_summary:
                        append_metric(history, "eval/aligned_mean_bbox_iou", global_step, float(eval_summary["aligned_mean_bbox_iou"]))
                    if "direct_mean_iou" in eval_summary:
                        append_metric(history, "eval/direct_mean_iou", global_step, float(eval_summary["direct_mean_iou"]))
                    if "direct_mean_bbox_iou" in eval_summary:
                        append_metric(history, "eval/direct_mean_bbox_iou", global_step, float(eval_summary["direct_mean_bbox_iou"]))
                    save_metrics_history(output_dir, history)
                    save_plots(output_dir, history)
                    metric = float(
                        sum(
                            float(eval_summary[key])
                            for key in ("aligned_mean_iou", "direct_mean_iou")
                            if key in eval_summary
                        )
                    )
                    if save_best_every_steps > 0 and global_step % save_best_every_steps == 0 and metric > best_metric:
                        best_metric = metric
                        export_geometric_model(
                            accelerator.unwrap_model(model),
                            output_dir / "best_export",
                            OmegaConf.to_container(cfg, resolve=True),
                            bool(cfg.get("export_copy_sam_checkpoint", False)),
                        )

    if accelerator.is_main_process:
        export_geometric_model(accelerator.unwrap_model(model), output_dir / "final_export", OmegaConf.to_container(cfg, resolve=True), bool(cfg.get("export_copy_sam_checkpoint", False)))


if __name__ == "__main__":
    main()
