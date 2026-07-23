import json
from pathlib import Path
from typing import Any, Dict

import torch

from training_scripts.refine_segzero.common import (
    resolve_export_sam_checkpoint,
    save_export_metadata,
    trainable_state_has_learnable_query,
    warn_if_missing_learnable_query,
)
from training_scripts.refine_segzero.geometric_query_model import RefineSegZeroModel


def export_geometric_model(
    model: RefineSegZeroModel,
    export_dir: Path,
    resolved_config: Dict[str, Any],
    copy_sam_checkpoint: bool = False,
) -> None:
    trainable_state = model.geometric_state_dict()
    metadata = {
        "stage": "geometric_query",
        "sam_model_cfg": str(resolved_config["sam_model_cfg"]),
        "sam_checkpoint_path": str(resolved_config["sam_checkpoint_path"]),
        "base_model_path": str(resolved_config["base_model_path"]),
        "processor_path": str(resolved_config.get("processor_path", "") or resolved_config["base_model_path"]),
        "aligned_output_tokens": int(resolved_config.get("aligned_output_tokens", 4)),
        "direct_output_tokens": int(resolved_config.get("direct_output_tokens", 4)),
        "num_learnable_queries": int(resolved_config.get("num_learnable_queries", 64)),
        "hidden_state_layer": int(resolved_config.get("hidden_state_layer", -2)),
        "connector_qformer_dim": int(resolved_config.get("connector_qformer_dim", 1024)),
        "attn_implementation": str(resolved_config.get("attn_implementation", "flash_attention_2")),
        "prompt_space": str(resolved_config.get("prompt_space", "dense")),
        "has_learnable_query": bool(trainable_state_has_learnable_query(trainable_state)),
        "has_aligned_learnable_query": bool("aligned_learnable_query" in trainable_state),
        "has_direct_learnable_query": bool("direct_learnable_query" in trainable_state),
        "query_layout": "dual_branch",
        "direct_lora_enabled": bool(getattr(model, "has_direct_lora_adapter", False)),
        "direct_lora_adapter_path": "mllm_direct_lora" if bool(getattr(model, "has_direct_lora_adapter", False)) else "",
        "direct_lora_adapter_name": str(getattr(model, "direct_lora_adapter_name", "direct")),
        "direct_lora_r": int(getattr(model, "direct_lora_r", resolved_config.get("direct_lora_r", 0))),
        "direct_lora_alpha": int(getattr(model, "direct_lora_alpha", resolved_config.get("direct_lora_alpha", 0))),
        "direct_lora_dropout": float(getattr(model, "direct_lora_dropout", resolved_config.get("direct_lora_dropout", 0.0))),
        "direct_lora_target_modules": list(getattr(model, "direct_lora_target_modules", resolved_config.get("direct_lora_target_modules", []))),
        "mllm_export_is_clean_base": True,
    }
    save_export_metadata(
        export_dir=export_dir,
        metadata=metadata,
        trainable_state_dict=trainable_state,
        processor=model.processor,
        qwen=model.get_qwen_base_model_for_export(),
        sam_state_dict={name: param.detach().cpu() for name, param in model.sam.state_dict().items()},
        copy_sam_checkpoint=copy_sam_checkpoint,
        qwen_state_dict=model.get_qwen_clean_base_state_dict_for_export(),
    )
    if bool(getattr(model, "has_direct_lora_adapter", False)):
        model.save_direct_lora_adapter(export_dir / "mllm_direct_lora")
    with open(export_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved_config, f, ensure_ascii=False, indent=2)


def load_geometric_model_from_export(export_dir: Path, device: torch.device) -> RefineSegZeroModel:
    with open(export_dir / "export_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    sam_checkpoint_path, sam_model_cfg = resolve_export_sam_checkpoint(export_dir)
    model = RefineSegZeroModel(
        base_model_path=str(export_dir / "mllm"),
        processor_path=str(export_dir / "mllm"),
        sam_model_path=sam_checkpoint_path,
        sam_model_cfg=sam_model_cfg,
        attn_implementation=str(metadata.get("attn_implementation", "flash_attention_2")),
        aligned_output_tokens=int(metadata.get("aligned_output_tokens", 4)),
        direct_output_tokens=int(metadata.get("direct_output_tokens", 4)),
        num_learnable_queries=int(metadata.get("num_learnable_queries", 64)),
        prompt_space=str(metadata.get("prompt_space", "sparse")),
        hidden_state_layer=int(metadata.get("hidden_state_layer", -1)),
        connector_qformer_dim=int(metadata.get("connector_qformer_dim", 1024)),
        freeze_mllm=True,
        freeze_sam=True,
    )
    state_dict = torch.load(export_dir / "trainable" / "trainable_state_dict.pt", map_location="cpu")
    warn_if_missing_learnable_query(state_dict, f"Geometric export at {export_dir}")
    if "learnable_query" in state_dict:
        state_dict.setdefault("aligned_learnable_query", state_dict["learnable_query"])
        state_dict.setdefault("direct_learnable_query", state_dict["learnable_query"])
    model.load_custom_state_dict(state_dict, strict=False)
    if bool(metadata.get("direct_lora_enabled", False)):
        adapter_rel_path = str(metadata.get("direct_lora_adapter_path", "mllm_direct_lora") or "mllm_direct_lora")
        adapter_path = export_dir / adapter_rel_path
        model.load_direct_lora_adapter(
            str(adapter_path),
            adapter_name=str(metadata.get("direct_lora_adapter_name", "direct")),
            is_trainable=False,
        )
        model.direct_lora_r = int(metadata.get("direct_lora_r", 0))
        model.direct_lora_alpha = int(metadata.get("direct_lora_alpha", 0))
        model.direct_lora_dropout = float(metadata.get("direct_lora_dropout", 0.0))
        model.direct_lora_target_modules = list(metadata.get("direct_lora_target_modules", []))
        print(
            "[geometric-export-load] Loaded direct LoRA adapter from "
            f"{adapter_path} | adapter={model.direct_lora_adapter_name}",
            flush=True,
        )

    aligned_query_loaded = "aligned_learnable_query" in state_dict
    direct_query_loaded = "direct_learnable_query" in state_dict
    legacy_query_loaded = "learnable_query" in state_dict
    aligned_connector_count = sum(1 for key in state_dict if str(key).startswith("aligned_connector."))
    direct_connector_count = sum(1 for key in state_dict if str(key).startswith("direct_connector."))

    print(
        "[geometric-export-load] Loaded trainable state from "
        f"{export_dir / 'trainable' / 'trainable_state_dict.pt'} | "
        f"aligned_query={aligned_query_loaded} "
        f"direct_query={direct_query_loaded} "
        f"legacy_query={legacy_query_loaded} "
        f"aligned_connector_keys={aligned_connector_count} "
        f"direct_connector_keys={direct_connector_count}",
        flush=True,
    )

    sam_state = torch.load(export_dir / "sam2" / "sam2_state_dict.pt", map_location="cpu")
    model.sam.load_state_dict(sam_state, strict=False)
    model.to(device)
    model.eval()
    return model

