import argparse
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from hashlib import sha1
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from training_scripts.refine_segzero.Rl_data.build_init_box_samples import build_init_box_record
from training_scripts.refine_segzero.Rl_data.rl_data_utils import (
    append_jsonl,
    env_rank,
    iter_source_record_batches_shard,
    jsonl_to_json,
    merge_part_dir,
    wait_for_parts,
    write_json,
)
from training_scripts.refine_segzero.Rl_data.sam_utils import build_sam2_predictor
from training_scripts.refine_segzero.common import (
    build_mask_crop_image,
    compute_box_iou,
    compute_iou,
    encode_binary_mask,
    extract_answer_json_and_think,
    resolve_export_sam_checkpoint,
)
from training_scripts.refine_segzero.data import geometric_collate_fn
from training_scripts.refine_segzero.geometric_query_export import load_geometric_model_from_export
from training_scripts.refine_segzero.common import RefineSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json-paths", nargs="*", default=[])
    parser.add_argument("--val-json-paths", nargs="*", default=[])
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--stage1-export-dir", type=str, default="")
    parser.add_argument("--mllm-model-path", type=str, default="")
    parser.add_argument("--processor-path", type=str, default="")
    parser.add_argument("--sam-checkpoint-path", type=str, default="")
    parser.add_argument("--sam-model-cfg", type=str, default="sam2_hiera_l.yaml")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-sample-ratio-per-file", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--num-parts", type=int, default=-1)
    parser.add_argument("--resize-size", type=int, default=840)
    parser.add_argument("--sam-image-size", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def _cleanup_output_dir(output_dir: Path) -> None:
    targets = [
        output_dir / "init_box_train_parts",
        output_dir / "init_box_val_parts",
        output_dir / "stage1_cache_train_parts",
        output_dir / "stage1_cache_val_parts",
        output_dir / "stage1_crop_images",
        output_dir / "stage1_proposal_visualizations",
        output_dir / "stage1_part_summaries",
        output_dir / "stage1_done_markers",
        output_dir / "init_box_train.json",
        output_dir / "init_box_val.json",
        output_dir / "stage1_cache_train.json",
        output_dir / "stage1_cache_val.json",
        output_dir / "stage1_data_summary.json",
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _wait_for_cleanup(output_dir: Path, idx: int, overwrite_output: bool) -> None:
    if not overwrite_output:
        return
    marker = output_dir / "stage1_cleanup_done.json"
    if idx == 0:
        _cleanup_output_dir(output_dir)
        write_json(marker, {"status": "ready"})
    wait_for_parts([marker], timeout_s=60, stable_checks=1)


def _device_for_rank(rank: int) -> torch.device:
    if torch.cuda.is_available():
        device_count = max(torch.cuda.device_count(), 1)
        return torch.device(f"cuda:{rank % device_count}")
    return torch.device("cpu")




def _dtype_from_args(args: argparse.Namespace) -> torch.dtype:
    if bool(args.fp16):
        return torch.float16
    if bool(args.bf16):
        return torch.bfloat16
    return torch.float32


class PlainMLLMStage1Model:
    def __init__(
        self,
        model_path: str,
        processor_path: str,
        device: torch.device,
        torch_dtype: torch.dtype,
        attn_implementation: str,
    ) -> None:
        processor_source = processor_path or model_path
        self.processor = AutoProcessor.from_pretrained(
            processor_source,
            padding_side="left",
            trust_remote_code=True,
        )
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        self.qwen.to(device)
        self.qwen.eval()

    def _move_inputs_to_device(self, model_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(self.qwen.device) if torch.is_tensor(value) else value for key, value in model_inputs.items()}

    def _generate_once(
        self,
        model_inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_return_sequences: int = 1,
    ) -> Tuple[List[Any], Any]:
        device_inputs = self._move_inputs_to_device(model_inputs)
        generate_kwargs = dict(
            **device_inputs,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        with torch.inference_mode():
            outputs = self.qwen.generate(**generate_kwargs)
        generated = outputs.sequences
        input_len = int(device_inputs["input_ids"].shape[1])
        trimmed = generated[:, input_len:]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        results: List[Any] = []
        for seq, text in zip(trimmed, texts):
            try:
                answer, think, _ = extract_answer_json_and_think(text)
            except Exception:
                answer, think = {"bbox": [0, 0, 0, 0], "points_1": [0, 0], "points_2": [0, 0]}, ""
            results.append(SimpleNamespace(output_text=text, answer=answer, think=think, generated_ids=seq.detach().cpu()))
        return results, outputs


def _load_stage1_model_and_sam_config(args: argparse.Namespace, device: torch.device) -> Tuple[Any, str, str]:
    stage1_export_dir = str(args.stage1_export_dir).strip()
    mllm_model_path = str(args.mllm_model_path).strip()
    if bool(stage1_export_dir) == bool(mllm_model_path):
        raise ValueError("Pass exactly one of --stage1-export-dir or --mllm-model-path.")

    if stage1_export_dir:
        export_dir = Path(stage1_export_dir)
        model = load_geometric_model_from_export(export_dir, device=device)
        sam_checkpoint_path, sam_model_cfg = resolve_export_sam_checkpoint(export_dir)
    else:
        sam_checkpoint_path = str(args.sam_checkpoint_path).strip()
        if not sam_checkpoint_path:
            raise ValueError("--sam-checkpoint-path is required when using --mllm-model-path.")
        model = PlainMLLMStage1Model(
            model_path=mllm_model_path,
            processor_path=str(args.processor_path).strip(),
            device=device,
            torch_dtype=_dtype_from_args(args),
            attn_implementation=str(args.attn_implementation),
        )
        sam_model_cfg = str(args.sam_model_cfg)
    return model, sam_checkpoint_path, sam_model_cfg


def _render_stage1_proposal_visualization(
    image_path: str,
    pred_bbox: Sequence[int],
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    try:
        canvas = image.copy()
    finally:
        image.close()
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = [int(v) for v in pred_bbox[:4]]
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
    canvas.save(output_path)
    canvas.close()


def _build_stage1_cache_record(
    sample: Dict[str, Any],
    prediction: Dict[str, Any],
    crop_path: Path,
    proposal_visualization_path: Path,
) -> Dict[str, Any]:
    bbox_iou = float(prediction.get("bbox_iou", 0.0))
    pred_mask_rle = prediction.get("pred_mask_rle")
    mask_iou = 0.0
    if pred_mask_rle is not None:
        pred_mask = sample["gt_mask"].copy()
        from training_scripts.refine_segzero.common import decode_rle_mask

        pred_mask = decode_rle_mask(pred_mask_rle)
        _, _, mask_iou = compute_iou(pred_mask, sample["gt_mask"])
    return {
        "image_id": sample["image_id"],
        "sample_id": sample["sample_id"],
        "ann_id": sample["sample_id"],
        "image_path": sample["image_path"],
        "question": sample["refexp"],
        "stage1_output_text": prediction.get("model_output_text", ""),
        "stage1_think": prediction.get("model_think", ""),
        "stage1_box": prediction.get("pred_bbox", [0, 0, 0, 0]),
        "stage1_points": prediction.get("pred_points", [[0, 0], [0, 0]]),
        "stage1_mask_rle": pred_mask_rle,
        "crop_image_path": str(crop_path),
        "proposal_visualization_path": str(proposal_visualization_path),
        "bbox_iou_stage1": bbox_iou,
        "mask_iou_stage1": float(mask_iou),
        "gt_box": sample["gt_box"],
        "gt_point": sample["gt_point"],
        "image_size": [sample["width"], sample["height"]],
    }


def _to_refine_sample(sample: Dict[str, Any]) -> RefineSample:
    return RefineSample(
        image_path=sample["image_path"],
        image_id=sample["image_id"],
        sample_id=sample["sample_id"],
        question_text=sample["question_text"],
        refexp=sample["refexp"],
        gt_mask=sample["gt_mask"],
        gt_box_xyxy=sample["gt_box"],
        width=sample["width"],
        height=sample["height"],
        raw_item={},
    )


def _run_stage1_inference_batch(
    model: Any,
    segmentation_model: Any,
    batch_samples: Sequence[Dict[str, Any]],
    resize_size: int,
    sam_image_size: int,
    max_new_tokens: int,
    max_pixels: int = 2007040,
    min_pixels: int = 3136,
) -> List[Dict[str, Any]]:
    refine_samples = [_to_refine_sample(sample) for sample in batch_samples]
    collated_batch = geometric_collate_fn(
        batch=refine_samples,
        processor=model.processor,
        resize_size=resize_size,
        sam_image_size=sam_image_size,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )
    model_inputs = {key: value.to(model.qwen.device) for key, value in collated_batch["model_inputs"].items()}
    with torch.inference_mode():
        generations = model._generate_once(model_inputs, max_new_tokens=max_new_tokens, do_sample=False)[0]

    predictions: List[Dict[str, Any]] = []
    for sample, generation in zip(batch_samples, generations):
        answer = generation.answer or {}
        pred_bbox = [0, 0, 0, 0]
        pred_points = [[0, 0], [0, 0]]
        pred_mask = np.zeros_like(sample["gt_mask"], dtype=np.uint8)
        error_message = ""
        try:
            bbox = answer.get("bbox", [0, 0, 0, 0])
            points_1 = answer.get("points_1", [0, 0])
            points_2 = answer.get("points_2", [0, 0])
            scale = float(resize_size) / float(max(sample["width"], sample["height"]))
            resized_width = max(1, int(round(sample["width"] * scale)))
            resized_height = max(1, int(round(sample["height"] * scale)))
            x_factor = sample["width"] / float(resized_width)
            y_factor = sample["height"] / float(resized_height)
            pred_bbox = [
                int(round(float(bbox[0]) * x_factor)),
                int(round(float(bbox[1]) * y_factor)),
                int(round(float(bbox[2]) * x_factor)),
                int(round(float(bbox[3]) * y_factor)),
            ]
            pred_points = [
                [int(round(float(points_1[0]) * x_factor)), int(round(float(points_1[1]) * y_factor))],
                [int(round(float(points_2[0]) * x_factor)), int(round(float(points_2[1]) * y_factor))],
            ]
            image = Image.open(sample["image_path"]).convert("RGB")
            try:
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=model.qwen.device.type == "cuda"):
                    segmentation_model.set_image(image)
                    masks, scores, _ = segmentation_model.predict(
                        point_coords=np.asarray(pred_points, dtype=np.float32),
                        point_labels=np.asarray([1, 1], dtype=np.int32),
                        box=np.asarray(pred_bbox, dtype=np.float32),
                    )
                sorted_idx = np.argsort(scores)[::-1]
                pred_mask = masks[sorted_idx][0].astype(np.uint8)
            finally:
                image.close()
        except Exception as exc:
            error_message = str(exc)
        _, _, mask_iou = compute_iou(pred_mask, sample["gt_mask"])
        predictions.append(
            {
                "image_id": sample["image_id"],
                "ann_id": sample["sample_id"],
                "question": sample["refexp"],
                "model_output_text": generation.output_text,
                "model_think": generation.think,
                "pred_bbox": pred_bbox,
                "pred_points": pred_points,
                "pred_mask_rle": encode_binary_mask(pred_mask),
                "bbox_iou": compute_box_iou(pred_bbox, sample["gt_box"]),
                "iou": float(mask_iou),
                "error": error_message,
            }
        )
    return predictions

def _build_unique_sample_image_name(sample: Dict[str, Any]) -> str:
    image_stem = Path(str(sample["image_id"])).stem
    sample_id = str(sample["sample_id"])
    refexp_hash = sha1(str(sample["refexp"]).encode("utf-8")).hexdigest()[:10]
    return f"{image_stem}__{sample_id}__{refexp_hash}.png"


def _run_split(
    split_name: str,
    json_paths: Sequence[str],
    model: Any,
    segmentation_model: Any,
    args: argparse.Namespace,
    output_dir: Path,
    idx: int,
    num_parts: int,
) -> Dict[str, int]:
    if not json_paths:
        return {"init_count": 0, "cache_count": 0}
    batch_size = max(int(args.batch_size), 1)
    record_batches, shard_len = iter_source_record_batches_shard(
        json_paths=json_paths,
        image_root=args.image_root,
        max_sample_ratio_per_file=args.max_sample_ratio_per_file,
        sample_seed=args.sample_seed,
        idx=idx,
        num_parts=num_parts,
        batch_size=batch_size,
    )
    crop_dir = output_dir / "stage1_crop_images" / split_name / f"part_{idx}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    proposal_vis_dir = output_dir / "stage1_proposal_visualizations" / split_name / f"part_{idx}"
    proposal_vis_dir.mkdir(parents=True, exist_ok=True)
    init_jsonl_path = output_dir / f"init_box_{split_name}_parts" / f"part_{idx}.jsonl.tmp"
    cache_jsonl_path = output_dir / f"stage1_cache_{split_name}_parts" / f"part_{idx}.jsonl.tmp"
    for tmp_path in (init_jsonl_path, cache_jsonl_path):
        if tmp_path.exists():
            tmp_path.unlink()
    start_time = time.time()
    progress = tqdm(
        total=shard_len,
        desc=f"stage1-{split_name}-rank{idx}",
        position=idx,
        leave=True,
        dynamic_ncols=True,
    )
    processed_count = 0
    for batch_samples in record_batches:
        for sample in batch_samples:
            append_jsonl(init_jsonl_path, build_init_box_record(sample))
        predictions = _run_stage1_inference_batch(
            model=model,
            segmentation_model=segmentation_model,
            batch_samples=batch_samples,
            resize_size=args.resize_size,
            sam_image_size=args.sam_image_size,
            max_new_tokens=args.max_new_tokens,
        )
        for sample, prediction in zip(batch_samples, predictions):
            pred_mask_rle = prediction.get("pred_mask_rle")
            unique_name = _build_unique_sample_image_name(sample)
            crop_path = crop_dir / unique_name
            proposal_visualization_path = proposal_vis_dir / unique_name
            if pred_mask_rle is not None:
                from training_scripts.refine_segzero.common import decode_rle_mask

                pred_mask = decode_rle_mask(pred_mask_rle)
                full_image = Image.open(sample["image_path"]).convert("RGB")
                try:
                    crop_image = build_mask_crop_image(full_image, pred_mask)
                finally:
                    full_image.close()
            else:
                x1, y1, x2, y2 = [int(v) for v in prediction.get("pred_bbox", [0, 0, 0, 0])]
                full_image = Image.open(sample["image_path"]).convert("RGB")
                try:
                    fallback_mask = sample["gt_mask"] * 0
                    if x2 >= x1 and y2 >= y1:
                        fallback_mask[y1 : y2 + 1, x1 : x2 + 1] = 1
                    crop_image = build_mask_crop_image(full_image, fallback_mask)
                finally:
                    full_image.close()
            crop_image.save(crop_path)
            crop_image.close()
            _render_stage1_proposal_visualization(
                image_path=sample["image_path"],
                pred_bbox=prediction.get("pred_bbox", [0, 0, 0, 0]),
                output_path=proposal_visualization_path,
            )
            append_jsonl(
                cache_jsonl_path,
                _build_stage1_cache_record(
                    sample,
                    prediction,
                    crop_path,
                    proposal_visualization_path,
                ),
            )
        processed_count += len(batch_samples)
        progress.update(len(batch_samples))
        elapsed = max(time.time() - start_time, 1e-6)
        progress.set_postfix(
            processed=processed_count,
            total=shard_len,
            batch=len(batch_samples),
            speed=f"{processed_count / elapsed:.2f}/s",
        )

    progress.close()

    init_count = jsonl_to_json(init_jsonl_path, output_dir / f"init_box_{split_name}_parts" / f"part_{idx}.json")
    cache_count = jsonl_to_json(cache_jsonl_path, output_dir / f"stage1_cache_{split_name}_parts" / f"part_{idx}.json")
    for tmp_path in (init_jsonl_path, cache_jsonl_path):
        if tmp_path.exists():
            tmp_path.unlink()
    return {"init_count": init_count, "cache_count": cache_count}


def main() -> None:
    args = parse_args()
    rank, world_size = env_rank()
    idx = args.idx if args.idx >= 0 else rank
    num_parts = args.num_parts if args.num_parts > 0 else world_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _wait_for_cleanup(output_dir, idx, args.overwrite_output)

    device = _device_for_rank(idx)
    model, sam_checkpoint_path, sam_model_cfg = _load_stage1_model_and_sam_config(args, device)
    segmentation_model = build_sam2_predictor(sam_checkpoint_path, sam_model_cfg, device)
    if hasattr(segmentation_model, "model"):
        segmentation_model.model = segmentation_model.model.to(device)

    train_stats = _run_split("train", args.train_json_paths, model, segmentation_model, args, output_dir, idx, num_parts)
    val_stats = _run_split("val", args.val_json_paths, model, segmentation_model, args, output_dir, idx, num_parts)
    write_json(
        output_dir / "stage1_part_summaries" / f"part_{idx}.json",
        {
            "rank": idx,
            "num_parts": num_parts,
            "device": str(device),
            "train": train_stats,
            "val": val_stats,
        },
    )
    write_json(
        output_dir / "stage1_done_markers" / f"part_{idx}.json",
        {
            "rank": idx,
            "num_parts": num_parts,
            "status": "done",
            "train": train_stats,
            "val": val_stats,
        },
    )

    if idx == 0:
        done_markers = [
            output_dir / "stage1_done_markers" / f"part_{part_idx}.json"
            for part_idx in range(num_parts)
        ]
        wait_for_parts(done_markers)
        init_train = merge_part_dir(output_dir / "init_box_train_parts", output_dir / "init_box_train.json") if args.train_json_paths else []
        init_val = merge_part_dir(output_dir / "init_box_val_parts", output_dir / "init_box_val.json") if args.val_json_paths else []
        cache_train = merge_part_dir(output_dir / "stage1_cache_train_parts", output_dir / "stage1_cache_train.json") if args.train_json_paths else []
        cache_val = merge_part_dir(output_dir / "stage1_cache_val_parts", output_dir / "stage1_cache_val.json") if args.val_json_paths else []
        write_json(
            output_dir / "stage1_data_summary.json",
            {
                "train_init_box": len(init_train),
                "val_init_box": len(init_val),
                "train_stage1_cache": len(cache_train),
                "val_stage1_cache": len(cache_val),
                "stage1_export_dir": args.stage1_export_dir,
            },
        )
        cleanup_marker = output_dir / "stage1_cleanup_done.json"
        if cleanup_marker.exists():
            cleanup_marker.unlink()


if __name__ == "__main__":
    main()
