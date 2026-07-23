import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from training_scripts.refine_segzero.Rl_data.rl_data_utils import (
    env_rank,
    merge_part_dir,
    sample_indices,
    wait_for_parts,
    write_json,
)
from training_scripts.refine_segzero.common import compute_box_iou, load_json_records
from training_scripts.refine_segzero.prompts import (
    QUERY_REFLECT_REFLECT_RL_ANSWER_EXAMPLE,
    build_query_reflect_reflect_rl_prompt,
)

ACCEPT_IOU_THRESHOLD = 0.6
REJECT_IOU_THRESHOLD = 0.2
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-train-paths", nargs="*", default=[])
    parser.add_argument("--stage1-val-paths", nargs="*", default=[])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--num-parts", type=int, default=-1)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--mllm-model-path", type=str, default="")
    parser.add_argument("--reflect-sample-count", type=int, default=16)
    parser.add_argument("--reflect-max-new-tokens", type=int, default=256)
    parser.add_argument("--reflect-temperature", type=float, default=1.2)
    parser.add_argument("--reflect-top-p", type=float, default=1.0)
    parser.add_argument("--debug-sample-output-limit", type=int, default=0)
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def _cleanup_output_dir(output_dir: Path) -> None:
    targets = [
        output_dir / "reflect_train_parts",
        output_dir / "reflect_val_parts",
        output_dir / "reflect_sample_debug_parts",
        output_dir / "reflect_part_summaries",
        output_dir / "reflect_done_markers",
        output_dir / "reflect_train.json",
        output_dir / "reflect_val.json",
        output_dir / "reflect_sample_debug.json",
        output_dir / "reflect_data_summary.json",
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _wait_for_cleanup(output_dir: Path, idx: int, overwrite_output: bool) -> None:
    if not overwrite_output:
        return
    marker = output_dir / "reflect_cleanup_done.json"
    if idx == 0:
        _cleanup_output_dir(output_dir)
        write_json(marker, {"status": "ready"})
    wait_for_parts([marker], timeout_s=60, stable_checks=1)


def _load_records(paths: Sequence[str]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for path in paths:
        if str(path).strip():
            merged.extend(load_json_records(str(path)))
    return merged


def _device_for_rank(rank: int) -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        device_count = max(torch.cuda.device_count(), 1)
        device = torch.device(f"cuda:{local_rank % device_count}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def _load_mllm(model_path: str, device: torch.device, attn_implementation: str) -> tuple[Any, Any]:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation=attn_implementation,
        device_map=None,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    return model, processor


def _parse_reflect_decision_with_error(output_text: str) -> tuple[str | None, str]:
    match = ANSWER_RE.search(output_text)
    if not match:
        return None, "missing_answer"
    try:
        answer = json.loads(match.group(1).strip())
    except Exception:
        return None, "invalid_json"
    if not isinstance(answer, dict) or set(answer.keys()) != {"decision"}:
        return None, "invalid_schema"
    decision = str(answer.get("decision", "")).strip().lower()
    if decision not in {"accept", "reject"}:
        return None, "invalid_decision"
    return decision, ""


def _parse_reflect_decision(output_text: str) -> str | None:
    decision, _ = _parse_reflect_decision_with_error(output_text)
    return decision


def _build_reflect_messages(image: Image.Image, prompt_override: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": str(prompt_override)},
            ],
        }
    ]


def _sample_reflect_counts(
    model: Any,
    processor: Any,
    record: Dict[str, Any],
    target_decision: str,
    sample_count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    collect_debug: bool = False,
) -> tuple[Dict[str, int], Dict[str, Any] | None]:
    sample_count = max(int(sample_count), 0)
    if sample_count == 0:
        return {
            "reflect_sample_parse_error_count": 0,
            "reflect_sample_correct_count": 0,
            "reflect_sample_wrong_count": 0,
        }, None

    image_path = str((record.get("image_paths") or [""])[0])
    parse_error_count = 0
    correct_count = 0
    wrong_count = 0

    try:
        model_device = next(model.parameters()).device
        image = Image.open(image_path).convert("RGB")
        try:
            messages = _build_reflect_messages(image, str(record.get("prompt_override", "")))
            texts = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
            image_inputs, video_inputs = process_vision_info([messages])
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(model_device) if torch.is_tensor(value) else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    num_return_sequences=sample_count,
                )
            input_len = int(inputs["input_ids"].shape[1])
            generated_ids_trimmed = generated_ids[:, input_len:]
            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        finally:
            image.close()
    except Exception as exc:
        counts = {
            "reflect_sample_parse_error_count": sample_count,
            "reflect_sample_correct_count": 0,
            "reflect_sample_wrong_count": 0,
        }
        debug_entry = None
        if collect_debug:
            debug_entry = {
                "meta_image_id": record.get("meta_image_id", ""),
                "meta_sample_id": record.get("meta_sample_id", ""),
                "problem": record.get("problem", ""),
                "image_path": image_path,
                "target_decision": target_decision,
                "counts": counts,
                "error": str(exc),
                "log": f"sample_id={record.get('meta_sample_id', '')} sampling_failed parse_error={sample_count} error={exc}",
                "outputs": [],
            }
        return counts, debug_entry

    debug_outputs: List[Dict[str, Any]] = []
    for output_idx, output_text in enumerate(output_texts):
        decision, parse_error = _parse_reflect_decision_with_error(output_text)
        is_correct = bool(decision == target_decision) if decision is not None else False
        if decision is None:
            parse_error_count += 1
        elif is_correct:
            correct_count += 1
        else:
            wrong_count += 1
        if collect_debug:
            debug_outputs.append(
                {
                    "index": int(output_idx),
                    "output_text": output_text,
                    "parsed_decision": decision,
                    "parse_error": parse_error,
                    "is_correct": is_correct,
                }
            )

    counts = {
        "reflect_sample_parse_error_count": int(parse_error_count),
        "reflect_sample_correct_count": int(correct_count),
        "reflect_sample_wrong_count": int(wrong_count),
    }
    debug_entry = None
    if collect_debug:
        debug_entry = {
            "meta_image_id": record.get("meta_image_id", ""),
            "meta_sample_id": record.get("meta_sample_id", ""),
            "problem": record.get("problem", ""),
            "image_path": image_path,
            "target_decision": target_decision,
            "counts": counts,
            "log": (
                f"sample_id={record.get('meta_sample_id', '')} target={target_decision} "
                f"correct={correct_count} wrong={wrong_count} parse_error={parse_error_count}"
            ),
            "outputs": debug_outputs,
        }
    return counts, debug_entry


def _append_debug_record(
    debug_payload: List[Dict[str, Any]],
    debug_entry: Dict[str, Any] | None,
    output_path: Path,
) -> None:
    if debug_entry is None:
        return
    debug_payload.append(debug_entry)
    write_json(output_path, debug_payload)


def _summarize_decisions(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    accept_count = 0
    reject_count = 0
    for record in records:
        try:
            solution = json.loads(str(record.get("solution", "{}")))
        except Exception:
            solution = {}
        decision = str(solution.get("target_decision", "")).strip().lower()
        if decision == "accept":
            accept_count += 1
        elif decision == "reject":
            reject_count += 1
    return {
        "accept_count": accept_count,
        "reject_count": reject_count,
    }


def build_reflect_record(stage1_payload: Dict[str, Any], confidence_threshold: float) -> Dict[str, Any] | None:
    pred_box = [int(v) for v in stage1_payload.get("stage1_box", [0, 0, 0, 0])]
    pred_points = stage1_payload.get("stage1_points", [[0, 0], [0, 0]])
    gt_box = [int(v) for v in stage1_payload.get("gt_box", [0, 0, 0, 0])]
    bbox_iou = float(stage1_payload.get("bbox_iou_stage1", compute_box_iou(pred_box, gt_box)))
    if bbox_iou >= ACCEPT_IOU_THRESHOLD:
        target_decision = "accept"
    elif bbox_iou < REJECT_IOU_THRESHOLD:
        target_decision = "reject"
    else:
        return None
    refexp = str(stage1_payload.get("question", ""))
    prompt_override = build_query_reflect_reflect_rl_prompt(question=refexp)
    solution = {
        "task_type": "reflect",
        "target_decision": target_decision,
        "accept_iou_threshold": ACCEPT_IOU_THRESHOLD,
        "reject_iou_threshold": REJECT_IOU_THRESHOLD,
        "bbox_iou_stage1": bbox_iou,
        "stage1_box": pred_box,
        "image_id": stage1_payload.get("image_id", ""),
        "sample_id": stage1_payload.get("sample_id", ""),
    }
    return {
        "problem": refexp,
        "solution": json.dumps(solution, ensure_ascii=False),
        "task_type": "reflect",
        "data_source": "reflect",
        "image_paths": [
            stage1_payload.get("proposal_visualization_path", ""),
        ],
        "prompt_override": prompt_override,
        "answer_format_example": QUERY_REFLECT_REFLECT_RL_ANSWER_EXAMPLE,
        "meta_image_id": stage1_payload.get("image_id", ""),
        "meta_sample_id": stage1_payload.get("sample_id", ""),
        "stage1_output_text": stage1_payload.get("stage1_output_text", ""),
        "stage1_think": stage1_payload.get("stage1_think", ""),
        "stage1_box": pred_box,
        "stage1_points": pred_points,
        "stage1_mask_rle": stage1_payload.get("stage1_mask_rle"),
    }

def _build_split(
    split_name: str,
    records: Sequence[Dict[str, Any]],
    output_dir: Path,
    idx: int,
    num_parts: int,
    confidence_threshold: float,
    model: Any,
    processor: Any,
    reflect_sample_count: int,
    reflect_max_new_tokens: int,
    reflect_temperature: float,
    reflect_top_p: float,
    debug_sample_output_limit: int,
) -> Dict[str, int]:
    if not records:
        return {
            "kept_records": 0,
            "accept_count": 0,
            "reject_count": 0,
            "filtered_mid_quality_count": 0,
        }
    shard = sample_indices(len(records), idx, num_parts)
    shard_indices = list(shard)
    payload: List[Dict[str, Any]] = []
    debug_payload: List[Dict[str, Any]] = []
    debug_output_path = output_dir / "reflect_sample_debug_parts" / f"part_{idx}_{split_name}.json"
    accept_count = 0
    reject_count = 0
    filtered_mid_quality_count = 0
    start_time = time.time()
    progress = tqdm(
        shard_indices,
        desc=f"reflect-{split_name}-rank{idx}",
        total=len(shard_indices),
        position=idx,
        leave=True,
        dynamic_ncols=True,
    )
    for processed_count, record_idx in enumerate(progress, start=1):
        record = build_reflect_record(records[record_idx], confidence_threshold)
        if record is None:
            filtered_mid_quality_count += 1
        else:
            solution = json.loads(str(record["solution"]))
            decision = str(solution.get("target_decision", "")).strip().lower()
            counts, debug_entry = _sample_reflect_counts(
                model=model,
                processor=processor,
                record=record,
                target_decision=decision,
                sample_count=reflect_sample_count,
                max_new_tokens=reflect_max_new_tokens,
                temperature=reflect_temperature,
                top_p=reflect_top_p,
                collect_debug=len(debug_payload) < int(debug_sample_output_limit),
            )
            record.update(counts)
            _append_debug_record(debug_payload, debug_entry, debug_output_path)
            payload.append(record)
            if decision == "accept":
                accept_count += 1
            elif decision == "reject":
                reject_count += 1
        elapsed = max(time.time() - start_time, 1e-6)
        progress.set_postfix(
            processed=processed_count,
            total=len(shard_indices),
            kept=len(payload),
            accept=accept_count,
            reject=reject_count,
            filtered=filtered_mid_quality_count,
            speed=f"{processed_count / elapsed:.2f}/s",
        )
    progress.close()
    write_json(output_dir / f"reflect_{split_name}_parts" / f"part_{idx}.json", payload)
    return {
        "kept_records": len(payload),
        "accept_count": accept_count,
        "reject_count": reject_count,
        "filtered_mid_quality_count": filtered_mid_quality_count,
    }


def main() -> None:
    args = parse_args()
    rank, world_size = env_rank()
    idx = args.idx if args.idx >= 0 else rank
    num_parts = args.num_parts if args.num_parts > 0 else world_size
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _wait_for_cleanup(output_dir, idx, args.overwrite_output)

    if not str(args.mllm_model_path).strip():
        raise ValueError("--mllm-model-path is required for reflect sampling.")
    device = _device_for_rank(idx)
    model, processor = _load_mllm(args.mllm_model_path, device=device, attn_implementation=args.attn_implementation)

    train_records = _load_records(args.stage1_train_paths)
    val_records = _load_records(args.stage1_val_paths)
    train_stats = _build_split(
        "train",
        train_records,
        output_dir,
        idx,
        num_parts,
        args.confidence_threshold,
        model,
        processor,
        args.reflect_sample_count,
        args.reflect_max_new_tokens,
        args.reflect_temperature,
        args.reflect_top_p,
        args.debug_sample_output_limit,
    )
    val_stats = _build_split(
        "val",
        val_records,
        output_dir,
        idx,
        num_parts,
        args.confidence_threshold,
        model,
        processor,
        args.reflect_sample_count,
        args.reflect_max_new_tokens,
        args.reflect_temperature,
        args.reflect_top_p,
        args.debug_sample_output_limit,
    )
    write_json(
        output_dir / "reflect_part_summaries" / f"part_{idx}.json",
        {
            "rank": idx,
            "num_parts": num_parts,
            "accept_threshold": ACCEPT_IOU_THRESHOLD,
            "reject_threshold": REJECT_IOU_THRESHOLD,
            "train": train_stats,
            "val": val_stats,
        },
    )
    write_json(
        output_dir / "reflect_done_markers" / f"part_{idx}.json",
        {
            "rank": idx,
            "num_parts": num_parts,
            "status": "done",
            "accept_threshold": ACCEPT_IOU_THRESHOLD,
            "reject_threshold": REJECT_IOU_THRESHOLD,
            "train": train_stats,
            "val": val_stats,
        },
    )

    if idx == 0:
        done_markers = [
            output_dir / "reflect_done_markers" / f"part_{part_idx}.json"
            for part_idx in range(num_parts)
        ]
        wait_for_parts(done_markers)
        merged_train = merge_part_dir(output_dir / "reflect_train_parts", output_dir / "reflect_train.json") if train_records else []
        merged_val = merge_part_dir(output_dir / "reflect_val_parts", output_dir / "reflect_val.json") if val_records else []
        if args.debug_sample_output_limit > 0:
            debug_records = merge_part_dir(
                output_dir / "reflect_sample_debug_parts",
                output_dir / "reflect_sample_debug.json",
            )
            write_json(
                output_dir / "reflect_sample_debug.json",
                debug_records[: int(args.debug_sample_output_limit)],
            )
        train_decisions = _summarize_decisions(merged_train)
        val_decisions = _summarize_decisions(merged_val)
        write_json(
            output_dir / "reflect_data_summary.json",
            {
                "train_records": len(merged_train),
                "val_records": len(merged_val),
                "accept_threshold": ACCEPT_IOU_THRESHOLD,
                "reject_threshold": REJECT_IOU_THRESHOLD,
                "train_accept_count": train_decisions["accept_count"],
                "train_reject_count": train_decisions["reject_count"],
                "val_accept_count": val_decisions["accept_count"],
                "val_reject_count": val_decisions["reject_count"],
                "train_filtered_mid_quality_count": max(
                    len(train_records) - len(merged_train),
                    0,
                ),
                "val_filtered_mid_quality_count": max(
                    len(val_records) - len(merged_val),
                    0,
                ),
                "legacy_confidence_threshold_arg": args.confidence_threshold,
            },
        )
        cleanup_marker = output_dir / "reflect_cleanup_done.json"
        if cleanup_marker.exists():
            cleanup_marker.unlink()


if __name__ == "__main__":
    main()
