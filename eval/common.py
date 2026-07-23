import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image


from training_scripts.eval.query_sam_lib import (
    DEFAULT_SYSTEM_PROMPT,
    ReasonSegEvalDataset,
    SegZeroOfflineSegDataset,
    build_generation_messages,
    build_teacher_forced_inputs,
    compute_box_iou,
    compute_iou,
    encode_binary_mask,
    extract_bbox_points_think,
    load_eval_dataset,
    mask_to_box,
)

def split_indices(total_len: int, idx: int, num_parts: int) -> range:
    part_size = total_len // num_parts
    start_idx = idx * part_size
    end_idx = start_idx + part_size if idx < num_parts - 1 else total_len
    return range(start_idx, end_idx)


def ensure_output_dir(output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_prediction_part(output_dir: str, idx: int, results: List[Dict[str, Any]]) -> Path:
    output_path = ensure_output_dir(output_dir) / f"output_{idx}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_path


def get_sample_fields(dataset: Any, sample: Any, index: int) -> Tuple[Image.Image, int, int, str, np.ndarray, str, str]:
    if isinstance(dataset, ReasonSegEvalDataset):
        return (
            sample.image,
            sample.img_width,
            sample.img_height,
            sample.question,
            sample.gt_mask,
            sample.image_id,
            sample.ann_id,
        )
    image = Image.open(sample.image_path).convert("RGB")
    width, height = image.size
    return (
        image,
        width,
        height,
        sample.question,
        sample.gt_mask,
        sample.image_id,
        sample.raw_item.get("ann_id", str(index)),
    )


def build_prediction_record(
    image_id: str,
    ann_id: str,
    question: str,
    output_text: str,
    think: str,
    pred_bbox: List[int],
    pred_points: List[List[int]],
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    error: str = "",
) -> Dict[str, Any]:
    intersection, union, iou = compute_iou(pred_mask, gt_mask)
    gt_bbox = mask_to_box(gt_mask)
    bbox_iou = compute_box_iou(pred_bbox, gt_bbox)
    return {
        "image_id": image_id,
        "ann_id": ann_id,
        "question": question,
        "model_output_text": output_text,
        "model_think": think,
        "pred_bbox": pred_bbox,
        "pred_points": pred_points,
        "gt_bbox": gt_bbox,
        "bbox_iou": bbox_iou,
        "pred_mask_rle": encode_binary_mask(pred_mask),
        "gt_mask_rle": encode_binary_mask(gt_mask),
        "intersection": int(intersection),
        "union": int(union),
        "iou": float(iou),
        "error": error,
    }
