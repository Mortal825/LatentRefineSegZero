import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from training_scripts.eval.common import build_prediction_record
from training_scripts.eval.query_sam_lib import mask_to_box

QUERY_SINGLE_OBJECT_TEMPLATE = (
    "Please find '{Question}' with bbox and points."
    "Compare the difference between objects and find the most closely matched one."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format."
    "i.e., <think> thinking process here </think>"
    "<answer>{Answer}</answer>"
)


def ensure_rle_counts_bytes(rle: Dict[str, Any]) -> Dict[str, Any]:
    fixed_rle = dict(rle)
    counts = fixed_rle.get("counts")
    if isinstance(counts, str):
        fixed_rle["counts"] = counts.encode("utf-8")
    return fixed_rle


def decode_rle_mask(rle: Dict[str, Any]) -> np.ndarray:
    mask = mask_utils.decode(ensure_rle_counts_bytes(rle))
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def box_xyxy_from_normalized(box: List[float], width: int, height: int) -> List[int]:
    return [
        int(round(box[0] * width)),
        int(round(box[1] * height)),
        int(round(box[2] * width)),
        int(round(box[3] * height)),
    ]


def extract_refexp(question_text: str) -> str:
    match = re.search(r'describes:\s*"(.+?)"\.?$', question_text.strip())
    if match:
        return match.group(1)
    return question_text.strip()


def load_refcocog_records(ref_json_path: str) -> List[Dict[str, Any]]:
    path = Path(ref_json_path)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        first_nonempty = ""
        for line in f:
            if line.strip():
                first_nonempty = line.lstrip()
                break
        f.seek(0)
        if first_nonempty.startswith("{"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def get_refcocog_sample(sample: Dict[str, Any], image_root: str) -> Dict[str, Any]:
    image_path = Path(image_root) / sample["image"]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    question_text = sample["conversations"][0]["value"]
    refexp = extract_refexp(question_text)
    gt_mask = decode_rle_mask(sample["objects"][0]["rle"])
    return {
        "image": image,
        "image_id": sample["image"],
        "ann_id": str(sample["id"]),
        "question": refexp,
        "question_text": question_text,
        "gt_mask": gt_mask,
        "gt_bbox": mask_to_box(gt_mask),
        "width": width,
        "height": height,
    }


def build_refcocog_messages(image: Image.Image, question: str, resize_size: int) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.resize((resize_size, resize_size), Image.BILINEAR)},
                {
                    "type": "text",
                    "text": QUERY_SINGLE_OBJECT_TEMPLATE.format(
                        Question=question.lower().strip("."),
                        Answer='{"bbox": [10,100,200,210], "points_1": [30,110], "points_2": [35,180]}',
                    ),
                },
            ],
        }
    ]


def build_refcocog_prediction_record(
    sample_info: Dict[str, Any],
    output_text: str,
    think: str,
    pred_bbox: List[int],
    pred_points: List[List[int]],
    pred_mask: np.ndarray,
    error: str = "",
) -> Dict[str, Any]:
    record = build_prediction_record(
        image_id=sample_info["image_id"],
        ann_id=sample_info["ann_id"],
        question=sample_info["question"],
        output_text=output_text,
        think=think,
        pred_bbox=pred_bbox,
        pred_points=pred_points,
        pred_mask=pred_mask,
        gt_mask=np.asarray(sample_info["gt_mask"], dtype=np.uint8),
        error=error,
    )
    return record
