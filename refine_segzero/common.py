import json
import math
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


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


def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": list(rle["size"]), "counts": counts}


def mask_to_box(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[int, int, float]:
    intersection = int(np.logical_and(mask_a > 0, mask_b > 0).sum())
    union = int(np.logical_or(mask_a > 0, mask_b > 0).sum())
    iou = 0.0 if union == 0 else float(intersection) / float(union)
    return intersection, union, iou


def compute_box_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = [int(v) for v in box_a]
    bx1, by1, bx2, by2 = [int(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    area_a = max(ax2 - ax1 + 1, 0) * max(ay2 - ay1 + 1, 0)
    area_b = max(bx2 - bx1 + 1, 0) * max(by2 - by1 + 1, 0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def normalized_box_l1(pred_box: Sequence[int], gt_box: Sequence[int], width: int, height: int) -> float:
    norm = [max(width, 1), max(height, 1), max(width, 1), max(height, 1)]
    return float(sum(abs(float(a) - float(b)) / float(n) for a, b, n in zip(pred_box, gt_box, norm)) / 4.0)


def dice_loss(pred_masks: torch.Tensor, gt_masks: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    pred_probs = torch.sigmoid(pred_masks)
    numerator = 2 * (pred_probs * gt_masks).sum(dim=(-1, -2))
    denominator = pred_probs.sum(dim=(-1, -2)) + gt_masks.sum(dim=(-1, -2))
    return 1 - (numerator + eps) / (denominator + eps)


def resize_longest_side(image: Image.Image, longest_side: int) -> Image.Image:
    width, height = image.size
    scale = float(longest_side) / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.BILINEAR)


def build_sam_image_tensor(image: Image.Image, sam_image_size: int) -> torch.Tensor:
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(-1, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(-1, 1, 1)
    resized = image.resize((sam_image_size, sam_image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - pixel_mean) / pixel_std


def extract_refexp(question_text: str) -> str:
    match = re.search(r'describes:\s*"(.+?)"\.?$', question_text.strip())
    if match:
        return match.group(1)
    return question_text.strip()


def load_json_records(path: str) -> List[Dict[str, Any]]:
    json_path = Path(path)
    with open(json_path, "r", encoding="utf-8") as f:
        if json_path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        first = ""
        for line in f:
            if line.strip():
                first = line.lstrip()
                break
        f.seek(0)
        if first.startswith("{"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


class ImagePathResolver:
    def __init__(self, image_root: str):
        self.image_root = Path(image_root)
        self._stem_to_path: Dict[str, str] = {}
        self._name_to_path: Dict[str, str] = {}
        for path in self.image_root.rglob("*.jpg"):
            if path.is_file():
                self._stem_to_path.setdefault(path.stem, str(path))
                self._name_to_path.setdefault(path.name, str(path))

    def resolve(self, image_id: str) -> str:
        image_id_path = Path(image_id)
        if image_id_path.is_absolute() and image_id_path.exists():
            return str(image_id_path)
        direct = self.image_root / image_id
        if direct.exists():
            return str(direct)
        if image_id in self._name_to_path:
            return self._name_to_path[image_id]
        stem = Path(image_id).stem
        if stem in self._stem_to_path:
            return self._stem_to_path[stem]
        raise FileNotFoundError(f"Image {image_id!r} not found under {self.image_root}")


@dataclass
class RefineSample:
    image_path: str
    image_id: str
    sample_id: str
    question_text: str
    refexp: str
    gt_mask: np.ndarray
    gt_box_xyxy: List[int]
    width: int
    height: int
    raw_item: Dict[str, Any]


def extract_answer_json_and_think(output_text: str) -> Tuple[Dict[str, Any], str, Tuple[int, int]]:
    answer_match = ANSWER_RE.search(output_text)
    if not answer_match:
        raise ValueError(f"Missing <answer> block: {output_text}")
    answer_text = answer_match.group(1).strip()
    answer = json.loads(answer_text)
    think_match = THINK_RE.search(output_text)
    think = think_match.group(1).strip() if think_match else ""
    return answer, think, answer_match.span(1)


def answer_to_prompts(answer: Dict[str, Any]) -> Tuple[List[int], List[List[int]]]:
    bbox = [int(round(float(v))) for v in answer["bbox"]]
    points = [
        [int(round(float(v))) for v in answer["points_1"]],
        [int(round(float(v))) for v in answer["points_2"]],
    ]
    return bbox, points


def scale_answer_to_image(answer: Dict[str, Any], width: int, height: int, resize_size: int) -> Dict[str, Any]:
    x_factor = width / float(resize_size)
    y_factor = height / float(resize_size)
    return {
        "bbox": [
            int(round(float(answer["bbox"][0]) * x_factor)),
            int(round(float(answer["bbox"][1]) * y_factor)),
            int(round(float(answer["bbox"][2]) * x_factor)),
            int(round(float(answer["bbox"][3]) * y_factor)),
        ],
        "points_1": [
            int(round(float(answer["points_1"][0]) * x_factor)),
            int(round(float(answer["points_1"][1]) * y_factor)),
        ],
        "points_2": [
            int(round(float(answer["points_2"][0]) * x_factor)),
            int(round(float(answer["points_2"][1]) * y_factor)),
        ],
    }


def answer_json_string(answer: Dict[str, Any]) -> str:
    return json.dumps(answer, ensure_ascii=False)


def tokenize_answer_span(tokenizer: Any, output_text: str, answer_span: Tuple[int, int]) -> Tuple[List[int], bool]:
    tokenized = tokenizer(output_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = tokenized["offset_mapping"]
    token_indices = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > answer_span[0] and start < answer_span[1]
    ]
    return token_indices, len(token_indices) > 0


def find_subsequence(sequence: Sequence[int], target: Sequence[int]) -> Tuple[int, int]:
    if not sequence or not target or len(target) > len(sequence):
        return -1, -1
    target_len = len(target)
    for start_idx in range(len(sequence) - target_len + 1):
        if list(sequence[start_idx : start_idx + target_len]) == list(target):
            return start_idx, start_idx + target_len
    return -1, -1


def build_prediction_record(
    image_id: str,
    sample_id: str,
    question: str,
    output_text: str,
    think: str,
    pred_bbox: List[int],
    pred_points: List[List[int]],
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    branch_type: str = "",
    extra: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Dict[str, Any]:
    intersection, union, iou = compute_iou(pred_mask, gt_mask)
    gt_bbox = mask_to_box(gt_mask)
    payload = {
        "image_id": image_id,
        "ann_id": sample_id,
        "question": question,
        "model_output_text": output_text,
        "model_think": think,
        "pred_bbox": pred_bbox,
        "pred_points": pred_points,
        "gt_bbox": gt_bbox,
        "bbox_iou": compute_box_iou(pred_bbox, gt_bbox),
        "pred_mask_rle": encode_binary_mask(pred_mask),
        "gt_mask_rle": encode_binary_mask(gt_mask),
        "intersection": int(intersection),
        "union": int(union),
        "iou": float(iou),
        "branch_type": branch_type,
        "error": error,
    }
    if extra:
        payload.update(extra)
    return payload


def save_metrics_history(output_dir: Path, history: Dict[str, List[Dict[str, float]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def append_metric(history: Dict[str, List[Dict[str, float]]], name: str, step: int, value: float) -> None:
    history.setdefault(name, []).append({"step": int(step), "value": float(value)})



def trainable_state_has_learnable_query(state_dict: Dict[str, Any]) -> bool:
    return (
        "learnable_query" in state_dict
        or "aligned_learnable_query" in state_dict
        or "direct_learnable_query" in state_dict
    )


def warn_if_missing_learnable_query(state_dict: Dict[str, Any], source: str) -> None:
    if trainable_state_has_learnable_query(state_dict):
        return
    warnings.warn(
        (
            f"{source} is missing `learnable_query`. "
            "This export/checkpoint will fall back to a randomly initialized query, "
            "and refine_segzero direct/aligned branch metrics may be severely distorted."
        ),
        RuntimeWarning,
        stacklevel=2,
    )


def save_export_metadata(
    export_dir: Path,
    metadata: Dict[str, Any],
    trainable_state_dict: Dict[str, torch.Tensor],
    processor: Any,
    qwen: Any,
    sam_state_dict: Dict[str, torch.Tensor],
    copy_sam_checkpoint: bool = False,
    qwen_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    mllm_dir = export_dir / "mllm"
    trainable_dir = export_dir / "trainable"
    sam2_dir = export_dir / "sam2"
    mllm_dir.mkdir(parents=True, exist_ok=True)
    trainable_dir.mkdir(parents=True, exist_ok=True)
    sam2_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"safe_serialization": False}
    if qwen_state_dict is not None:
        bad_qwen_keys = [
            str(key)
            for key in qwen_state_dict
            if "lora_" in str(key).lower() or ".base_layer." in str(key)
        ]
        if bad_qwen_keys:
            raise ValueError(f"Refusing to export non-clean base mllm state; examples={bad_qwen_keys[:5]}")
        save_kwargs["state_dict"] = qwen_state_dict
    qwen.save_pretrained(mllm_dir, **save_kwargs)
    processor.save_pretrained(mllm_dir)
    torch.save(trainable_state_dict, trainable_dir / "trainable_state_dict.pt")
    torch.save(sam_state_dict, sam2_dir / "sam2_state_dict.pt")
    with open(export_dir / "export_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(sam2_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sam_model_cfg": metadata["sam_model_cfg"],
                "sam_checkpoint_path": metadata["sam_checkpoint_path"],
                "copied_checkpoint": None,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if copy_sam_checkpoint:
        source = Path(str(metadata["sam_checkpoint_path"]))
        if source.is_file():
            copied = sam2_dir / source.name
            shutil.copy2(source, copied)
            with open(sam2_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sam_model_cfg": metadata["sam_model_cfg"],
                        "sam_checkpoint_path": metadata["sam_checkpoint_path"],
                        "copied_checkpoint": str(copied),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )


def resolve_export_sam_checkpoint(export_dir: Path) -> Tuple[str, str]:
    with open(export_dir / "sam2" / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return str(metadata.get("copied_checkpoint") or metadata["sam_checkpoint_path"]), str(metadata["sam_model_cfg"])


def dtype_from_config(cfg: Any) -> torch.dtype:
    if bool(cfg.get("bf16", True)):
        return torch.bfloat16
    if bool(cfg.get("fp16", False)):
        return torch.float16
    return torch.float32


def summarize_mask(mask: np.ndarray) -> str:
    box = mask_to_box(mask)
    area = int(mask.sum())
    return f"mask_box={box}, mask_area={area}"


def render_mask_overlay_image(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB")
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_uint8, mode="L").resize(base.size, Image.NEAREST)
    red_overlay = Image.new("RGB", base.size, (255, 64, 64))
    blended = Image.blend(base, red_overlay, float(alpha))
    return Image.composite(blended, base, mask_image)


def build_mask_crop_image(
    image: Image.Image,
    mask: np.ndarray,
    padding_ratio: float = 0.1,
    min_crop_size: int = 32,
) -> Image.Image:
    box = mask_to_box(mask)
    if box == [0, 0, 0, 0]:
        return image.convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in box]
    width, height = image.size
    box_w = max(1, x2 - x1 + 1)
    box_h = max(1, y2 - y1 + 1)
    pad_x = max(int(round(box_w * float(padding_ratio))), int(min_crop_size // 4))
    pad_y = max(int(round(box_h * float(padding_ratio))), int(min_crop_size // 4))
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x + 1)
    crop_y2 = min(height, y2 + pad_y + 1)
    crop = image.crop((crop_x1, crop_y1, crop_x2, crop_y2)).convert("RGB")
    if min(crop.size) >= int(min_crop_size):
        return crop
    target_w = max(int(min_crop_size), crop.size[0])
    target_h = max(int(min_crop_size), crop.size[1])
    padded = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    offset = ((target_w - crop.size[0]) // 2, (target_h - crop.size[1]) // 2)
    padded.paste(crop, offset)
    return padded


def mean_of(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / max(len(values), 1))


def kl_divergence_with_temperature(student: torch.Tensor, teacher: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    student_log_probs = F.log_softmax(student / temperature, dim=-1)
    teacher_probs = F.softmax(teacher.detach() / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def finite_or_zero(value: torch.Tensor) -> torch.Tensor:
    if torch.isfinite(value).all():
        return value
    return torch.zeros_like(value)
