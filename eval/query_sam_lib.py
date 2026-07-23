import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from verl.models.transformers.qwen2_5_vl import get_rope_index


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful visual reasoning assistant. Read the image and the existing reasoning, "
    "then use them as context for segmentation refinement."
)
DEFAULT_SAM2_CONFIG = "sam2_hiera_l.yaml"
QUESTION_TEMPLATE = (
    "Please find '{Question}' with bbox and points."
    "Compare the difference between objects and find the most closely matched one."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format."
    "i.e., <think> thinking process here </think>"
    "<answer>{Answer}</answer>"
)


def resolve_model_path(base_model_path: str, segzero_checkpoint_path: Optional[str]) -> str:
    if segzero_checkpoint_path:
        checkpoint_path = Path(segzero_checkpoint_path)
        huggingface_path = checkpoint_path / "huggingface"
        if huggingface_path.exists():
            return str(huggingface_path)
        if checkpoint_path.exists():
            return str(checkpoint_path)
    return base_model_path


def ensure_rle_counts_bytes(rle: Dict[str, Any]) -> Dict[str, Any]:
    fixed_rle = dict(rle)
    counts = fixed_rle.get("counts")
    if isinstance(counts, str):
        fixed_rle["counts"] = counts.encode("utf-8")
    return fixed_rle


def decode_rle_mask(rle: Dict[str, Any]) -> np.ndarray:
    fixed_rle = ensure_rle_counts_bytes(rle)
    mask = mask_utils.decode(fixed_rle)
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


def compute_box_iou(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    area_a = max(ax2 - ax1 + 1, 0) * max(ay2 - ay1 + 1, 0)
    area_b = max(bx2 - bx1 + 1, 0) * max(by2 - by1 + 1, 0)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area) / float(union)


def resize_longest_side(image: Image.Image, longest_side: int) -> Image.Image:
    width, height = image.size
    scale = float(longest_side) / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.BILINEAR)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[int, int, float]:
    intersection = int(np.logical_and(mask_a > 0, mask_b > 0).sum())
    union = int(np.logical_or(mask_a > 0, mask_b > 0).sum())
    iou = 0.0 if union == 0 else float(intersection) / float(union)
    return intersection, union, iou


def build_answer_json(model_parsed_answer: Dict[str, Any]) -> str:
    answer = {
        "bbox": model_parsed_answer["bbox"],
        "points_1": model_parsed_answer["points_1"],
        "points_2": model_parsed_answer["points_2"],
    }
    return json.dumps(answer, ensure_ascii=False)


def extract_bbox_points_think(
    output_text: str,
    x_factor: float,
    y_factor: float,
) -> Tuple[List[int], List[List[int]], str, Dict[str, Any]]:
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL)
    if not answer_match:
        raise ValueError(f"Failed to parse <answer> block: {output_text}")
    answer = json.loads(answer_match.group(1))
    bbox = answer["bbox"]
    pred_bbox = [
        int(round(float(bbox[0]) * x_factor)),
        int(round(float(bbox[1]) * y_factor)),
        int(round(float(bbox[2]) * x_factor)),
        int(round(float(bbox[3]) * y_factor)),
    ]
    pred_points = [
        [
            int(round(float(answer["points_1"][0]) * x_factor)),
            int(round(float(answer["points_1"][1]) * y_factor)),
        ],
        [
            int(round(float(answer["points_2"][0]) * x_factor)),
            int(round(float(answer["points_2"][1]) * y_factor)),
        ],
    ]
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""
    return pred_bbox, pred_points, think, answer


def build_generation_messages(image: Image.Image, question: str, resize_size: int) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.resize((resize_size, resize_size), Image.BILINEAR)},
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(
                        Question=question.lower().strip("."),
                        Answer="{'bbox': [10,100,200,210], 'points_1': [30,110], 'points_2': [35,180]}",
                    ),
                },
            ],
        }
    ]


def build_teacher_forced_inputs(
    processor: AutoProcessor,
    image: Image.Image,
    question: str,
    think: str,
    answer: Dict[str, Any],
    resize_size: int,
    max_pixels: int,
    min_pixels: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dict[str, torch.Tensor]:
    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.min_pixels = min_pixels
    assistant_content = f"<think>{think}</think>\n<answer>{json.dumps(answer, ensure_ascii=False)}</answer>"
    conversations = [
        [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.resize((resize_size, resize_size), Image.BILINEAR)},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": assistant_content},
        ]
    ]
    texts = [processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in conversations]
    image_inputs, video_inputs = process_vision_info(conversations)
    return processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def save_visualization(image: Image.Image, pred_mask: np.ndarray, gt_mask: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image, alpha=0.6)
    overlay = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    overlay[pred_mask > 0] = [255, 0, 0]
    plt.imshow(overlay, alpha=0.35)
    plt.title("Pred Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image, alpha=0.6)
    gt_overlay = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    gt_overlay[gt_mask > 0] = [0, 255, 0]
    plt.imshow(gt_overlay, alpha=0.35)
    plt.title("GT Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


from pathlib import Path
from typing import Dict

class ImagePathResolver:
    def __init__(self, image_root: str):
        self.image_root = Path(image_root)
        self._stem_to_path: Dict[str, str] = {}
        self._name_to_path: Dict[str, str] = {}

        for path in self.image_root.rglob("*.jpg"):  # 只索引 JPG
            if path.is_file():
                self._stem_to_path.setdefault(path.stem, str(path))
                self._name_to_path.setdefault(path.name, str(path))

    def resolve(self, image_id: str) -> str:
        # 绝对路径且存在，直接返回
        image_id_path = Path(image_id)
        if image_id_path.is_absolute() and image_id_path.exists() and image_id_path.suffix.lower() == ".jpg":
            return str(image_id_path)

        # 相对路径拼接
        direct_path = self.image_root / image_id
        if direct_path.exists() and direct_path.suffix.lower() == ".jpg":
            return str(direct_path)

        # name/stem 查找
        if image_id in self._name_to_path:
            return self._name_to_path[image_id]
        stem = Path(image_id).stem
        if stem in self._stem_to_path:
            return self._stem_to_path[stem]

        raise FileNotFoundError(f"JPG file for image_id={image_id!r} not found under {self.image_root}")

@dataclass
class SampleRecord:
    image_path: str
    image_id: str
    question: str
    model_think: str
    answer_json: str
    input_box: List[int]
    input_points: List[List[int]]
    gt_mask: np.ndarray
    gt_box: List[int]
    input_iou: float
    img_height: int
    img_width: int
    sam_input_mask: Optional[np.ndarray]
    raw_item: Dict[str, Any]


@dataclass
class ReasonSegSampleRecord:
    image: Image.Image
    image_id: str
    ann_id: str
    question: str
    gt_mask: np.ndarray
    img_height: int
    img_width: int
    raw_item: Dict[str, Any]


class SegZeroOfflineSegDataset(Dataset):
    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def __init__(
        self,
        train_json_path: str,
        image_root: str,
        answer_resize: int = 840,
        sam_image_size: int = 1024,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.train_json_path = train_json_path
        self.image_root = image_root
        self.answer_resize = answer_resize
        self.sam_image_size = sam_image_size
        self.system_prompt = system_prompt
        self.path_resolver = ImagePathResolver(image_root)
        with open(train_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError("Expected the training JSON to be a list of records.")
        self.items = payload

    def __len__(self) -> int:
        return len(self.items)

    def _get_input_box_and_points(self, item: Dict[str, Any]) -> Tuple[List[int], List[List[int]]]:
        if item.get("pred_bbox") and item.get("pred_points"):
            return item["pred_bbox"], item["pred_points"]

        model_answer = item["model_parsed_answer"]
        width = item["img_width"]
        height = item["img_height"]
        x_factor = width / float(self.answer_resize)
        y_factor = height / float(self.answer_resize)
        bbox = [
            int(round(model_answer["bbox"][0] * x_factor)),
            int(round(model_answer["bbox"][1] * y_factor)),
            int(round(model_answer["bbox"][2] * x_factor)),
            int(round(model_answer["bbox"][3] * y_factor)),
        ]
        points = [
            [
                int(round(model_answer["points_1"][0] * x_factor)),
                int(round(model_answer["points_1"][1] * y_factor)),
            ],
            [
                int(round(model_answer["points_2"][0] * x_factor)),
                int(round(model_answer["points_2"][1] * y_factor)),
            ],
        ]
        return bbox, points

    def __getitem__(self, idx: int) -> SampleRecord:
        item = self.items[idx]
        image_path = self.path_resolver.resolve(item["image_id"])
        gt_mask = decode_rle_mask(item["gt_mask_rle"])
        gt_box = mask_to_box(gt_mask)
        input_box, input_points = self._get_input_box_and_points(item)
        input_iou = compute_box_iou(input_box, gt_box)
        sam_input_mask = None
        if item.get("sam_mask_rle") is not None:
            sam_input_mask = decode_rle_mask(item["sam_mask_rle"])
        return SampleRecord(
            image_path=image_path,
            image_id=item["image_id"],
            question=item["question"],
            model_think=item["model_think"],
            answer_json=build_answer_json(item["model_parsed_answer"]),
            input_box=input_box,
            input_points=input_points,
            gt_mask=gt_mask,
            gt_box=gt_box,
            input_iou=input_iou,
            img_height=item["img_height"],
            img_width=item["img_width"],
            sam_input_mask=sam_input_mask,
            raw_item=item,
        )

    def collate_fn(
        self,
        batch: List[SampleRecord],
        processor: AutoProcessor,
        max_pixels: int,
        min_pixels: int,
    ) -> Dict[str, Any]:
        conversations = []
        sam_images = []
        gt_masks = []
        gt_boxes = []
        input_boxes = []
        input_points = []
        input_ious = []
        image_paths = []
        answer_jsons = []

        processor.image_processor.max_pixels = max_pixels
        processor.image_processor.min_pixels = min_pixels

        for sample in batch:
            image = Image.open(sample.image_path).convert("RGB")
            sam_images.append(self._build_sam_image(image))
            gt_masks.append(torch.from_numpy(sample.gt_mask.astype(np.float32)))
            gt_boxes.append(sample.gt_box)
            input_boxes.append(sample.input_box)
            input_points.append(sample.input_points)
            input_ious.append(sample.input_iou)
            image_paths.append(sample.image_path)
            answer_jsons.append(sample.answer_json)

            assistant_content = f"<think>{sample.model_think}</think>\n<answer>{sample.answer_json}</answer>"
            conversations.append(
                [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": resize_longest_side(image, longest_side=840)},
                            {"type": "text", "text": sample.question},
                        ],
                    },
                    {"role": "assistant", "content": assistant_content},
                ]
            )

        texts = [processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in conversations]
        image_inputs, video_inputs = process_vision_info(conversations)
        model_inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {
            "model_inputs": model_inputs,
            "sam_images": torch.stack(sam_images),
            "gt_masks": gt_masks,
            "gt_boxes": gt_boxes,
            "input_boxes": input_boxes,
            "input_points": input_points,
            "input_ious": torch.tensor(input_ious, dtype=torch.float32),
            "image_paths": image_paths,
            "answer_jsons": answer_jsons,
        }

    def _build_sam_image(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize((self.sam_image_size, self.sam_image_size), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - self.pixel_mean) / self.pixel_std


class ReasonSegEvalDataset(Dataset):
    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def __init__(self, dataset_path: str, split: str = "test", sam_image_size: int = 1024):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "datasets is required to load ReasonSeg parquet datasets. Install the `datasets` package first."
            ) from exc
        self.dataset_path = dataset_path
        self.split = split
        self.sam_image_size = sam_image_size
        self.dataset = load_dataset(dataset_path, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> ReasonSegSampleRecord:
        item = self.dataset[idx]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        gt_mask = np.asarray(item["mask"], dtype=np.uint8)
        img_height = int(item["img_height"])
        img_width = int(item["img_width"])
        if gt_mask.shape != (img_height, img_width):
            gt_mask = gt_mask.reshape((img_height, img_width))
        return ReasonSegSampleRecord(
            image=image,
            image_id=str(item["image_id"]),
            ann_id=str(item["ann_id"]),
            question=str(item["text"]),
            gt_mask=gt_mask,
            img_height=img_height,
            img_width=img_width,
            raw_item=item,
        )

    def _build_sam_image(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize((self.sam_image_size, self.sam_image_size), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - self.pixel_mean) / self.pixel_std


def load_eval_dataset(
    eval_mode: str,
    eval_json_path: str,
    eval_dataset_path: str,
    image_root: str,
    answer_resize: int,
    sam_image_size: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dataset:
    if eval_mode == "reasonseg_dataset":
        if not eval_dataset_path:
            raise ValueError("eval_dataset_path is required when eval_mode=reasonseg_dataset")
        return ReasonSegEvalDataset(
            dataset_path=eval_dataset_path,
            split="test",
            sam_image_size=sam_image_size,
        )
    if not eval_json_path:
        raise ValueError("eval_json_path is required when eval_mode=offline_json")
    return SegZeroOfflineSegDataset(
        train_json_path=eval_json_path,
        image_root=image_root,
        answer_resize=answer_resize,
        sam_image_size=sam_image_size,
        system_prompt=system_prompt,
    )


def export_query_sam_components(
    model: "SegZeroQuerySAMModel",
    export_dir: Path,
    resolved_config: Dict[str, Any],
    sam_checkpoint_path: str,
    copy_sam_checkpoint: bool = False,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    mllm_dir = export_dir / "mllm"
    trainable_dir = export_dir / "trainable"
    sam2_dir = export_dir / "sam2"
    mllm_dir.mkdir(parents=True, exist_ok=True)
    trainable_dir.mkdir(parents=True, exist_ok=True)
    sam2_dir.mkdir(parents=True, exist_ok=True)

    model.qwen.save_pretrained(mllm_dir, safe_serialization=False)
    model.processor.save_pretrained(mllm_dir)
    torch.save(model.trainable_state_dict(), trainable_dir / "trainable_state_dict.pt")
    torch.save({name: param.detach().cpu() for name, param in model.sam.state_dict().items()}, sam2_dir / "sam2_state_dict.pt")

    export_metadata = {
        "num_query": int(model.num_query),
        "use_qwen_connector": bool(model.use_qwen_connector),
        "sam_model_cfg": str(model.sam_model_cfg),
        "sam_checkpoint_path": str(sam_checkpoint_path),
        "base_model_path": str(resolved_config.get("base_model_path", "")),
        "query_prompt_norm_type": str(resolved_config.get("query_prompt_norm_type", "layernorm")),
        "query_prompt_use_tanh": bool(resolved_config.get("query_prompt_use_tanh", True)),
        "query_prompt_init_scale": float(resolved_config.get("query_prompt_init_scale", 0.05)),
        "query_prompt_use_scale_gate": bool(resolved_config.get("query_prompt_use_scale_gate", True)),
        "query_prompt_log_stats": bool(resolved_config.get("query_prompt_log_stats", True)),
    }
    with open(export_dir / "export_metadata.json", "w", encoding="utf-8") as f:
        json.dump(export_metadata, f, ensure_ascii=False, indent=2)
    with open(export_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved_config, f, ensure_ascii=False, indent=2)

    sam_metadata = {
        "sam_checkpoint_path": str(sam_checkpoint_path),
        "sam_model_cfg": str(model.sam_model_cfg),
        "sam_state_dict_path": str(sam2_dir / "sam2_state_dict.pt"),
        "copied_checkpoint": None,
    }
    if copy_sam_checkpoint:
        source_path = Path(sam_checkpoint_path)
        if source_path.is_file():
            copied_path = sam2_dir / source_path.name
            shutil.copy2(source_path, copied_path)
            sam_metadata["copied_checkpoint"] = str(copied_path)
    with open(sam2_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(sam_metadata, f, ensure_ascii=False, indent=2)


def resolve_export_sam_checkpoint(export_dir: Path) -> Tuple[str, str]:
    metadata_path = export_dir / "sam2" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"SAM2 metadata not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    sam_model_cfg = str(metadata["sam_model_cfg"])
    copied_checkpoint = metadata.get("copied_checkpoint")
    sam_checkpoint_path = str(copied_checkpoint or metadata["sam_checkpoint_path"])
    return sam_checkpoint_path, sam_model_cfg


class SimpleQueryConnector(nn.Module):
    def __init__(self, hidden_size: int, num_query: int, depth: int = 2, num_heads: int = 8):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_query, hidden_size))
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.0,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.pos_embed[:, : hidden_states.size(1)]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rms = hidden_states.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return hidden_states * rms * self.weight


class QueryPromptStabilizer(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_type: str = "layernorm",
        use_tanh: bool = True,
        init_scale: float = 0.05,
        use_scale_gate: bool = True,
    ):
        super().__init__()
        norm_type = norm_type.lower()
        if norm_type == "rmsnorm":
            self.norm = RMSNorm(dim)
        else:
            self.norm = nn.LayerNorm(dim)
        self.use_tanh = use_tanh
        self.use_scale_gate = use_scale_gate
        safe_scale = max(float(init_scale), 1e-6)
        gate_init = torch.log(torch.expm1(torch.tensor(safe_scale)))
        self.scale_gate = nn.Parameter(gate_init.clone()) if use_scale_gate else None

    def current_scale(self) -> torch.Tensor:
        if self.scale_gate is None:
            return torch.tensor(1.0)
        return F.softplus(self.scale_gate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        if self.use_tanh:
            hidden_states = torch.tanh(hidden_states)
        if self.scale_gate is not None:
            hidden_states = hidden_states * F.softplus(self.scale_gate).to(hidden_states.dtype)
        return hidden_states


def dice_loss(pred_masks: torch.Tensor, gt_masks: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    pred_probs = torch.sigmoid(pred_masks)
    numerator = 2 * (pred_probs * gt_masks).sum(dim=(-1, -2))
    denominator = pred_probs.sum(dim=(-1, -2)) + gt_masks.sum(dim=(-1, -2))
    return 1 - (numerator + eps) / (denominator + eps)


class SegZeroQuerySAMModel(nn.Module):
    def __init__(
        self,
        base_model_path: str,
        sam_model_path: str,
        processor_path: Optional[str] = None,
        num_query: int = 64,
        use_qwen_connector: bool = True,
        freeze_mllm: bool = True,
        freeze_sam: bool = True,
        sam_model_cfg: str = DEFAULT_SAM2_CONFIG,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: torch.dtype = torch.bfloat16,
        query_prompt_norm_type: str = "layernorm",
        query_prompt_use_tanh: bool = True,
        query_prompt_init_scale: float = 0.05,
        query_prompt_use_scale_gate: bool = True,
        query_prompt_log_stats: bool = True,
    ):
        super().__init__()
        self.num_query = num_query
        self.use_qwen_connector = use_qwen_connector
        self.sam_model_cfg = sam_model_cfg
        self.query_prompt_log_stats = query_prompt_log_stats
        self._last_prompt_stats: Dict[str, float] = {}
        model_path = processor_path or base_model_path
        self.processor = AutoProcessor.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        hidden_size = self.qwen.config.hidden_size
        self.learnable_query = nn.Parameter(torch.randn(1, num_query, hidden_size) * 0.02)
        self.connector = SimpleQueryConnector(hidden_size=hidden_size, num_query=num_query) if use_qwen_connector else nn.Identity()
        self.query_agg = nn.Conv1d(hidden_size, hidden_size, kernel_size=num_query//4,stride = num_query//4)
        self.proj_to_sam = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 256),
        )
        self.query_prompt_stabilizer = QueryPromptStabilizer(
            dim=256,
            norm_type=query_prompt_norm_type,
            use_tanh=query_prompt_use_tanh,
            init_scale=query_prompt_init_scale,
            use_scale_gate=query_prompt_use_scale_gate,
        )
        self.sam = self._build_sam_model(sam_model_path, sam_model_cfg)
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]

        if hasattr(self.sam, "maskmem_tpos_enc"):
            del self.sam.maskmem_tpos_enc
        if hasattr(self.sam, "memory_attention"):
            del self.sam.memory_attention
        if hasattr(self.sam, "memory_encoder"):
            del self.sam.memory_encoder

        if freeze_mllm:
            self.qwen.requires_grad_(False)
            self.qwen.eval()
        if freeze_sam:
            self.sam.requires_grad_(False)
            self.sam.eval()

    def _build_sam_model(self, sam_model_path: str, sam_model_cfg: str):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "sam2 is required for query-SAM offline training. Install the `sam2` package first."
            ) from exc
        sam2_model = build_sam2(sam_model_cfg, sam_model_path, device="cuda", dtype=torch.bfloat16)
        predictor = SAM2ImagePredictor(sam2_model)
        if not hasattr(predictor, "model"):
            raise AttributeError("Expected SAM2ImagePredictor.from_pretrained(...) to expose a `.model` attribute.")
        self._sam_predictor = predictor
        self._sam_transforms = predictor._transforms
        self._sam_mask_threshold = predictor.mask_threshold
        return predictor.model

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: param.detach().cpu() for name, param in self.named_parameters() if param.requires_grad}

    def load_trainable_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        return self.load_state_dict(state_dict, strict=strict)

    def build_optimizer(self, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
        params = [param for param in self.parameters() if param.requires_grad]
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)

    def get_sam_text_embeds(
        self,
        model_inputs: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = self.learnable_query.device
        attention_mask, inputs_embeds, position_ids = self._prepare_llm_inputs(
            input_ids=model_inputs["input_ids"].to(device),
            pixel_values=model_inputs["pixel_values"].to(device),
            image_grid_thw=model_inputs["image_grid_thw"].to(device),
            attention_mask=model_inputs["attention_mask"].to(device),
        )
        outputs = self.qwen.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        query_hidden_states = outputs.hidden_states[-1][:, -self.num_query :]
        query_hidden_states = self.connector(query_hidden_states)
        query_hidden_states = self.query_agg(query_hidden_states.transpose(1, 2)).transpose(1, 2).contiguous()
        return self.query_prompt_stabilizer(self.proj_to_sam(query_hidden_states))

    def _prepare_llm_inputs(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        inputs_embeds = self.qwen.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.qwen.visual.dtype)
            image_embeds = self.qwen.visual(pixel_values, grid_thw=image_grid_thw)
            image_mask = (
                (input_ids == self.qwen.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype))

        batch_size = inputs_embeds.size(0)
        query_embeds = self.learnable_query.expand(batch_size, -1, -1).to(inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)
        query_attention = torch.ones(batch_size, self.num_query, device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, query_attention], dim=1)

        seq_len = attention_mask.size(1)
        position_ids = torch.zeros((3, batch_size, seq_len), device=input_ids.device, dtype=torch.long)
        for batch_idx in range(batch_size):
            base_attention = attention_mask[batch_idx, :-self.num_query]
            base_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[batch_idx],
                image_grid_thw=image_grid_thw[batch_idx : batch_idx + 1],
                attention_mask=base_attention,
            )
            valid_len = int(base_attention.sum().item())
            position_ids[:, batch_idx, :-self.num_query] = 1
            position_ids[:, batch_idx, : base_position_ids.size(1)] = base_position_ids
            if valid_len > 0:
                last_position = base_position_ids[:, valid_len - 1 : valid_len]
            else:
                last_position = torch.zeros((3, 1), device=input_ids.device, dtype=torch.long)
            query_offsets = torch.arange(1, self.num_query + 1, device=input_ids.device, dtype=torch.long).view(1, -1)
            position_ids[:, batch_idx, -self.num_query :] = last_position + query_offsets
        return attention_mask, inputs_embeds, position_ids

    def _predict_masks(
        self,
        sam_images: torch.Tensor,
        sam_text_embeds: torch.Tensor,
        gt_masks: List[torch.Tensor],
        input_boxes: Optional[List[List[int]]] = None,
        input_points: Optional[List[List[List[int]]]] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[torch.Tensor]:
        with torch.no_grad():
            backbone_out = self.sam.forward_image(sam_images)
            _, image_embeddings, _, _ = self.sam._prepare_backbone_features(backbone_out)
            image_embeddings = [feature.to(sam_images.dtype) for feature in image_embeddings]
            if self.sam.directly_add_no_mem_embed:
                image_embeddings[-1] = image_embeddings[-1] + self.sam.no_mem_embed

        batch_size = sam_images.shape[0]
        feats = [
            feature.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feature, feat_size in zip(image_embeddings[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        pred_masks: List[torch.Tensor] = []
        query_prompt_mean_abs_values: List[float] = []
        query_prompt_max_abs_values: List[float] = []
        query_prompt_l2_values: List[float] = []
        query_prompt_isfinite_values: List[float] = []
        sam_sparse_mean_abs_values: List[float] = []
        sam_sparse_max_abs_values: List[float] = []
        sam_sparse_l2_values: List[float] = []
        sam_sparse_isfinite_values: List[float] = []
        decoder_mask_isfinite_values: List[float] = []
        for idx in range(batch_size):
            concat_points = None
            if input_points is not None and input_points[idx]:
                if image_sizes is None:
                    raise ValueError("image_sizes is required when input_points are provided.")
                point_tensor = torch.tensor(input_points[idx], device=sam_images.device, dtype=torch.float32)
                point_tensor = self._sam_transforms.transform_coords(
                    point_tensor,
                    normalize=True,
                    orig_hw=image_sizes[idx],
                )
                point_labels = torch.ones((len(input_points[idx]),), device=sam_images.device, dtype=torch.int32)
                concat_points = (point_tensor.unsqueeze(0), point_labels.unsqueeze(0))
            if input_boxes is not None and input_boxes[idx]:
                if image_sizes is None:
                    raise ValueError("image_sizes is required when input_boxes are provided.")
                prompt_boxes = torch.tensor(input_boxes[idx], device=sam_images.device, dtype=torch.float32).unsqueeze(0)
                prompt_boxes = self._sam_transforms.transform_boxes(
                    prompt_boxes,
                    normalize=True,
                    orig_hw=image_sizes[idx],
                )
                box_coords = prompt_boxes.reshape(-1, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=sam_images.device)
                if concat_points is not None:
                    concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                    concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                    concat_points = (concat_coords, concat_labels)
                else:
                    concat_points = (box_coords, box_labels)
            sparse_embeddings, dense_embeddings = self.sam.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
            if self.query_prompt_log_stats:
                sam_sparse_mean_abs_values.append(float(sparse_embeddings.detach().abs().mean().float().item()))
                sam_sparse_max_abs_values.append(float(sparse_embeddings.detach().abs().max().float().item()))
                sam_sparse_l2_values.append(float(sparse_embeddings.detach().float().pow(2).mean().sqrt().item()))
                sam_sparse_isfinite_values.append(float(torch.isfinite(sparse_embeddings).all().item()))
                query_prompt = sam_text_embeds[idx].unsqueeze(0)
                query_prompt_mean_abs_values.append(float(query_prompt.detach().abs().mean().float().item()))
                query_prompt_max_abs_values.append(float(query_prompt.detach().abs().max().float().item()))
                query_prompt_l2_values.append(float(query_prompt.detach().float().pow(2).mean().sqrt().item()))
                query_prompt_isfinite_values.append(float(torch.isfinite(query_prompt).all().item()))
            sparse_embeddings = torch.cat(
                [sparse_embeddings, sam_text_embeds[idx].unsqueeze(0)],
                dim=1
            )
            high_res_features = [feat_level[idx].unsqueeze(0) for feat_level in features["high_res_feats"]]
            batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
            low_res_masks, _, _, _ = self.sam.sam_mask_decoder(
                image_embeddings=features["image_embed"][idx].unsqueeze(0),
                image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            if image_sizes is not None:
                pred_mask = self._sam_transforms.postprocess_masks(low_res_masks.float(), tuple(image_sizes[idx]))
            else:
                target_hw = tuple(gt_masks[idx].shape[-2:])
                pred_mask = F.interpolate(low_res_masks.float(), size=target_hw, mode="bilinear", align_corners=False)
            if self.query_prompt_log_stats:
                decoder_mask_isfinite_values.append(float(torch.isfinite(pred_mask).all().item()))
            pred_masks.append(pred_mask)
        if self.query_prompt_log_stats:
            self._last_prompt_stats = {
                "query_prompt_mean_abs": float(np.mean(query_prompt_mean_abs_values)) if query_prompt_mean_abs_values else 0.0,
                "query_prompt_max_abs": float(np.mean(query_prompt_max_abs_values)) if query_prompt_max_abs_values else 0.0,
                "query_prompt_l2": float(np.mean(query_prompt_l2_values)) if query_prompt_l2_values else 0.0,
                "query_prompt_isfinite": float(np.mean(query_prompt_isfinite_values)) if query_prompt_isfinite_values else 1.0,
                "sam_sparse_mean_abs": float(np.mean(sam_sparse_mean_abs_values)) if sam_sparse_mean_abs_values else 0.0,
                "sam_sparse_max_abs": float(np.mean(sam_sparse_max_abs_values)) if sam_sparse_max_abs_values else 0.0,
                "sam_sparse_l2": float(np.mean(sam_sparse_l2_values)) if sam_sparse_l2_values else 0.0,
                "sam_sparse_isfinite": float(np.mean(sam_sparse_isfinite_values)) if sam_sparse_isfinite_values else 1.0,
                "decoder_mask_isfinite": float(np.mean(decoder_mask_isfinite_values)) if decoder_mask_isfinite_values else 1.0,
                "query_prompt_scale": float(self.query_prompt_stabilizer.current_scale().detach().float().item()),
            }
        return pred_masks

    def forward(
        self,
        batch: Dict[str, Any],
        match_iou_threshold: float,
        bce_weight: float,
        dice_weight: float,
        use_iou_filter_for_loss: bool = False,
        train_use_prompt_boxes: bool = True,
        train_use_prompt_points: bool = True,
    ) -> Dict[str, Any]:
        self.qwen.eval()
        self.sam.eval()
        model_inputs = batch["model_inputs"]
        sam_images = batch["sam_images"]
        gt_masks = batch["gt_masks"]
        input_ious = batch["input_ious"].to(sam_images.device)
        input_boxes = batch["input_boxes"] if train_use_prompt_boxes else None
        input_points = batch["input_points"] if train_use_prompt_points else None
        sam_text_embeds = self.get_sam_text_embeds(model_inputs, device=sam_images.device)
        pred_masks = self._predict_masks(
            sam_images=sam_images,
            sam_text_embeds=sam_text_embeds,
            gt_masks=gt_masks,
            input_boxes=input_boxes,
            input_points=input_points,
            image_sizes=[tuple(mask.shape[-2:]) for mask in gt_masks],
        )

        dtype = pred_masks[0].dtype if pred_masks else torch.float32
        prompt_stats = dict(self._last_prompt_stats)
        total_bce = torch.zeros((), device=sam_images.device, dtype=dtype)
        total_dice = torch.zeros_like(total_bce)
        nan_count = 0
        nonfinite_mask_count = 0
        valid_count = 0
        for idx, pred_mask in enumerate(pred_masks):
            if use_iou_filter_for_loss and not bool((input_ious[idx] < match_iou_threshold).item()):
                continue
            if not torch.isfinite(pred_mask).all():
                nonfinite_mask_count += 1
                continue
            gt_mask = gt_masks[idx].to(pred_mask.device, pred_mask.dtype).unsqueeze(0).unsqueeze(0)
            sample_bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask)
            sample_dice = dice_loss(pred_mask, gt_mask).mean()
            if not torch.isfinite(sample_bce) or not torch.isfinite(sample_dice):
                nan_count += 1
                continue
            total_bce = total_bce + sample_bce
            total_dice = total_dice + sample_dice
            valid_count += 1

        if valid_count == 0:
            zero_loss = torch.zeros((), device=sam_images.device, dtype=sam_text_embeds.dtype)
            mean_bce = zero_loss
            mean_dice = zero_loss
            loss = zero_loss
        else:
            mean_bce = total_bce / valid_count
            mean_dice = total_dice / valid_count
            loss = mean_bce * bce_weight + mean_dice * dice_weight
            if not torch.isfinite(loss):
                nan_count += 1
                zero_loss = torch.zeros((), device=sam_images.device, dtype=sam_text_embeds.dtype)
                mean_bce = zero_loss
                mean_dice = zero_loss
                loss = zero_loss

        return {
            "loss": loss,
            "bce_loss": mean_bce.detach(),
            "dice_loss": mean_dice.detach(),
            "valid_count": valid_count,
            "num_supervised": valid_count,
            "nan_count": nan_count,
            "nonfinite_mask_count": nonfinite_mask_count,
            "input_ious": input_ious.detach(),
            "sam_text_embeds_isfinite": float(torch.isfinite(sam_text_embeds).all().item()),
            **prompt_stats,
        }
