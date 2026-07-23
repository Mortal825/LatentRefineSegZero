from pathlib import Path
import json
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import ConcatDataset, Dataset, Subset

from training_scripts.refine_segzero.common import (
    ImagePathResolver,
    RefineSample,
    build_sam_image_tensor,
    decode_rle_mask,
    extract_refexp,
    load_json_records,
    mask_to_box,
    resize_longest_side,
)
from training_scripts.refine_segzero.prompts import DEFAULT_SYSTEM_PROMPT, GEOMETRIC_QUERY_TEMPLATE


def _strip_reflect_output_to_reason_only(text: str) -> str:
    text = str(text or "").strip()
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if not think_match:
        return re.sub(r"<answer>.*?</answer>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    think = re.sub(r"\s*4\.\s*.*$", "", think_match.group(1).strip(), flags=re.DOTALL).strip()
    return f"<think>{think}</think>" if think else ""


class RefExpJsonDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_root: str,
        max_sample_ratio: Optional[float] = None,
        sample_seed: int = 42,
    ):
        self.items = load_json_records(json_path)
        target_count = len(self.items)
        if max_sample_ratio is not None and float(max_sample_ratio) > 0:
            ratio = min(max(float(max_sample_ratio), 0.0), 1.0)
            target_count = min(target_count, max(1, int(round(len(self.items) * ratio))))
        if target_count < len(self.items):
            rng = random.Random(int(sample_seed))
            sampled_indices = sorted(rng.sample(range(len(self.items)), target_count))
            self.items = [self.items[idx] for idx in sampled_indices]
        self.path_resolver = ImagePathResolver(image_root)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> RefineSample:
        item = self.items[index]
        image_path = self.path_resolver.resolve(item["image"])
        image = Image.open(image_path).convert("RGB")
        gt_mask = decode_rle_mask(item["objects"][0]["rle"])
        return RefineSample(
            image_path=image_path,
            image_id=str(item["image"]),
            sample_id=str(item["id"]),
            question_text=str(item["conversations"][0]["value"]),
            refexp=extract_refexp(str(item["conversations"][0]["value"])),
            gt_mask=gt_mask,
            gt_box_xyxy=mask_to_box(gt_mask),
            width=image.width,
            height=image.height,
            raw_item=item,
        )

class Stage1CacheDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        max_sample_ratio: Optional[float] = None,
        sample_seed: int = 42,
    ):
        records = load_json_records(json_path)
        target_count = len(records)
        if max_sample_ratio is not None and float(max_sample_ratio) > 0:
            ratio = min(max(float(max_sample_ratio), 0.0), 1.0)
            target_count = min(target_count, max(1, int(round(len(records) * ratio))))
        if target_count < len(records):
            rng = random.Random(int(sample_seed))
            records = [records[idx] for idx in sorted(rng.sample(range(len(records)), target_count))]

        self.items: List[Dict[str, Any]] = []
        missing = 0
        for record in records:
            if not record.get("image_path") or not record.get("gt_mask_rle"):
                missing += 1
                continue
            self.items.append(record)

        if not self.items:
            raise ValueError(f"No usable stage1 cache samples were built from {json_path}.")
        if missing:
            print(f"Stage1CacheDataset skipped {missing} records without image_path or gt_mask_rle.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> RefineSample:
        record = self.items[index]
        gt_mask = decode_rle_mask(record["gt_mask_rle"])
        image_path = str(record["image_path"])
        image = Image.open(image_path).convert("RGB")
        try:
            width, height = image.width, image.height
        finally:
            image.close()

        image_size = record.get("image_size")
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            width, height = int(image_size[0]), int(image_size[1])

        question = str(record.get("question", record.get("refexp", "")))
        question_text = str(record.get("question_text", question))
        raw_item = dict(record)
        raw_item["stage1_box"] = record.get("stage1_box", [0, 0, 0, 0])
        raw_item["stage1_points"] = record.get("stage1_points", [[0, 0], [0, 0]])

        return RefineSample(
            image_path=image_path,
            image_id=str(record.get("image_id", record.get("meta_image_id", ""))),
            sample_id=str(record.get("sample_id", record.get("ann_id", record.get("meta_sample_id", "")))),
            question_text=question_text,
            refexp=question,
            gt_mask=gt_mask,
            gt_box_xyxy=[int(v) for v in record.get("gt_box", mask_to_box(gt_mask))],
            width=width,
            height=height,
            raw_item=raw_item,
        )
def _direct_sft_key(image_id: Any, sample_id: Any) -> Tuple[str, str]:
    return str(image_id), str(sample_id)


def _load_source_sample_index(
    json_paths: Sequence[str],
    image_root: str,
    max_sample_ratio_per_file: Optional[float],
    sample_seed: int,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    resolver = ImagePathResolver(image_root)

    for dataset_idx, json_path in enumerate(path for path in json_paths if str(path).strip()):
        items = load_json_records(str(json_path))
        target_count = len(items)

        if max_sample_ratio_per_file is not None and float(max_sample_ratio_per_file) > 0:
            ratio = min(max(float(max_sample_ratio_per_file), 0.0), 1.0)
            target_count = min(target_count, max(1, int(round(len(items) * ratio))))

        if target_count < len(items):
            rng = random.Random(int(sample_seed) + dataset_idx)
            items = [items[idx] for idx in sorted(rng.sample(range(len(items)), target_count))]

        for item in items:
            image_id = str(item["image"])
            sample_id = str(item["id"])
            question_text = str(item["conversations"][0]["value"])

            index[_direct_sft_key(image_id, sample_id)] = {
                "image_path": resolver.resolve(image_id),
                "image_id": image_id,
                "sample_id": sample_id,
                "question_text": question_text,
                "refexp": extract_refexp(question_text),
                "raw_item": item,
            }

    return index


def _load_stage1_cache_index(stage1_cache_json_path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not str(stage1_cache_json_path).strip():
        return {}

    index: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for record in load_json_records(str(stage1_cache_json_path)):
        image_id = record.get("image_id", record.get("meta_image_id", ""))
        sample_id = record.get("sample_id", record.get("ann_id", record.get("meta_sample_id", "")))
        index[_direct_sft_key(image_id, sample_id)] = record

    return index


def _reflect_pred_decision(record: Dict[str, Any]) -> str:
    for key in ("pred_decision", "reflect_decision", "decision", "parsed_decision"):
        value = str(record.get(key, "")).strip().lower()
        if value in {"accept", "reject"}:
            return value

    try:
        solution = json.loads(str(record.get("solution", "{}")))
    except Exception:
        solution = {}

    value = str(solution.get("target_decision", "")).strip().lower()
    return value if value in {"accept", "reject"} else ""


class DirectRejectSftDataset(Dataset):
    def __init__(
        self,
        reflect_json_path: str,
        max_sample_ratio: Optional[float] = None,
        sample_seed: int = 42,
    ):
        if not str(reflect_json_path).strip():
            raise ValueError("direct SFT requires reflect_json_path.")

        records = [
            record
            for record in load_json_records(str(reflect_json_path))
            if _reflect_pred_decision(record) in {"accept", "reject"}
            # if _reflect_pred_decision(record) == "reject"
        ]

        if max_sample_ratio is not None and float(max_sample_ratio) > 0:
            ratio = min(max(float(max_sample_ratio), 0.0), 1.0)
            target_count = min(len(records), max(1, int(round(len(records) * ratio))))
            if target_count < len(records):
                rng = random.Random(int(sample_seed))
                records = [records[idx] for idx in sorted(rng.sample(range(len(records)), target_count))]

        self.items: List[Dict[str, Any]] = []
        missing = 0

        for record in records:
            if not record.get("image_path") or not record.get("gt_mask_rle"):
                missing += 1
                continue
            self.items.append(record)

        if not self.items:
            raise ValueError(
                "No reject direct SFT samples were built. Check reflect_json_path decisions "
                "and make sure records contain image_path and gt_mask_rle."
            )

        if missing:
            print(f"DirectRejectSftDataset skipped {missing} reject records without image_path or gt_mask_rle.")



    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> RefineSample:
        record = self.items[index]
        context = {
            "reflect_prompt": record.get("reflect_prompt", ""),
            "reflect_decision": _reflect_pred_decision(record),
            "reflect_output_text": record.get("reflect_output_text", record.get("output_text", "")),
            "stage1_output_text": record.get("stage1_output_text", ""),
            "stage1_think": record.get("stage1_think", ""),
            "stage1_box": record.get("stage1_box", [0, 0, 0, 0]),
            "stage1_points": record.get("stage1_points", [[0, 0], [0, 0]]),
            "reflect_record": record,
            "proposal_visualization_path": record.get("proposal_visualization_path", ""),
            "reflect_record": record,
        }

        raw_item = dict(record)
        raw_item["direct_sft_context"] = context

        gt_mask = decode_rle_mask(record["gt_mask_rle"])
        image_path = str(record["image_path"])

        image = Image.open(image_path).convert("RGB")
        try:
            width, height = image.width, image.height
        finally:
            image.close()

        question = str(record.get("question", record.get("refexp", "")))
        question_text = str(record.get("question_text", question))

        return RefineSample(
            image_path=image_path,
            image_id=str(record.get("image_id", record.get("meta_image_id", ""))),
            sample_id=str(record.get("sample_id", record.get("ann_id", record.get("meta_sample_id", "")))),
            question_text=question_text,
            refexp=question,
            gt_mask=gt_mask,
            gt_box_xyxy=[int(v) for v in record.get("gt_box", mask_to_box(gt_mask))],
            width=width,
            height=height,
            raw_item=raw_item,
        )



def build_direct_reject_sft_dataset(
    reflect_json_path: str,
    max_sample_ratio: Optional[float] = None,
    sample_seed: int = 42,
) -> DirectRejectSftDataset:
    return DirectRejectSftDataset(
        reflect_json_path=reflect_json_path,
        max_sample_ratio=max_sample_ratio,
        sample_seed=sample_seed,
    )


def build_stage1_cache_dataset(
    json_paths: Sequence[str],
    max_sample_ratio_per_file: Optional[float] = None,
    sample_seed: int = 42,
) -> Dataset:
    datasets = [
        Stage1CacheDataset(
            str(path),
            max_sample_ratio=max_sample_ratio_per_file,
            sample_seed=sample_seed + dataset_idx,
        )
        for dataset_idx, path in enumerate(json_paths)
        if str(path).strip()
    ]
    if not datasets:
        raise ValueError("At least one stage1 cache json path is required.")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

def build_train_dataset(
    train_json_paths: Sequence[str],
    image_root: str,
    max_sample_ratio_per_file: Optional[float] = None,
    sample_seed: int = 42,
) -> Dataset:
    datasets = [
        RefExpJsonDataset(
            path,
            image_root=image_root,
            max_sample_ratio=max_sample_ratio_per_file,
            sample_seed=sample_seed + dataset_idx,
        )
        for dataset_idx, path in enumerate(train_json_paths)
        if str(path).strip()
    ]
    if not datasets:
        raise ValueError("At least one train json path is required.")
    dataset: Dataset
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)
    return dataset


def build_eval_dataset(
    eval_json_path: str,
    image_root: str,
    max_sample_ratio: Optional[float] = None,
    sample_seed: int = 42,
) -> RefExpJsonDataset:
    return RefExpJsonDataset(
        eval_json_path,
        image_root=image_root,
        max_sample_ratio=max_sample_ratio,
        sample_seed=sample_seed,
    )


def geometric_collate_fn(
    batch: List[RefineSample],
    processor: Any,
    resize_size: int,
    sam_image_size: int,
    max_pixels: int,
    min_pixels: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dict[str, Any]:
    conversations = []
    sam_images = []
    gt_masks = []
    gt_boxes = []
    image_sizes = []
    meta = []

    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.min_pixels = min_pixels

    for sample in batch:
        cached_image = sample.raw_item.get("image_obj") if isinstance(sample.raw_item, dict) else None
        if isinstance(cached_image, Image.Image):
            image = cached_image.convert("RGB")
        else:
            image = Image.open(sample.image_path).convert("RGB")
        sam_images.append(build_sam_image_tensor(image, sam_image_size))
        gt_masks.append(torch.from_numpy(sample.gt_mask.astype(np.float32)))
        gt_boxes.append(sample.gt_box_xyxy)
        image_sizes.append((sample.height, sample.width))
        conversations.append(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resize_longest_side(image, longest_side=resize_size)},
                        {
                            "type": "text",
                            "text": GEOMETRIC_QUERY_TEMPLATE.format(Question=sample.refexp.lower().strip(".")),
                        },
                    ],
                },
            ]
        )
        meta.append(
            {
                "image_id": sample.image_id,
                "sample_id": sample.sample_id,
                "question": sample.refexp,
                "question_text": sample.question_text,
                "image_path": sample.image_path,
                "width": sample.width,
                "height": sample.height,
                "stage1_box": sample.raw_item.get("stage1_box", [0, 0, 0, 0]) if isinstance(sample.raw_item, dict) else [0, 0, 0, 0],
                "stage1_points": sample.raw_item.get("stage1_points", [[0, 0], [0, 0]]) if isinstance(sample.raw_item, dict) else [[0, 0], [0, 0]],
                "gt_point": sample.raw_item.get("gt_point") if isinstance(sample.raw_item, dict) else None,
            }
        )

    texts = [processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) for conv in conversations]
    image_inputs, video_inputs = process_vision_info(conversations)
    model_inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return {
        "model_inputs": model_inputs,
        "sam_images": torch.stack(sam_images),
        "gt_masks": gt_masks,
        "gt_boxes": gt_boxes,
        "image_sizes": image_sizes,
        "meta": meta,
    }

## 濡傛灉鍙槸鏀?direct 璁粌鏃?MLLM 杈撳叆 prompt 鐨勬嫾鎺ユ柟寮忥紝涓昏鍙敼 data.py 閲岀殑 direct_reject_sft_collate_fn 灏辫銆?
def direct_reject_sft_collate_fn(
    batch: List[RefineSample],
    processor: Any,
    resize_size: int,
    sam_image_size: int,
    max_pixels: int,
    min_pixels: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    direct_prompt_mode: str = "query_reflect",
) -> Dict[str, Any]:
    conversations = []
    sam_images = []
    gt_masks = []
    gt_boxes = []
    image_sizes = []
    meta = []

    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.min_pixels = min_pixels

    prompt_mode = str(direct_prompt_mode or "query_reflect").strip().lower()
    if prompt_mode == "reflect_context":
        prompt_mode = "query_reflect"
    if prompt_mode not in {"raw", "query_reflect", "query_reflect_reason_only"}:
        raise ValueError(f"Unsupported direct_prompt_mode: {direct_prompt_mode}")
    for sample in batch:
        cached_image = sample.raw_item.get("image_obj") if isinstance(sample.raw_item, dict) else None
        if isinstance(cached_image, Image.Image):
            image = cached_image.convert("RGB")
        else:
            image = Image.open(sample.image_path).convert("RGB")

        context = sample.raw_item.get("direct_sft_context", {}) if isinstance(sample.raw_item, dict) else {}
        reflect_prompt = str(context.get("reflect_prompt", "")).strip()
        stage1_output_text = str(context.get("stage1_output_text", "")).strip()
        reflect_output_text = str(context.get("reflect_output_text", "")).strip()
        reflect_decision = str(context.get("reflect_decision", "reject")).strip().lower() or "reject"
        proposal_visualization_path = str(context.get("proposal_visualization_path", "")).strip()
        if not reflect_output_text:
            reflect_output_text = f'<answer>{{"decision":"{reflect_decision}"}}</answer>'
        if prompt_mode == "query_reflect_reason_only":
            reflect_output_text = _strip_reflect_output_to_reason_only(reflect_output_text)

        sam_images.append(build_sam_image_tensor(image, sam_image_size))
        gt_masks.append(torch.from_numpy(sample.gt_mask.astype(np.float32)))
        gt_boxes.append(sample.gt_box_xyxy)
        image_sizes.append((sample.height, sample.width))

        if prompt_mode == "raw":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resize_longest_side(image, longest_side=resize_size)},
                        {
                            "type": "text",
                            "text": GEOMETRIC_QUERY_TEMPLATE.format(Question=sample.refexp.lower().strip(".")),
                        },
                    ],
                },
            ]
        else:
            proposal_image = image
            if proposal_visualization_path:
                proposal_image = Image.open(proposal_visualization_path).convert("RGB")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resize_longest_side(proposal_image, longest_side=resize_size)},
                        {"type": "text", "text": reflect_prompt},
                    ],
                },
            ]
            if reflect_output_text:
                conversation.append({"role": "assistant", "content": reflect_output_text})

        conversations.append(conversation)
        meta.append(
            {
                "image_id": sample.image_id,
                "sample_id": sample.sample_id,
                "question": sample.refexp,
                "question_text": sample.question_text,
                "image_path": sample.image_path,
                "width": sample.width,
                "height": sample.height,
                "reflect_decision": reflect_decision,
                "direct_prompt_mode": prompt_mode,
                "stage1_box": context.get("stage1_box", [0, 0, 0, 0]),
                "stage1_points": context.get("stage1_points", [[0, 0], [0, 0]]),
            }
        )
    texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]
    ### 鍙鍖?    # for idx, text in enumerate(texts):
    #     print("=" * 80)
    #     print("[DIRECT_SFT_DEBUG] prompt index:", idx)
    #     print(text)
    #     print("=" * 80)
    debug_prompt_dir = str(os.environ.get("SEGZERO_DIRECT_EVAL_PROMPT_DIR", "")).strip()
    if debug_prompt_dir:
        debug_dir = Path(debug_prompt_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        for idx, (text, sample) in enumerate(zip(texts, batch)):
            image_stem = Path(str(sample.image_id)).stem
            sample_label = re.sub(r"[^0-9a-zA-Z._-]+", "_", str(sample.sample_id)).strip("_") or str(idx)
            output_path = debug_dir / f"{image_stem}__{sample_label}.txt"
            output_path.write_text(text, encoding="utf-8")
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
        "image_sizes": image_sizes,
        "meta": meta,
    }
