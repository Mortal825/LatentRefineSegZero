import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from training_scripts.refine_segzero.common import (
    ImagePathResolver,
    decode_rle_mask,
    extract_refexp,
    load_json_records,
    mask_to_box,
)


def env_rank() -> Tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def sample_indices(total_len: int, idx: int, num_parts: int) -> range:
    start = (total_len * idx) // num_parts
    end = (total_len * (idx + 1)) // num_parts
    return range(start, end)


def center_point_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        box = mask_to_box(mask)
        return [int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)]
    return [int(round(float(xs.mean()))), int(round(float(ys.mean())))]


def points_from_mask(mask: np.ndarray) -> List[List[int]]:
    ys, xs = np.where(mask > 0)
    center = center_point_from_mask(mask)
    if len(xs) < 2 or len(ys) < 2:
        return [center, center]

    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center_arr = np.asarray(center, dtype=np.float32)
    first_idx = int(np.argmax(np.sum((coords - center_arr) ** 2, axis=1)))
    first = coords[first_idx]
    second_idx = int(np.argmax(np.sum((coords - first) ** 2, axis=1)))
    second = coords[second_idx]
    return [
        [int(round(float(first[0]))), int(round(float(first[1])))],
        [int(round(float(second[0]))), int(round(float(second[1])))],
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _load_json_with_retry(path: Path, retries: int = 10, sleep_s: float = 1.0) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            time.sleep(sleep_s)
    if last_error is not None:
        raise last_error


def merge_part_dir(parts_dir: Path, output_path: Path) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for path in sorted(parts_dir.glob("part_*.json")):
        merged.extend(_load_json_with_retry(path))
    write_json(output_path, merged)
    return merged


def wait_for_parts(paths: Iterable[Path], timeout_s: int = 600, stable_checks: int = 2) -> None:
    deadline = time.time() + timeout_s
    pending = {Path(path): 0 for path in paths}
    last_sizes: Dict[Path, int] = {}
    while pending and time.time() < deadline:
        next_pending: Dict[Path, int] = {}
        for path, stable_count in pending.items():
            if not path.exists():
                next_pending[path] = 0
                continue
            current_size = path.stat().st_size
            previous_size = last_sizes.get(path)
            if current_size > 0 and current_size == previous_size:
                stable_count += 1
            else:
                stable_count = 0
            last_sizes[path] = current_size
            if stable_count < stable_checks:
                next_pending[path] = stable_count
        pending = next_pending
        if pending:
            time.sleep(1.0)
    if pending:
        raise TimeoutError(f"Timed out waiting for part files: {list(pending)}")


def load_source_records(
    json_paths: Sequence[str],
    image_root: str,
    max_sample_ratio_per_file: float,
    sample_seed: int,
) -> List[Dict[str, Any]]:
    resolver = ImagePathResolver(image_root)
    merged: List[Dict[str, Any]] = []
    for file_idx, json_path in enumerate(path for path in json_paths if str(path).strip()):
        items = load_json_records(str(json_path))
        target_count = len(items)
        if max_sample_ratio_per_file is not None and float(max_sample_ratio_per_file) > 0:
            ratio = min(max(float(max_sample_ratio_per_file), 0.0), 1.0)
            target_count = min(target_count, max(1, int(round(len(items) * ratio))))
        if target_count < len(items):
            rng = random.Random(int(sample_seed) + file_idx)
            sampled_indices = sorted(rng.sample(range(len(items)), target_count))
            items = [items[item_idx] for item_idx in sampled_indices]
        for item in items:
            image_path = resolver.resolve(item["image"])
            question_text = str(item["conversations"][0]["value"])
            gt_mask = decode_rle_mask(item["objects"][0]["rle"])
            image = Image.open(image_path).convert("RGB")
            merged.append(
                {
                    "image_path": image_path,
                    "image_id": str(item["image"]),
                    "sample_id": str(item["id"]),
                    "question_text": question_text,
                    "refexp": extract_refexp(question_text),
                    "gt_mask": gt_mask,
                    "gt_box": mask_to_box(gt_mask),
                    "gt_point": center_point_from_mask(gt_mask),
                    "gt_points": points_from_mask(gt_mask),
                    "width": image.width,
                    "height": image.height,
                }
            )
            image.close()
    return merged


def load_source_records_shard(
    json_paths: Sequence[str],
    image_root: str,
    max_sample_ratio_per_file: float,
    sample_seed: int,
    idx: int,
    num_parts: int,
) -> List[Dict[str, Any]]:
    resolver = ImagePathResolver(image_root)
    file_items: List[Tuple[List[Dict[str, Any]], List[int]]] = []
    total_count = 0

    for file_idx, json_path in enumerate(path for path in json_paths if str(path).strip()):
        items = load_json_records(str(json_path))
        target_count = len(items)
        if max_sample_ratio_per_file is not None and float(max_sample_ratio_per_file) > 0:
            ratio = min(max(float(max_sample_ratio_per_file), 0.0), 1.0)
            target_count = min(target_count, max(1, int(round(len(items) * ratio))))

        if target_count < len(items):
            rng = random.Random(int(sample_seed) + file_idx)
            selected_indices = sorted(rng.sample(range(len(items)), target_count))
        else:
            selected_indices = list(range(len(items)))

        file_items.append((items, selected_indices))
        total_count += len(selected_indices)

    shard = sample_indices(total_count, idx, num_parts)
    merged: List[Dict[str, Any]] = []
    global_index = 0

    for items, selected_indices in file_items:
        for item_idx in selected_indices:
            if global_index >= shard.stop:
                break

            if global_index >= shard.start:
                item = items[item_idx]
                image_path = resolver.resolve(item["image"])
                question_text = str(item["conversations"][0]["value"])
                gt_mask = decode_rle_mask(item["objects"][0]["rle"])
                image = Image.open(image_path).convert("RGB")
                merged.append(
                    {
                        "image_path": image_path,
                        "image_id": str(item["image"]),
                        "sample_id": str(item["id"]),
                        "question_text": question_text,
                        "refexp": extract_refexp(question_text),
                        "gt_mask": gt_mask,
                        "gt_box": mask_to_box(gt_mask),
                        "gt_point": center_point_from_mask(gt_mask),
                        "gt_points": points_from_mask(gt_mask),
                        "width": image.width,
                        "height": image.height,
                    }
                )
                image.close()

            global_index += 1

        if global_index >= shard.stop:
            break

    return merged


def iter_source_record_batches_shard(
    json_paths: Sequence[str],
    image_root: str,
    max_sample_ratio_per_file: float,
    sample_seed: int,
    idx: int,
    num_parts: int,
    batch_size: int,
) -> Tuple[Iterable[List[Dict[str, Any]]], int]:
    resolver = ImagePathResolver(image_root)
    file_items: List[Tuple[List[Dict[str, Any]], List[int]]] = []
    total_count = 0

    for file_idx, json_path in enumerate(path for path in json_paths if str(path).strip()):
        items = load_json_records(str(json_path))
        target_count = len(items)
        if max_sample_ratio_per_file is not None and float(max_sample_ratio_per_file) > 0:
            ratio = min(max(float(max_sample_ratio_per_file), 0.0), 1.0)
            target_count = min(target_count, max(1, int(round(len(items) * ratio))))

        if target_count < len(items):
            rng = random.Random(int(sample_seed) + file_idx)
            selected_indices = sorted(rng.sample(range(len(items)), target_count))
        else:
            selected_indices = list(range(len(items)))

        file_items.append((items, selected_indices))
        total_count += len(selected_indices)

    shard = sample_indices(total_count, idx, num_parts)
    resolved_batch_size = max(int(batch_size), 1)

    def _iter_batches() -> Iterable[List[Dict[str, Any]]]:
        batch: List[Dict[str, Any]] = []
        global_index = 0
        for items, selected_indices in file_items:
            for item_idx in selected_indices:
                if global_index >= shard.stop:
                    break

                if global_index >= shard.start:
                    item = items[item_idx]
                    image_path = resolver.resolve(item["image"])
                    question_text = str(item["conversations"][0]["value"])
                    gt_mask = decode_rle_mask(item["objects"][0]["rle"])
                    image = Image.open(image_path).convert("RGB")
                    batch.append(
                        {
                            "image_path": image_path,
                            "image_id": str(item["image"]),
                            "sample_id": str(item["id"]),
                            "question_text": question_text,
                            "refexp": extract_refexp(question_text),
                            "gt_mask": gt_mask,
                            "gt_box": mask_to_box(gt_mask),
                            "gt_point": center_point_from_mask(gt_mask),
                            "gt_points": points_from_mask(gt_mask),
                            "width": image.width,
                            "height": image.height,
                        }
                    )
                    image.close()
                    if len(batch) >= resolved_batch_size:
                        yield batch
                        batch = []

                global_index += 1

            if global_index >= shard.stop:
                break
        if batch:
            yield batch

    return _iter_batches(), len(shard)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def jsonl_to_json(jsonl_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    count = 0
    with open(tmp_path, "w", encoding="utf-8") as out_f:
        out_f.write("[")
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    if count:
                        out_f.write(",")
                    out_f.write("\n")
                    out_f.write(line)
                    count += 1
        if count:
            out_f.write("\n")
        out_f.write("]")
        out_f.flush()
        os.fsync(out_f.fileno())
    os.replace(tmp_path, output_path)
    return count
