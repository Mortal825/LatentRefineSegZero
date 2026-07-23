import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from training_scripts.refine_segzero.Rl_data.rl_data_utils import write_json
from training_scripts.refine_segzero.common import load_json_records

LOW_IOU_THRESHOLD = 0.2
HIGH_IOU_THRESHOLD = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-train-path", type=str, required=True)
    parser.add_argument("--init-val-path", type=str, default="")
    parser.add_argument("--reflect-train-path", type=str, required=True)
    parser.add_argument("--reflect-val-path", type=str, default="")
    parser.add_argument("--stage1-cache-train-path", type=str, required=True)
    parser.add_argument("--stage1-cache-val-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--reflect-train-correct-min", type=int, default=-1)
    parser.add_argument("--reflect-train-correct-max", type=int, default=-1)
    parser.add_argument("--reflect-train-accept-correct-min", type=int, default=-1)
    parser.add_argument("--reflect-train-accept-correct-max", type=int, default=-1)
    parser.add_argument("--reflect-train-reject-correct-min", type=int, default=-1)
    parser.add_argument("--reflect-train-reject-correct-max", type=int, default=-1)
    parser.add_argument("--init-train-sample-count", type=int, default=-1)
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def _cleanup_output_dir(output_dir: Path) -> None:
    targets = [
        output_dir / "train.json",
        output_dir / "val.json",
        output_dir / "data_summary.json",
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _load(path: str) -> List[Dict[str, Any]]:
    if not str(path).strip():
        return []
    return load_json_records(path)


def _target_decision(record: Dict[str, Any]) -> str:
    try:
        solution = json.loads(str(record.get("solution", "{}")))
    except Exception:
        solution = {}
    return str(solution.get("target_decision", "")).strip().lower()


def _reflect_decision_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    accept_count = 0
    reject_count = 0
    other_count = 0
    for record in records:
        decision = _target_decision(record)
        if decision == "accept":
            accept_count += 1
        elif decision == "reject":
            reject_count += 1
        else:
            other_count += 1
    return {
        "accept": accept_count,
        "reject": reject_count,
        "other": other_count,
    }


def _bucket_init_records(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {"low": [], "mid": [], "high": []}
    for record in records:
        try:
            bbox_iou = float(record.get("bbox_iou_stage1", 0.0))
        except Exception:
            bbox_iou = 0.0
        if bbox_iou < LOW_IOU_THRESHOLD:
            buckets["low"].append(record)
        elif bbox_iou < HIGH_IOU_THRESHOLD:
            buckets["mid"].append(record)
        else:
            buckets["high"].append(record)
    return buckets


def _allocate_evenly(total: int, labels: Sequence[str]) -> Dict[str, int]:
    base = total // len(labels)
    remainder = total % len(labels)
    allocation = {label: base for label in labels}
    for label in labels[:remainder]:
        allocation[label] += 1
    return allocation


def _sample_init_box_records(
    records: Sequence[Dict[str, Any]],
    target_total: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if target_total < 0 or len(records) <= target_total:
        kept = list(records)
        buckets = _bucket_init_records(kept)
        return kept, {
            "enabled": target_total >= 0,
            "target_total": target_total if target_total >= 0 else None,
            "before_total": len(records),
            "after_total": len(kept),
            "filtered_out": len(records) - len(kept),
            "thresholds": {
                "low_lt": LOW_IOU_THRESHOLD,
                "mid_range": [LOW_IOU_THRESHOLD, HIGH_IOU_THRESHOLD],
                "high_gte": HIGH_IOU_THRESHOLD,
            },
            "target_by_bucket": None,
            "selected_by_bucket": {label: len(bucket_records) for label, bucket_records in buckets.items()},
            "shortage_by_bucket": {label: 0 for label in buckets},
            "unfilled_after_reallocation": 0,
        }

    rng = random.Random(seed)
    buckets = _bucket_init_records(records)
    for bucket_records in buckets.values():
        rng.shuffle(bucket_records)

    labels = ["low", "mid", "high"]
    target_by_bucket = _allocate_evenly(int(target_total), labels)
    selected: Dict[str, List[Dict[str, Any]]] = {label: [] for label in labels}
    shortage: Dict[str, int] = {}
    carryover = 0
    spare_labels: List[str] = []

    for label in labels:
        target = target_by_bucket[label]
        available = len(buckets[label])
        take = min(available, target)
        selected[label] = buckets[label][:take]
        shortage[label] = max(0, target - take)
        carryover += shortage[label]
        if available > take:
            spare_labels.append(label)

    for label in spare_labels:
        if carryover <= 0:
            break
        available_records = buckets[label][len(selected[label]) :]
        extra_take = min(len(available_records), carryover)
        if extra_take > 0:
            selected[label].extend(available_records[:extra_take])
            carryover -= extra_take

    kept = list(selected["low"]) + list(selected["mid"]) + list(selected["high"])
    rng.shuffle(kept)
    selected_by_bucket = {label: len(selected[label]) for label in labels}
    return kept, {
        "enabled": True,
        "target_total": int(target_total),
        "before_total": len(records),
        "after_total": len(kept),
        "filtered_out": len(records) - len(kept),
        "thresholds": {
            "low_lt": LOW_IOU_THRESHOLD,
            "mid_range": [LOW_IOU_THRESHOLD, HIGH_IOU_THRESHOLD],
            "high_gte": HIGH_IOU_THRESHOLD,
        },
        "target_by_bucket": target_by_bucket,
        "selected_by_bucket": selected_by_bucket,
        "shortage_by_bucket": shortage,
        "unfilled_after_reallocation": max(carryover, 0),
    }


def _filter_reflect_by_correct_count(
    records: Sequence[Dict[str, Any]],
    min_count: int,
    max_count: int,
    accept_min_count: int,
    accept_max_count: int,
    reject_min_count: int,
    reject_max_count: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    accept_min = accept_min_count if accept_min_count >= 0 else min_count
    accept_max = accept_max_count if accept_max_count >= 0 else max_count
    reject_min = reject_min_count if reject_min_count >= 0 else min_count
    reject_max = reject_max_count if reject_max_count >= 0 else max_count
    filter_enabled = any(value >= 0 for value in (accept_min, accept_max, reject_min, reject_max))
    if not filter_enabled:
        kept = list(records)
        decision_counts = _reflect_decision_counts(kept)
        return kept, {
            "enabled": False,
            "correct_count_filter": None,
            "accept_correct_count_filter": None,
            "reject_correct_count_filter": None,
            "before_total": len(records),
            "after_total": len(kept),
            "filtered_out": 0,
            "before_accept": decision_counts["accept"],
            "before_reject": decision_counts["reject"],
            "after_accept": decision_counts["accept"],
            "after_reject": decision_counts["reject"],
            "missing_correct_count": 0,
            "unknown_decision_count": decision_counts["other"],
        }

    accept_lower = max(int(accept_min), 0) if accept_min >= 0 else 0
    accept_upper = int(accept_max) if accept_max >= 0 else 10**9
    reject_lower = max(int(reject_min), 0) if reject_min >= 0 else 0
    reject_upper = int(reject_max) if reject_max >= 0 else 10**9
    kept: List[Dict[str, Any]] = []
    missing_correct_count = 0
    unknown_decision_count = 0
    for record in records:
        if "reflect_sample_correct_count" not in record:
            missing_correct_count += 1
            continue
        try:
            correct_count = int(record.get("reflect_sample_correct_count"))
        except Exception:
            missing_correct_count += 1
            continue
        decision = _target_decision(record)
        if decision == "accept":
            lower, upper = accept_lower, accept_upper
        elif decision == "reject":
            lower, upper = reject_lower, reject_upper
        else:
            unknown_decision_count += 1
            continue
        if lower <= correct_count <= upper:
            kept.append(record)

    before_counts = _reflect_decision_counts(records)
    after_counts = _reflect_decision_counts(kept)
    return kept, {
        "enabled": True,
        "correct_count_filter": {
            "accept": [accept_lower, accept_upper],
            "reject": [reject_lower, reject_upper],
        },
        "accept_correct_count_filter": [accept_lower, accept_upper],
        "reject_correct_count_filter": [reject_lower, reject_upper],
        "before_total": len(records),
        "after_total": len(kept),
        "filtered_out": len(records) - len(kept),
        "before_accept": before_counts["accept"],
        "before_reject": before_counts["reject"],
        "after_accept": after_counts["accept"],
        "after_reject": after_counts["reject"],
        "missing_correct_count": missing_correct_count,
        "unknown_decision_count": unknown_decision_count,
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _init_box_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(record.get("meta_sample_id", "")).strip(),
        _normalize_text(str(record.get("problem", ""))),
    )


def _stage1_cache_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(record.get("sample_id", "")).strip(),
        _normalize_text(str(record.get("question", ""))),
    )


def _build_stage1_index(stage1_records: Sequence[Dict[str, Any]]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], int]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    duplicate_count = 0
    for record in stage1_records:
        key = _stage1_cache_key(record)
        if key in index:
            duplicate_count += 1
            continue
        index[key] = record
    return index, duplicate_count


def _enrich_init_box_records(
    init_records: Sequence[Dict[str, Any]],
    stage1_records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stage1_index, duplicate_count = _build_stage1_index(stage1_records)
    enriched: List[Dict[str, Any]] = []
    missing_keys: List[Tuple[str, str]] = []
    for record in init_records:
        key = _init_box_key(record)
        stage1_payload = stage1_index.get(key)
        if stage1_payload is None:
            missing_keys.append(key)
            continue
        merged = dict(record)
        merged["bbox_iou_stage1"] = float(stage1_payload.get("bbox_iou_stage1", 0.0))
        merged["mask_iou_stage1"] = float(stage1_payload.get("mask_iou_stage1", 0.0))
        merged["stage1_box"] = stage1_payload.get("stage1_box", [0, 0, 0, 0])
        merged["stage1_points"] = stage1_payload.get("stage1_points", [[0, 0], [0, 0]])
        merged["stage1_mask_rle"] = stage1_payload.get("stage1_mask_rle")
        if stage1_payload.get("proposal_visualization_path"):
            merged["proposal_visualization_path"] = stage1_payload.get("proposal_visualization_path")
        if stage1_payload.get("crop_image_path"):
            merged["crop_image_path"] = stage1_payload.get("crop_image_path")
        enriched.append(merged)
    if missing_keys:
        preview = ", ".join(f"sample_id={sample_id}, text={text}" for sample_id, text in missing_keys[:3])
        raise ValueError(
            f"Failed to match {len(missing_keys)} init_box records to stage1 cache using (sample_id, normalized_text). "
            f"Examples: {preview}"
        )
    return enriched, {
        "init_box_enriched_count": len(enriched),
        "init_box_missing_stage1_match_count": 0,
        "duplicate_stage1_key_count": duplicate_count,
    }


def _merge_records(init_records: List[Dict[str, Any]], reflect_records: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    merged = list(init_records) + list(reflect_records)
    rng = random.Random(seed)
    rng.shuffle(merged)
    return merged


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite_output:
        _cleanup_output_dir(output_dir)

    init_train = _load(args.init_train_path)
    init_val = _load(args.init_val_path)
    reflect_train = _load(args.reflect_train_path)
    reflect_val = _load(args.reflect_val_path)
    stage1_cache_train = _load(args.stage1_cache_train_path)
    stage1_cache_val = _load(args.stage1_cache_val_path)

    init_train_enriched, train_enrich_stats = _enrich_init_box_records(init_train, stage1_cache_train)
    init_val_enriched, val_enrich_stats = _enrich_init_box_records(init_val, stage1_cache_val)
    init_train_sampled, init_train_sampling_stats = _sample_init_box_records(
        init_train_enriched,
        target_total=int(args.init_train_sample_count),
        seed=int(args.shuffle_seed),
    )
    reflect_train_filtered, reflect_train_filter_stats = _filter_reflect_by_correct_count(
        reflect_train,
        min_count=int(args.reflect_train_correct_min),
        max_count=int(args.reflect_train_correct_max),
        accept_min_count=int(args.reflect_train_accept_correct_min),
        accept_max_count=int(args.reflect_train_accept_correct_max),
        reject_min_count=int(args.reflect_train_reject_correct_min),
        reject_max_count=int(args.reflect_train_reject_correct_max),
    )

    train_records = _merge_records(init_train_sampled, reflect_train_filtered, args.shuffle_seed)
    val_records = _merge_records(init_val_enriched, reflect_val, args.shuffle_seed + 1)

    write_json(output_dir / "train.json", train_records)
    if val_records:
        write_json(output_dir / "val.json", val_records)
    write_json(
        output_dir / "data_summary.json",
        {
            "train_total": len(train_records),
            "val_total": len(val_records),
            "train_init_box": len(init_train_sampled),
            "train_reflect": len(reflect_train_filtered),
            "val_init_box": len(init_val_enriched),
            "val_reflect": len(reflect_val),
            "train_init_box_before_sampling": init_train_sampling_stats["before_total"],
            "train_init_box_after_sampling": init_train_sampling_stats["after_total"],
            "train_init_box_filtered_out": init_train_sampling_stats["filtered_out"],
            "train_init_box_low_after_sampling": init_train_sampling_stats["selected_by_bucket"]["low"],
            "train_init_box_mid_after_sampling": init_train_sampling_stats["selected_by_bucket"]["mid"],
            "train_init_box_high_after_sampling": init_train_sampling_stats["selected_by_bucket"]["high"],
            "train_reflect_before_filter": reflect_train_filter_stats["before_total"],
            "train_reflect_after_filter": reflect_train_filter_stats["after_total"],
            "train_reflect_filtered_out": reflect_train_filter_stats["filtered_out"],
            "train_reflect_accept_before_filter": reflect_train_filter_stats["before_accept"],
            "train_reflect_reject_before_filter": reflect_train_filter_stats["before_reject"],
            "train_reflect_accept_after_filter": reflect_train_filter_stats["after_accept"],
            "train_reflect_reject_after_filter": reflect_train_filter_stats["after_reject"],
            "reflect_train_correct_count_filter": reflect_train_filter_stats["correct_count_filter"],
            "reflect_train_accept_correct_count_filter": reflect_train_filter_stats["accept_correct_count_filter"],
            "reflect_train_reject_correct_count_filter": reflect_train_filter_stats["reject_correct_count_filter"],
            "reflect_train_missing_correct_count": reflect_train_filter_stats["missing_correct_count"],
            "reflect_train_unknown_decision_count": reflect_train_filter_stats["unknown_decision_count"],
            "init_train_path": args.init_train_path,
            "reflect_train_path": args.reflect_train_path,
            "stage1_cache_train_path": args.stage1_cache_train_path,
            "stage1_cache_val_path": args.stage1_cache_val_path,
            "train_init_box_enrich": train_enrich_stats,
            "val_init_box_enrich": val_enrich_stats,
            "train_init_box_sampling": init_train_sampling_stats,
            "reflect_train_filter": reflect_train_filter_stats,
        },
    )


if __name__ == "__main__":
    main()
