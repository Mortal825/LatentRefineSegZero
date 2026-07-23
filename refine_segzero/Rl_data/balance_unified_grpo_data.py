import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from training_scripts.refine_segzero.Rl_data.rl_data_utils import write_json
from training_scripts.refine_segzero.common import load_json_records

LOW_IOU_THRESHOLD = 0.2
HIGH_IOU_THRESHOLD = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--val-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--shuffle-seed", type=int, default=42)
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


def _split_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    init_records: List[Dict[str, Any]] = []
    accept_records: List[Dict[str, Any]] = []
    reject_records: List[Dict[str, Any]] = []
    for record in records:
        task_type = str(record.get("task_type", "")).strip().lower()
        if task_type == "init_box":
            init_records.append(record)
            continue
        if task_type != "reflect":
            continue
        try:
            solution = json.loads(str(record.get("solution", "{}")))
        except Exception:
            solution = {}
        decision = str(solution.get("target_decision", "")).strip().lower()
        if decision == "accept":
            accept_records.append(record)
        elif decision == "reject":
            reject_records.append(record)
    return init_records, accept_records, reject_records


def _bucket_init_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = {"low": [], "mid": [], "high": []}
    for record in records:
        bbox_iou = float(record.get("bbox_iou_stage1", 0.0))
        if bbox_iou < LOW_IOU_THRESHOLD:
            buckets["low"].append(record)
        elif bbox_iou < HIGH_IOU_THRESHOLD:
            buckets["mid"].append(record)
        else:
            buckets["high"].append(record)
    return buckets


def _build_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    init_records, accept_records, reject_records = _split_records(records)
    init_buckets = _bucket_init_records(init_records)
    reflect_total = len(accept_records) + len(reject_records)
    total = len(records)
    return {
        "total": total,
        "init_box": len(init_records),
        "init_box_low": len(init_buckets["low"]),
        "init_box_mid": len(init_buckets["mid"]),
        "init_box_high": len(init_buckets["high"]),
        "reflect_total": reflect_total,
        "reflect_accept": len(accept_records),
        "reflect_reject": len(reject_records),
        "init_to_reflect_ratio": (
            round(len(init_records) / reflect_total, 6) if reflect_total > 0 else None
        ),
        "reflect_accept_ratio": (
            round(len(accept_records) / reflect_total, 6) if reflect_total > 0 else None
        ),
        "reflect_reject_ratio": (
            round(len(reject_records) / reflect_total, 6) if reflect_total > 0 else None
        ),
    }


def _allocate_evenly(total: int, labels: List[str]) -> Dict[str, int]:
    base = total // len(labels)
    remainder = total % len(labels)
    allocation = {label: base for label in labels}
    for label in labels[:remainder]:
        allocation[label] += 1
    return allocation


def _sample_init_box_records(
    init_records: List[Dict[str, Any]],
    target_total: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    buckets = _bucket_init_records(init_records)
    for bucket_records in buckets.values():
        rng.shuffle(bucket_records)

    labels = ["low", "mid", "high"]
    target_by_bucket = _allocate_evenly(target_total, labels)
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

    flat_selected = list(selected["low"]) + list(selected["mid"]) + list(selected["high"])
    rng.shuffle(flat_selected)
    actual_counts = {label: len(selected[label]) for label in labels}
    return flat_selected, {
        "target_total": target_total,
        "target_by_bucket": target_by_bucket,
        "selected_by_bucket": actual_counts,
        "shortage_by_bucket": shortage,
        "unfilled_after_reallocation": max(carryover, 0),
    }


def _rebalance_records(records: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    init_records, accept_records, reject_records = _split_records(records)
    rng = random.Random(seed)
    rng.shuffle(accept_records)
    rng.shuffle(reject_records)
    rng.shuffle(init_records)

    reject_keep = list(reject_records)
    target_reject = len(reject_keep)
    accept_target = target_reject * 2
    accept_keep = accept_records[:accept_target]
    init_target = min(len(init_records), target_reject)
    init_keep, init_sampling = _sample_init_box_records(init_records, init_target, rng)

    balanced = list(init_keep) + list(accept_keep) + list(reject_keep)
    rng.shuffle(balanced)

    return balanced, {
        "target_reject_preserved": target_reject,
        "target_init_box": init_target,
        "target_reflect_accept": accept_target,
        "selected_init_box": len(init_keep),
        "selected_reflect_accept": len(accept_keep),
        "selected_reflect_reject": len(reject_keep),
        "shortage": {
            "accept_shortfall": max(0, accept_target - len(accept_keep)),
            "init_shortfall": max(0, init_target - len(init_keep)),
        },
        "init_box_sampling": init_sampling,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite_output:
        _cleanup_output_dir(output_dir)

    train_records = _load(args.train_path)
    val_records = _load(args.val_path)

    balanced_train, train_selection = _rebalance_records(train_records, args.shuffle_seed)
    balanced_val = list(val_records)
    val_selection = {
        "mode": "passthrough",
        "reason": "validation set keeps all records without ratio balancing",
        "selected_total": len(balanced_val),
    } if val_records else {}

    write_json(output_dir / "train.json", balanced_train)
    if balanced_val:
        write_json(output_dir / "val.json", balanced_val)

    summary = {
        "input_paths": {
            "train": args.train_path,
            "val": args.val_path,
        },
        "sampling_rule": {
            "init_box_to_accept_to_reject": "1:2:1",
            "init_box_internal_iou_buckets": "1:1:1",
            "preserve_all_reject": True,
            "thresholds": {
                "low_lt": LOW_IOU_THRESHOLD,
                "mid_range": [LOW_IOU_THRESHOLD, HIGH_IOU_THRESHOLD],
                "high_gte": HIGH_IOU_THRESHOLD,
            },
            "selection_method": (
                "keep all reject, sample accept to 2x reject count, sample init_box to reject count, "
                "then distribute init_box across low/mid/high iou buckets as evenly as possible with spillover fill"
            ),
        },
        "original": {
            "train": _build_stats(train_records),
            "val": _build_stats(val_records),
        },
        "balanced": {
            "train": _build_stats(balanced_train),
            "val": _build_stats(balanced_val),
        },
        "selection": {
            "train": train_selection,
            "val": val_selection,
        },
    }
    write_json(output_dir / "data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
