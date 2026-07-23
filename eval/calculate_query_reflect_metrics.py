import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_parts(output_dir: Path) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("output_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            merged.extend(json.load(f))
    return merged


def _aggregate_metrics(records: List[Dict[str, Any]], metric_kind: str) -> Dict[str, Any]:
    if not records:
        if metric_kind == "bbox":
            return {
                "num_samples": 0,
                "bbox_iou_mean": 0.0,
                "bbox_acc_at_0_5": 0.0,
                "error_count": 0,
            }
        return {
            "num_samples": 0,
            "gIoU": 0.0,
            "cIoU": 0.0,
            "bbox_iou_mean": 0.0,
            "bbox_acc_at_0_5": 0.0,
            "error_count": 0,
        }

    if metric_kind == "bbox":
        bbox_ious = []
        bbox_hits = []
        error_count = 0
        for item in records:
            if "bbox_iou" in item and item["bbox_iou"] is not None:
                bbox_iou = float(item["bbox_iou"])
                bbox_ious.append(bbox_iou)
                bbox_hits.append(1.0 if bbox_iou > 0.5 else 0.0)
            if item.get("error"):
                error_count += 1
        return {
            "num_samples": len(records),
            "bbox_iou_mean": float(np.mean(bbox_ious)) if bbox_ious else 0.0,
            "bbox_acc_at_0_5": float(np.mean(bbox_hits)) if bbox_hits else 0.0,
            "error_count": int(error_count),
        }

    ious = []
    total_intersection = 0
    total_union = 0
    bbox_ious = []
    bbox_hits = []
    error_count = 0

    for item in records:
        intersection = int(item.get("intersection", 0))
        union = int(item.get("union", 0))
        iou = float(item.get("iou", (intersection / union) if union > 0 else 0.0))
        ious.append(iou)
        total_intersection += intersection
        total_union += union

        if "bbox_iou" in item and item["bbox_iou"] is not None:
            bbox_iou = float(item["bbox_iou"])
            bbox_ious.append(bbox_iou)
            bbox_hits.append(1.0 if bbox_iou > 0.5 else 0.0)

        if item.get("error"):
            error_count += 1

    return {
        "num_samples": len(records),
        "gIoU": float(np.mean(ious)) if ious else 0.0,
        "cIoU": float(total_intersection / total_union) if total_union > 0 else 0.0,
        "bbox_iou_mean": float(np.mean(bbox_ious)) if bbox_ious else 0.0,
        "bbox_acc_at_0_5": float(np.mean(bbox_hits)) if bbox_hits else 0.0,
        "error_count": int(error_count),
    }


def _build_oracle_style_summary(
    output_dir: Path,
    records: List[Dict[str, Any]],
    metric_kind: str,
) -> Dict[str, Any]:
    aligned_records = _load_parts(output_dir / "aligned_only")
    direct_records = _load_parts(output_dir / "direct_only")
    total = max(len(records), 1)
    aligned_count = sum(
        1
        for item in records
        if item.get("selected_branch") in {"accept_aligned", "aligned_no_reflect"}
    )
    direct_count = sum(
        1 for item in records if item.get("selected_branch") in {"reject_direct", "parse_error_direct"}
    )
    return {
        "oracle_final": _aggregate_metrics(records, metric_kind),
        "aligned_only": _aggregate_metrics(aligned_records, metric_kind),
        "direct_only": _aggregate_metrics(direct_records, metric_kind),
        "num_samples": len(records),
        "fallback_rate": float(direct_count / total),
        "selected_branch_counts": {
            "aligned": int(aligned_count),
            "direct": int(direct_count),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metric-kind", choices=["mask", "bbox"], default="mask")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = _load_parts(output_dir)
    if not records:
        raise ValueError(f"No prediction parts found under {output_dir}")

    final_metrics = _aggregate_metrics(records, args.metric_kind)
    branch_summary = {
        "num_samples": len(records),
        "accept_aligned_rate": sum(1 for item in records if item.get("selected_branch") == "accept_aligned") / max(len(records), 1),
        "aligned_no_reflect_rate": sum(1 for item in records if item.get("selected_branch") == "aligned_no_reflect") / max(len(records), 1),
        "reject_direct_rate": sum(1 for item in records if item.get("selected_branch") == "reject_direct") / max(len(records), 1),
        "parse_error_direct_rate": sum(1 for item in records if item.get("selected_branch") == "parse_error_direct") / max(len(records), 1),
        "error_rate": sum(1 for item in records if str(item.get("error", "")).strip()) / max(len(records), 1),
    }
    if args.metric_kind == "bbox":
        summary = {**final_metrics, **branch_summary}
    else:
        summary = {
            **final_metrics,
            "mean_iou": sum(float(item.get("iou", 0.0)) for item in records) / max(len(records), 1),
            "mean_bbox_iou": sum(float(item.get("bbox_iou", 0.0)) for item in records) / max(len(records), 1),
            **branch_summary,
        }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric_kind": args.metric_kind,
                "summary": summary,
                "oracle_style_summary": _build_oracle_style_summary(output_dir, records, args.metric_kind),
                "results": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
