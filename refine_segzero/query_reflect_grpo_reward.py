import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from training_scripts.refine_segzero.common import compute_box_iou


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
CONCLUSION_RE = re.compile(r"conclusion\b", re.IGNORECASE)


def _parse_json_answer(predict_str: str) -> Tuple[Optional[Any], str]:
    match = ANSWER_RE.search(predict_str)
    if not match:
        return None, "missing_answer"
    try:
        return json.loads(match.group(1).strip()), ""
    except Exception:
        return None, "invalid_json"


def _extract_think_conclusion_decision(predict_str: str) -> str:
    match = THINK_RE.search(predict_str)
    if not match:
        return ""

    think = match.group(1)
    conclusion_matches = list(CONCLUSION_RE.finditer(think))
    if not conclusion_matches:
        return ""

    conclusion_text = think[conclusion_matches[-1].end():]
    decisions = re.findall(r"\b(accept|reject)\b", conclusion_text, flags=re.IGNORECASE)
    unique_decisions = {decision.lower() for decision in decisions}
    if len(unique_decisions) != 1:
        return ""
    return decisions[0].lower()


def _safe_float(value: Any) -> Tuple[float, bool]:
    if value is None:
        return 0.0, False
    try:
        return float(value), True
    except Exception:
        return 0.0, False


def _point_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _point_reward(pred_point: Sequence[float], gt_point: Sequence[float], width: int, height: int) -> float:
    norm = max(float(width + height), 1.0)
    return max(0.0, 1.0 - (_point_distance(pred_point, gt_point) / norm))


def _normalized_box_l1(pred_box: Sequence[float], gt_box: Sequence[float], width: int, height: int) -> float:
    norm = max(float(width + height), 1.0)
    return sum(abs(float(a) - float(b)) for a, b in zip(pred_box, gt_box)) / norm


def _safe_float_list(value: Any, length: int) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) < length:
        return None
    try:
        return [float(v) for v in value[:length]]
    except Exception:
        return None


def _point_in_box(point: Sequence[float], box: Sequence[float]) -> bool:
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def _extract_box_points(answer: Any) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    if isinstance(answer, list) and answer:
        answer = answer[0]
    if not isinstance(answer, dict):
        return None, None, None
    box = _safe_float_list(answer.get("bbox"), 4)
    point_1 = _safe_float_list(answer.get("points_1"), 2)
    point_2 = _safe_float_list(answer.get("points_2"), 2)
    if box is None or point_1 is None or point_2 is None:
        return None, None, None
    return box, point_1, point_2


def _target_gt_points(target: Dict[str, Any]) -> List[List[float]]:
    gt_points = target.get("gt_points")
    if isinstance(gt_points, list) and len(gt_points) >= 2:
        point_1 = _safe_float_list(gt_points[0], 2)
        point_2 = _safe_float_list(gt_points[1], 2)
        if point_1 is not None and point_2 is not None:
            return [point_1, point_2]
    gt_point = _safe_float_list(target.get("gt_point"), 2) or [0.0, 0.0]
    return [gt_point, gt_point]


def _point_pair_rewards(
    pred_points: Sequence[Sequence[float]],
    gt_points: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> Tuple[float, float, float]:
    norm = max(float(width + height), 1.0)
    direct = _point_distance(pred_points[0], gt_points[0]) + _point_distance(pred_points[1], gt_points[1])
    swapped = _point_distance(pred_points[0], gt_points[1]) + _point_distance(pred_points[1], gt_points[0])
    if direct <= swapped:
        distances = [
            _point_distance(pred_points[0], gt_points[0]),
            _point_distance(pred_points[1], gt_points[1]),
        ]
    else:
        distances = [
            _point_distance(pred_points[0], gt_points[1]),
            _point_distance(pred_points[1], gt_points[0]),
        ]
    point_1 = max(0.0, 1.0 - distances[0] / norm)
    point_2 = max(0.0, 1.0 - distances[1] / norm)
    return (point_1 + point_2) / 2.0, point_1, point_2


def _extract_reflect_answer(answer: Any) -> Dict[str, Any]:
    if not isinstance(answer, dict):
        return {}
    has_extra_keys = any(
        key not in {"decision"} for key in answer.keys()
    )
    return {
        "decision": str(answer.get("decision", "")).lower().strip(),
        "has_extra_keys": has_extra_keys,
    }


def _score_init_box(answer: Any, target: Dict[str, Any]) -> Dict[str, float]:
    gt_box = target["gt_box"]
    gt_points = _target_gt_points(target)
    width, height = int(target["image_size"][0]), int(target["image_size"][1])
    pred_box, pred_point_1, pred_point_2 = _extract_box_points(answer)
    if pred_box is None or pred_point_1 is None or pred_point_2 is None:
        return {
            "reward/format": -0.1,
            "reward/box_iou": 0.0,
            "reward/box_l1": 0.0,
            "reward/point": 0.0,
            "reward/point_1": 0.0,
            "reward/point_2": 0.0,
            "reward/points_in_box": 0.0,
            "reward/total": 0.0,
        }
    box_iou = float(compute_box_iou([int(round(v)) for v in pred_box], [int(v) for v in gt_box]))
    box_l1 = max(0.0, 1.0 - _normalized_box_l1(pred_box, gt_box, width=width, height=height))
    points_in_box = 1.0 if _point_in_box(pred_point_1, pred_box) and _point_in_box(pred_point_2, pred_box) else 0.0
    if points_in_box:
        point, point_1, point_2 = _point_pair_rewards([pred_point_1, pred_point_2], gt_points, width=width, height=height)
    else:
        point, point_1, point_2 = 0.0, 0.0, 0.0
    return {
        "reward/format": 0.1,
        "reward/box_iou": box_iou,
        "reward/box_l1": box_l1,
        "reward/point": point,
        "reward/point_1": point_1,
        "reward/point_2": point_2,
        "reward/points_in_box": points_in_box,
        "reward/total": 0.1 + 0.3 * box_iou + 0.3 * box_l1 + 0.3 * point,
    }


def _score_reflect(answer: Any, target: Dict[str, Any], predict_str: str) -> Dict[str, float]:
    parsed = _extract_reflect_answer(answer)
    target_decision = str(target["target_decision"]).lower().strip()
    decision = parsed.get("decision", "")
    conclusion_decision = _extract_think_conclusion_decision(predict_str)

    valid_schema = (
        decision in {"accept", "reject"}
        and not bool(parsed.get("has_extra_keys"))
    )
    format_reward = 0.1 if valid_schema else -0.1

    conclusion_matches_decision = (
        conclusion_decision in {"accept", "reject"}
        and decision in {"accept", "reject"}
        and conclusion_decision == decision
    )
    conclusion_format_reward = 0.1 if conclusion_matches_decision else -0.1
    format_reward += conclusion_format_reward

    if decision == target_decision:
        decision_reward = 1.0 if decision == 'reject' else 1.0
    else:
        decision_reward = -1.0 if decision == 'reject' else -1.0
    total = format_reward + decision_reward

    accept_target_count = 1.0 if decision == "accept" else 0.0
    reject_target_count = 1.0 if decision == "reject" else 0.0
    accept_target_correct = 1.0 if target_decision == "accept" and decision == "accept" else 0.0
    reject_target_correct = 1.0 if target_decision == "reject" and decision == "reject" else 0.0
    return {
        "reward/format": format_reward,
        "reward/think_conclusion_format": conclusion_format_reward,
        "reward/think_conclusion_matches_decision": 1.0 if conclusion_matches_decision else 0.0,
        "reward/decision": decision_reward,
        "reward/reflect_total_count": 1.0,
        "reward/accept_target_count": accept_target_count,
        "reward/accept_target_correct": accept_target_correct,
        "reward/reject_target_count": reject_target_count,
        "reward/reject_target_correct": reject_target_correct,
        "reward/total": total,
    }


def query_reflect_grpo_compute_score(predict_str: str, ground_truth: str) -> Dict[str, Any]:
    target = json.loads(ground_truth)
    task_type = str(target.get("task_type", "init_box"))
    answer, parse_error = _parse_json_answer(predict_str)
    if parse_error:
        metrics = {
            "reward/format": 0.0,
            "reward/total": 0.0,
            "reward/task_type_init_box": 1.0 if task_type != "reflect" else 0.0,
            "reward/task_type_reflect": 1.0 if task_type == "reflect" else 0.0,
        }
        if task_type == "reflect":
            metrics["reward/format"] = -0.3
            metrics["reward/total"] = -0.3
            metrics.update(
                {
                    "reward/decision": -1.0,
                    "reward/think_conclusion_format": -0.3,
                    "reward/think_conclusion_matches_decision": 0.0,
                    "reward/reflect_total_count": 1.0,
                    "reward/accept_target_count": 0.0,
                    "reward/accept_target_correct": 0.0,
                    "reward/reject_target_count": 0.0,
                    "reward/reject_target_correct": 0.0,
                }
            )
        else:
            metrics.update(
                {
                    "reward/box_iou": 0.0,
                    "reward/box_l1": 0.0,
                    "reward/point": 0.0,
                    "reward/point_1": 0.0,
                    "reward/point_2": 0.0,
                    "reward/points_in_box": 0.0,
                }
            )
        return {"score": float(metrics["reward/total"]), "metrics": metrics}

    if task_type == "reflect":
        metrics = _score_reflect(answer, target, predict_str)
    else:
        metrics = _score_init_box(answer, target)
    metrics["reward/task_type_init_box"] = 1.0 if task_type != "reflect" else 0.0
    metrics["reward/task_type_reflect"] = 1.0 if task_type == "reflect" else 0.0
    return {"score": float(metrics["reward/total"]), "metrics": metrics}
