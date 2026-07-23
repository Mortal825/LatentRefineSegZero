import json
from typing import Dict

from training_scripts.refine_segzero.prompts import (
    QUERY_REFLECT_INIT_BOX_ANSWER_EXAMPLE,
    build_query_reflect_init_box_prompt,
)


def build_init_box_record(sample: Dict[str, object]) -> Dict[str, object]:
    prompt_override = build_query_reflect_init_box_prompt(str(sample["refexp"]))
    solution = {
        "task_type": "init_box",
        "gt_box": sample["gt_box"],
        "gt_point": sample["gt_point"],
        "gt_points": sample.get("gt_points", [sample["gt_point"], sample["gt_point"]]),
        "image_size": [sample["width"], sample["height"]],
        "image_id": sample["image_id"],
        "sample_id": sample["sample_id"],
    }
    return {
        "problem": sample["refexp"],
        "solution": json.dumps(solution, ensure_ascii=False),
        "task_type": "init_box",
        "data_source": "init_box",
        "image_paths": [sample["image_path"]],
        "prompt_override": prompt_override,
        "answer_format_example": QUERY_REFLECT_INIT_BOX_ANSWER_EXAMPLE,
        "meta_image_id": sample["image_id"],
        "meta_sample_id": sample["sample_id"],
    }


__all__ = ["QUERY_REFLECT_INIT_BOX_ANSWER_EXAMPLE", "build_init_box_record"]
