from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from training_scripts.refine_segzero.common import (
    RefineSample,
    answer_to_prompts,
    build_prediction_record,
    build_sam_image_tensor,
    compute_box_iou,
    mask_to_box,
)
from training_scripts.refine_segzero.data import geometric_collate_fn
from training_scripts.refine_segzero.geometric_query_export import load_geometric_model_from_export


def _scale_answer_to_longest_side(answer: Dict[str, Any], width: int, height: int, resize_size: int) -> Dict[str, Any]:
    scale = float(resize_size) / float(max(width, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    x_factor = width / float(resized_width)
    y_factor = height / float(resized_height)
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


def _sanitize_bbox_points(answer: Dict[str, Any], width: int, height: int, resize_size: int) -> Dict[str, Any]:
    scaled_answer = _scale_answer_to_longest_side(answer, width=width, height=height, resize_size=resize_size)
    bbox, points = answer_to_prompts(scaled_answer)
    return {"bbox": bbox, "points": points}


def _collate_samples_for_oracle(
    model: Any,
    samples: Sequence[RefineSample],
    resize_size: int,
    sam_image_size: int,
    max_pixels: int,
    min_pixels: int,
) -> Dict[str, Any]:
    return geometric_collate_fn(
        list(samples),
        processor=model.processor,
        resize_size=resize_size,
        sam_image_size=sam_image_size,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )


def _resolve_sample_image(sample: RefineSample) -> Image.Image:
    cached_image = sample.raw_item.get("image_obj") if isinstance(sample.raw_item, dict) else None
    if isinstance(cached_image, Image.Image):
        return cached_image.convert("RGB")
    return Image.open(sample.image_path).convert("RGB")


def _run_branch_from_collated_batch(
    model: Any,
    collated_batch: Dict[str, Any],
    idx: int,
    branch_embedding: torch.Tensor,
    *,
    input_box: Optional[List[int]],
    input_points: Optional[List[List[int]]],
    use_points: bool,
    refiner_points: Optional[List[List[int]]] = None,
    mode = "aligned",
) -> Dict[str, Any]:
    return model._segment_branch(
        sam_image=collated_batch["sam_images"][idx].to(model.qwen.device),
        image_size=collated_batch["image_sizes"][idx],
        gt_mask=collated_batch["gt_masks"][idx].to(device=model.qwen.device),
        gt_box=collated_batch["gt_boxes"][idx],
        branch_embedding=branch_embedding,
        input_box=input_box,
        input_points=input_points,
        refiner_points=refiner_points,
        use_points=use_points,
        mode = mode,
    )


def _mask_branch_record(
    image_id: str,
    sample_id: str,
    question: str,
    output_text: str,
    think: str,
    pred_bbox: List[int],
    pred_points: List[List[int]],
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    branch_type: str,
    extra: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Dict[str, Any]:
    return build_prediction_record(
        image_id=image_id,
        sample_id=sample_id,
        question=question,
        output_text=output_text,
        think=think,
        pred_bbox=pred_bbox,
        pred_points=pred_points,
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        branch_type=branch_type,
        extra=extra,
        error=error,
    )


def _bbox_branch_record(
    image_id: str,
    sample_id: str,
    question: str,
    output_text: str,
    think: str,
    pred_bbox: List[int],
    pred_points: List[List[int]],
    gt_bbox: List[int],
    branch_type: str,
    extra: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Dict[str, Any]:
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
        "branch_type": branch_type,
        "error": error,
    }
    if extra:
        payload.update(extra)
    return payload


def _empty_mask_branch_record(
    image_id: str,
    sample_id: str,
    question: str,
    output_text: str,
    think: str,
    gt_mask: np.ndarray,
    branch_type: str,
    extra: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Dict[str, Any]:
    zero_mask = np.zeros_like(gt_mask, dtype=np.uint8)
    return _mask_branch_record(
        image_id=image_id,
        sample_id=sample_id,
        question=question,
        output_text=output_text,
        think=think,
        pred_bbox=[0, 0, 0, 0],
        pred_points=[[0, 0], [0, 0]],
        pred_mask=zero_mask,
        gt_mask=gt_mask,
        branch_type=branch_type,
        extra=extra,
        error=error,
    )


def _empty_bbox_branch_record(
    image_id: str,
    sample_id: str,
    question: str,
    output_text: str,
    think: str,
    gt_bbox: List[int],
    branch_type: str,
    extra: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Dict[str, Any]:
    return _bbox_branch_record(
        image_id=image_id,
        sample_id=sample_id,
        question=question,
        output_text=output_text,
        think=think,
        pred_bbox=[0, 0, 0, 0],
        pred_points=[[0, 0], [0, 0]],
        gt_bbox=gt_bbox,
        branch_type=branch_type,
        extra=extra,
        error=error,
    )


def _select_final_branch(aligned_metric: float, threshold: float, fallback_metric: str) -> Dict[str, str]:
    if aligned_metric < threshold:
        return {
            "selected_branch": "direct",
            "fallback_reason": f"aligned_{fallback_metric}_below_threshold",
        }
    return {"selected_branch": "aligned", "fallback_reason": ""}


def _attach_branch_breakdown(
    final_record: Dict[str, Any],
    *,
    generated_bbox: List[int],
    generated_points: List[List[int]],
    threshold: float,
    fallback_metric: str,
    selected_branch: str,
    fallback_reason: str,
    aligned_record: Dict[str, Any],
    direct_record: Dict[str, Any],
) -> Dict[str, Any]:
    payload = dict(final_record)
    payload.update(
        {
            "generated_bbox": generated_bbox,
            "generated_points": generated_points,
            "threshold": float(threshold),
            "fallback_metric": fallback_metric,
            "selected_branch": selected_branch,
            "fallback_reason": fallback_reason,
            "aligned_branch": aligned_record,
            "direct_branch": direct_record,
        }
    )
    return payload


def run_geometric_inference(
    model: Any,
    image: Image.Image,
    question: str,
    image_id: str,
    sample_id: str,
    gt_mask: np.ndarray,
    resize_size: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    messages = model.build_generation_messages(image=image, question=question, resize_size=resize_size)
    model_inputs = model._processor_inputs(messages)
    generation = model._generate_once(model_inputs, max_new_tokens=max_new_tokens, do_sample=False)[0][0]
    sample_batch = {
        "model_inputs": model_inputs,
        "sam_images": torch.stack([build_sam_image_tensor(image, 1024)]),
        "gt_masks": [torch.from_numpy(gt_mask.astype(np.float32))],
        "gt_boxes": [mask_to_box(gt_mask)],
        "image_sizes": [(image.height, image.width)],
        "meta": [{"image_id": image_id, "sample_id": sample_id, "question": question, "width": image.width, "height": image.height}],
    }
    output = model.stage1_forward(batch=sample_batch)
    pred = output["aligned_predictions"][0]
    return build_prediction_record(
        image_id=image_id,
        sample_id=sample_id,
        question=question,
        pred_bbox=pred["bbox"],
        pred_points=pred["points"],
        pred_mask=pred["pred_mask"],
        gt_mask=gt_mask,
        output_text=generation.output_text,
        think=generation.think,
    )


def run_oracle_fallback_mask_inference(
    model: Any,
    image: Image.Image,
    question: str,
    image_id: str,
    sample_id: str,
    gt_mask: np.ndarray,
    resize_size: int,
    max_new_tokens: int,
    threshold: float = 0.5,
    fallback_metric: str = "mask_iou",
    sam_image_size: int = 1024,
) -> Dict[str, Dict[str, Any]]:
    sample = RefineSample(
        image_path="",
        image_id=image_id,
        sample_id=sample_id,
        question_text=question,
        refexp=question,
        gt_mask=gt_mask,
        gt_box_xyxy=mask_to_box(gt_mask),
        width=image.width,
        height=image.height,
        raw_item={"image_obj": image},
    )
    collated_batch = _collate_samples_for_oracle(
        model=model,
        samples=[sample],
        resize_size=resize_size,
        sam_image_size=sam_image_size,
        max_pixels=2007040,
        min_pixels=3136,
    )
    model_inputs = collated_batch["model_inputs"]
    generation = model._generate_once(model_inputs, max_new_tokens=max_new_tokens, do_sample=False)[0][0]
    geometry = _sanitize_bbox_points(generation.answer, width=image.width, height=image.height, resize_size=resize_size)
    with torch.no_grad():
        stage1_output = model.stage1_forward(batch=collated_batch)

    cached_hidden, cached_mask = model.extract_cached_query_hidden_for_inputs(model_inputs)
    aligned_embedding, _ = model.build_connector_embeddings(cached_hidden, cached_mask)
    aligned_embedding = aligned_embedding[0]

    common_extra = {
        "generated_bbox": geometry["bbox"],
        "generated_points": geometry["points"],
    }
    try:
        aligned_result = _run_branch_from_collated_batch(
            model=model,
            collated_batch=collated_batch,
            idx=0,
            branch_embedding=aligned_embedding,
            input_box=geometry["bbox"],
            input_points=geometry["points"],
            use_points=True,
        )
        aligned_record = _mask_branch_record(
            image_id=image_id,
            sample_id=sample_id,
            question=question,
            output_text=generation.output_text,
            think=generation.think,
            pred_bbox=aligned_result["pred_box"],
            pred_points=geometry["points"],
            pred_mask=aligned_result["pred_mask"],
            gt_mask=gt_mask,
            branch_type="aligned",
            extra=common_extra,
        )
    except Exception as exc:
        aligned_record = _empty_mask_branch_record(
            image_id=image_id,
            sample_id=sample_id,
            question=question,
            output_text=generation.output_text,
            think=generation.think,
            gt_mask=gt_mask,
            branch_type="aligned",
            extra=common_extra,
            error=str(exc),
        )

    try:
        direct_prediction = stage1_output["direct_predictions"][0]
        direct_record = _mask_branch_record(
            image_id=image_id,
            sample_id=sample_id,
            question=question,
            output_text=generation.output_text,
            think=generation.think,
            pred_bbox=direct_prediction["bbox"],
            pred_points=[[0, 0], [0, 0]],
            pred_mask=direct_prediction["pred_mask"],
            gt_mask=gt_mask,
            branch_type="direct",
            extra=common_extra,
        )
    except Exception as exc:
        direct_record = _empty_mask_branch_record(
            image_id=image_id,
            sample_id=sample_id,
            question=question,
            output_text=generation.output_text,
            think=generation.think,
            gt_mask=gt_mask,
            branch_type="direct",
            extra=common_extra,
            error=str(exc),
        )

    aligned_metric = float(aligned_record["iou"] if fallback_metric == "mask_iou" else aligned_record["bbox_iou"])
    selection = _select_final_branch(aligned_metric, threshold=threshold, fallback_metric=fallback_metric)
    chosen_record = direct_record if selection["selected_branch"] == "direct" else aligned_record
    final_record = _attach_branch_breakdown(
        chosen_record,
        generated_bbox=geometry["bbox"],
        generated_points=geometry["points"],
        threshold=threshold,
        fallback_metric=fallback_metric,
        selected_branch=selection["selected_branch"],
        fallback_reason=selection["fallback_reason"],
        aligned_record=aligned_record,
        direct_record=direct_record,
    )
    return {"final_record": final_record, "aligned_record": aligned_record, "direct_record": direct_record}


def run_oracle_fallback_mask_inference_batch(
    model: Any,
    samples: Sequence[RefineSample],
    resize_size: int,
    max_new_tokens: int,
    threshold: float = 0.5,
    fallback_metric: str = "mask_iou",
    sam_image_size: int = 1024,
    max_pixels: int = 2007040,
    min_pixels: int = 3136,
) -> List[Dict[str, Dict[str, Any]]]:
    collated_batch = _collate_samples_for_oracle(
        model=model,
        samples=samples,
        resize_size=resize_size,
        sam_image_size=sam_image_size,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )
    model_inputs = collated_batch["model_inputs"]
    generations = model._generate_once(model_inputs, max_new_tokens=max_new_tokens, do_sample=False)[0]
    with torch.no_grad():
        stage1_output = model.stage1_forward(batch=collated_batch)
    cached_hidden, cached_mask = model.extract_cached_query_hidden_for_inputs(model_inputs)
    aligned_embeddings, direct_embedding = model.build_connector_embeddings(cached_hidden, cached_mask)

    results: List[Dict[str, Dict[str, Any]]] = []
    for idx, (sample, generation) in enumerate(
        zip(samples, generations)
    ):
        image = _resolve_sample_image(sample)
        question = sample.refexp
        image_id = sample.image_id
        sample_id = sample.sample_id
        gt_mask = sample.gt_mask
        geometry = _sanitize_bbox_points(generation.answer, width=image.width, height=image.height, resize_size=resize_size)
        aligned_embedding = aligned_embeddings[idx]
        common_extra = {
            "generated_bbox": geometry["bbox"],
            "generated_points": geometry["points"],
        }
        try:
            aligned_result = _run_branch_from_collated_batch(
                model=model,
                collated_batch=collated_batch,
                idx=idx,
                branch_embedding=aligned_embedding,
                input_box=geometry["bbox"],
                input_points=geometry["points"],
                use_points=True,
            )
            aligned_record = _mask_branch_record(
                image_id=image_id,
                sample_id=sample_id,
                question=question,
                output_text=generation.output_text,
                think=generation.think,
                pred_bbox=aligned_result["pred_box"],
                pred_points=geometry["points"],
                pred_mask=aligned_result["pred_mask"],
                gt_mask=gt_mask,
                branch_type="aligned",
                extra=common_extra,
            )
        except Exception as exc:
            aligned_record = _empty_mask_branch_record(
                image_id=image_id,
                sample_id=sample_id,
                question=question,
                output_text=generation.output_text,
                think=generation.think,
                gt_mask=gt_mask,
                branch_type="aligned",
                extra=common_extra,
                error=str(exc),
            )

        try:
            direct_prediction = stage1_output["direct_predictions"][idx]
            direct_record = _mask_branch_record(
                image_id=image_id,
                sample_id=sample_id,
                question=question,
                output_text=generation.output_text,
                think=generation.think,
                pred_bbox=direct_prediction["bbox"],
                pred_points=[[0, 0], [0, 0]],
                pred_mask=direct_prediction["pred_mask"],
                gt_mask=gt_mask,
                branch_type="direct",
                extra=common_extra,
            )
        except Exception as exc:
            direct_record = _empty_mask_branch_record(
                image_id=image_id,
                sample_id=sample_id,
                question=question,
                output_text=generation.output_text,
                think=generation.think,
                gt_mask=gt_mask,
                branch_type="direct",
                extra=common_extra,
                error=str(exc),
            )

        aligned_metric = float(aligned_record["iou"] if fallback_metric == "mask_iou" else aligned_record["bbox_iou"])
        selection = _select_final_branch(aligned_metric, threshold=threshold, fallback_metric=fallback_metric)
        chosen_record = direct_record if selection["selected_branch"] == "direct" else aligned_record
        final_record = _attach_branch_breakdown(
            chosen_record,
            generated_bbox=geometry["bbox"],
            generated_points=geometry["points"],
            threshold=threshold,
            fallback_metric=fallback_metric,
            selected_branch=selection["selected_branch"],
            fallback_reason=selection["fallback_reason"],
            aligned_record=aligned_record,
            direct_record=direct_record,
        )
        results.append({"final_record": final_record, "aligned_record": aligned_record, "direct_record": direct_record})
    return results


def run_oracle_fallback_bbox_inference(
    model: Any,
    image: Image.Image,
    question: str,
    image_id: str,
    sample_id: str,
    gt_bbox: List[int],
    resize_size: int,
    max_new_tokens: int,
    threshold: float = 0.5,
    sam_image_size: int = 1024,
) -> Dict[str, Dict[str, Any]]:
    sample = RefineSample(
        image_path="",
        image_id=image_id,
        sample_id=sample_id,
        question_text=question,
        refexp=question,
        gt_mask=np.zeros((image.height, image.width), dtype=np.uint8),
        gt_box_xyxy=list(gt_bbox),
        width=image.width,
        height=image.height,
        raw_item={"image_obj": image},
    )
    collated_batch = _collate_samples_for_oracle(
        model=model,
        samples=[sample],
        resize_size=resize_size,
        sam_image_size=sam_image_size,
        max_pixels=2007040,
        min_pixels=3136,
    )
    model_inputs = collated_batch["model_inputs"]
    generation = model._generate_once(model_inputs, max_new_tokens=max_new_tokens, do_sample=False)[0][0]
    geometry = _sanitize_bbox_points(generation.answer, width=image.width, height=image.height, resize_size=resize_size)
    with torch.no_grad():
        stage1_output = model.stage1_forward(batch=collated_batch)

    cached_hidden, cached_mask = model.extract_cached_query_hidden_for_inputs(model_inputs)
    aligned_embedding, _ = model.build_connector_embeddings(cached_hidden, cached_mask)
    aligned_embedding = aligned_embedding[0]

    common_extra = {
        "generated_bbox": geometry["bbox"],
        "generated_points": geometry["points"],
        "threshold": float(threshold),
        "fallback_metric": "bbox_iou",
    }
    try:
        aligned_result = _run_branch_from_collated_batch(
            model=model,
            collated_batch=collated_batch,
            idx=0,
            branch_embedding=aligned_embedding,
            input_box=geometry["bbox"],
            input_points=geometry["points"],
            use_points=True,
        )
        aligned_record = _bbox_branch_record(
            image_id=image_id,
            sample_id=sample_id,
            question=question,
            output_text=generation.output_text,
            think=generation.think,
            pred_bbox=aligned_result["pred_box"],
            pred_points=geometry["points"],
            gt_bbox=gt_bbox,
            branch_type="aligned",
            extra=common_extra,
        )
    except Exception as exc:
        aligned_record = _empty_bbox_branch_record(
            image_id=image_id,
            sample_id=sample_id,
            question=question,
            output_text=generation.output_text,
            think=generation.think,
            gt_bbox=gt_bbox,
            branch_type="aligned",
            extra=common_extra,
            error=str(exc),
        )

    try:
        direct_prediction = stage1_output["direct_predictions"][0]
        direct_record = _bbox_branch_record(
            image_id=image_id,
            sample_id=sample_id,
            question=question,
            output_text=generation.output_text,
            think=generation.think,
            pred_bbox=direct_prediction["bbox"],
            pred_points=[[0, 0], [0, 0]],
            gt_bbox=gt_bbox,
            branch_type="direct",
            extra=common_extra,
        )
    except Exception as exc:
        direct_record = _empty_bbox_branch_record(
            image_id=image_id,
            sample_id=sample_id,
            question=question,
            output_text=generation.output_text,
            think=generation.think,
            gt_bbox=gt_bbox,
            branch_type="direct",
            extra=common_extra,
            error=str(exc),
        )

    selection = _select_final_branch(float(aligned_record["bbox_iou"]), threshold=threshold, fallback_metric="bbox_iou")
    chosen_record = direct_record if selection["selected_branch"] == "direct" else aligned_record
    final_record = _attach_branch_breakdown(
        chosen_record,
        generated_bbox=geometry["bbox"],
        generated_points=geometry["points"],
        threshold=threshold,
        fallback_metric="bbox_iou",
        selected_branch=selection["selected_branch"],
        fallback_reason=selection["fallback_reason"],
        aligned_record=aligned_record,
        direct_record=direct_record,
    )
    return {"final_record": final_record, "aligned_record": aligned_record, "direct_record": direct_record}


def run_oracle_fallback_bbox_inference_batch(
    model: Any,
    samples: Sequence[RefineSample],
    resize_size: int,
    max_new_tokens: int,
    threshold: float = 0.5,
    sam_image_size: int = 1024,
    max_pixels: int = 2007040,
    min_pixels: int = 3136,
) -> List[Dict[str, Dict[str, Any]]]:
    collated_batch = _collate_samples_for_oracle(
        model=model,
        samples=samples,
        resize_size=resize_size,
        sam_image_size=sam_image_size,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )
    model_inputs = collated_batch["model_inputs"]
    generations = model._generate_once(model_inputs, max_new_tokens=max_new_tokens, do_sample=False)[0]
    with torch.no_grad():
        stage1_output = model.stage1_forward(batch=collated_batch)
    cached_hidden, cached_mask = model.extract_cached_query_hidden_for_inputs(model_inputs)
    aligned_embeddings, _ = model.build_connector_embeddings(cached_hidden, cached_mask)

    results: List[Dict[str, Dict[str, Any]]] = []
    for idx, (sample, generation) in enumerate(
        zip(samples, generations)
    ):
        image = _resolve_sample_image(sample)
        question = sample.refexp
        image_id = sample.image_id
        sample_id = sample.sample_id
        gt_bbox = sample.gt_box_xyxy
        geometry = _sanitize_bbox_points(generation.answer, width=image.width, height=image.height, resize_size=resize_size)
        aligned_embedding = aligned_embeddings[idx]
        common_extra = {
            "generated_bbox": geometry["bbox"],
            "generated_points": geometry["points"],
            "threshold": float(threshold),
            "fallback_metric": "bbox_iou",
        }
        try:
            aligned_result = _run_branch_from_collated_batch(
                model=model,
                collated_batch=collated_batch,
                idx=idx,
                branch_embedding=aligned_embedding,
                input_box=geometry["bbox"],
                input_points=geometry["points"],
                use_points=True,
            )
            aligned_record = _bbox_branch_record(
                image_id=image_id,
                sample_id=sample_id,
                question=question,
                output_text=generation.output_text,
                think=generation.think,
                pred_bbox=aligned_result["pred_box"],
                pred_points=geometry["points"],
                gt_bbox=gt_bbox,
                branch_type="aligned",
                extra=common_extra,
            )
        except Exception as exc:
            aligned_record = _empty_bbox_branch_record(
                image_id=image_id,
                sample_id=sample_id,
                question=question,
                output_text=generation.output_text,
                think=generation.think,
                gt_bbox=gt_bbox,
                branch_type="aligned",
                extra=common_extra,
                error=str(exc),
            )

        try:
            direct_prediction = stage1_output["direct_predictions"][idx]
            direct_record = _bbox_branch_record(
                image_id=image_id,
                sample_id=sample_id,
                question=question,
                output_text=generation.output_text,
                think=generation.think,
                pred_bbox=direct_prediction["bbox"],
                pred_points=[[0, 0], [0, 0]],
                gt_bbox=gt_bbox,
                branch_type="direct",
                extra=common_extra,
            )
        except Exception as exc:
            direct_record = _empty_bbox_branch_record(
                image_id=image_id,
                sample_id=sample_id,
                question=question,
                output_text=generation.output_text,
                think=generation.think,
                gt_bbox=gt_bbox,
                branch_type="direct",
                extra=common_extra,
                error=str(exc),
            )

        selection = _select_final_branch(float(aligned_record["bbox_iou"]), threshold=threshold, fallback_metric="bbox_iou")
        chosen_record = direct_record if selection["selected_branch"] == "direct" else aligned_record
        final_record = _attach_branch_breakdown(
            chosen_record,
            generated_bbox=geometry["bbox"],
            generated_points=geometry["points"],
            threshold=threshold,
            fallback_metric="bbox_iou",
            selected_branch=selection["selected_branch"],
            fallback_reason=selection["fallback_reason"],
            aligned_record=aligned_record,
            direct_record=direct_record,
        )
        results.append({"final_record": final_record, "aligned_record": aligned_record, "direct_record": direct_record})
    return results


def load_model_from_export(export_dir: str, device: torch.device):
    return load_geometric_model_from_export(Path(export_dir), device=device)