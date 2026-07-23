import json
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from training_scripts.eval.common import save_prediction_part
from training_scripts.eval.query_reflect_dataset import QueryReflectDataset
from training_scripts.refine_segzero.common import (
    RefineSample,
    build_mask_crop_image,
    compute_iou,
    compute_box_iou,
    mask_to_box,
    resize_longest_side,
)


from training_scripts.refine_segzero.geometric_query_export import load_geometric_model_from_export
from training_scripts.refine_segzero.geometric_query_infer import (
    _bbox_branch_record,
    _collate_samples_for_oracle,
    _empty_mask_branch_record,
    _mask_branch_record,
    _run_branch_from_collated_batch,
    _sanitize_bbox_points,
)
from training_scripts.refine_segzero.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    GEOMETRIC_QUERY_TEMPLATE,
    build_query_reflect_reflect_rl_prompt,
)

def _strip_reflect_output_to_reason_only(text: str) -> str:
    text = str(text or "").strip()
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if not think_match:
        return re.sub(r"<answer>.*?</answer>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    think = re.sub(r"\s*4\.\s*.*$", "", think_match.group(1).strip(), flags=re.DOTALL).strip()
    return f"<think>{think}</think>" if think else ""

def resolve_part_args(idx: int, num_parts: int) -> Tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    resolved_idx = idx if idx >= 0 else rank
    resolved_parts = num_parts if num_parts > 0 else world_size
    return resolved_idx, resolved_parts


def split_indices(total_len: int, idx: int, num_parts: int) -> range:
    start = (total_len * idx) // num_parts
    end = (total_len * (idx + 1)) // num_parts
    return range(start, end)


def _store_reflect_processor_pixel_limits(model: Any) -> None:
    image_processor = getattr(getattr(model, "processor", None), "image_processor", None)
    if image_processor is None:
        return
    model._reflect_processor_max_pixels = getattr(image_processor, "max_pixels", None)
    model._reflect_processor_min_pixels = getattr(image_processor, "min_pixels", None)


def _restore_reflect_processor_pixel_limits(model: Any) -> None:
    image_processor = getattr(getattr(model, "processor", None), "image_processor", None)
    if image_processor is None:
        return
    if hasattr(model, "_reflect_processor_max_pixels"):
        image_processor.max_pixels = model._reflect_processor_max_pixels
    if hasattr(model, "_reflect_processor_min_pixels"):
        image_processor.min_pixels = model._reflect_processor_min_pixels


# 2026-04-25: 几何分支始终只使用同一个 geometry_model；
# reflect_model 只是 reflect 阶段是否替换 MLLM 的文本生成视图。
def _load_branch_models(
    geometric_export_dir: str,
    shared_mllm_path: Optional[str],
    device: torch.device,
):
    geometry_model = load_geometric_model_from_export(Path(geometric_export_dir), device=device)
    geometry_model.eval()
    _store_reflect_processor_pixel_limits(geometry_model)

    override_path = (shared_mllm_path or "").strip()
    if not override_path:
        reflect_model = geometry_model
    else:
        reflect_model = load_plain_reflect_mllm(override_path, device)
    return geometry_model, reflect_model


def load_plain_reflect_mllm(mllm_path: str, device: torch.device) -> Any:
    target_path = Path((mllm_path or "").strip())
    if not target_path.exists():
        raise FileNotFoundError(f"Reflect MLLM path does not exist: {target_path}")

    processor = AutoProcessor.from_pretrained(
        str(target_path),
        padding_side="left",
        trust_remote_code=True,
    )
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(target_path),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="flash_attention_2" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    qwen.requires_grad_(False)
    qwen.to(device)
    qwen.eval()

    class PlainReflectModel:
        pass

    model = PlainReflectModel()
    model.qwen = qwen
    model.processor = processor
    _store_reflect_processor_pixel_limits(model)
    return model


# 2026-04-25: 这里只替换 qwen + processor，不改 connector / SAM /
# segmentation 路径，因此 reflect_model 仍共享同样的几何分割能力。
def maybe_override_mllm_path(model: Any, mllm_path: Optional[str], device: torch.device) -> Any:
    override_path = (mllm_path or "").strip()
    if not override_path:
        return model

    target_path = Path(override_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Override MLLM path does not exist: {target_path}")

    old_qwen = model.qwen
    old_processor = model.processor
    dtype = next(old_qwen.parameters()).dtype
    attn_implementation = (
        getattr(old_qwen.config, "_attn_implementation", None)
        or getattr(old_qwen.config, "attn_implementation", None)
        or "flash_attention_2"
    )

    model.processor = AutoProcessor.from_pretrained(
        str(target_path),
        padding_side="left",
        trust_remote_code=True,
    )
    model.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(target_path),
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.qwen.requires_grad_(False)
    model.qwen.to(device)
    model.qwen.eval()
    _store_reflect_processor_pixel_limits(model)

    del old_qwen
    del old_processor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model


def _generate_text_from_messages(
    qwen: Any,
    processor: Any,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([messages])
    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {
        key: value.to(qwen.device) if torch.is_tensor(value) else value
        for key, value in model_inputs.items()
    }
    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        raise KeyError(
            f"Processor outputs do not contain 'input_ids'. Available keys: {sorted(model_inputs.keys())}"
        )
    with torch.inference_mode():
        generated = qwen.generate(
            **model_inputs,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


_CONCLUSION_DECISION_RE = re.compile(
    r"\bconclusion\s*:\s*(accept|reject)\b",
    flags=re.IGNORECASE,
)


def _teacher_forced_sequence_logprob(
    qwen: Any,
    model_inputs: Dict[str, torch.Tensor],
    generated_prefix_ids: Sequence[int],
    candidate_ids: Sequence[int],
) -> float:
    if not candidate_ids:
        raise ValueError("Cannot score an empty decision candidate.")

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    generated_prefix = torch.as_tensor(
        list(generated_prefix_ids),
        dtype=input_ids.dtype,
        device=input_ids.device,
    ).unsqueeze(0)
    candidate = torch.as_tensor(
        list(candidate_ids),
        dtype=input_ids.dtype,
        device=input_ids.device,
    ).unsqueeze(0)
    prefix_ids = torch.cat([input_ids, generated_prefix], dim=1)
    full_input_ids = torch.cat([prefix_ids, candidate], dim=1)
    suffix_mask = torch.ones(
        (attention_mask.shape[0], generated_prefix.shape[1] + candidate.shape[1]),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    full_attention_mask = torch.cat([attention_mask, suffix_mask], dim=1)

    forward_inputs = {
        key: value
        for key, value in model_inputs.items()
        if key not in {"input_ids", "attention_mask", "position_ids", "cache_position"}
    }
    with torch.inference_mode():
        outputs = qwen(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            use_cache=False,
            **forward_inputs,
        )
    prefix_length = int(prefix_ids.shape[1])
    candidate_length = int(candidate.shape[1])
    candidate_logits = outputs.logits[
        0,
        prefix_length - 1 : prefix_length + candidate_length - 1,
        :,
    ]
    candidate_logprobs = torch.log_softmax(candidate_logits.float(), dim=-1)
    token_logprobs = candidate_logprobs.gather(
        dim=-1,
        index=candidate[0].long().unsqueeze(-1),
    ).squeeze(-1)
    return float(token_logprobs.sum().item())


def _locate_conclusion_decision_token(
    processor: Any,
    generated_ids: Sequence[int],
    output_text: str,
) -> Tuple[int, str, str]:
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", output_text, flags=re.DOTALL | re.IGNORECASE)
    scope_text = think_match.group(1) if think_match else output_text
    scope_offset = think_match.start(1) if think_match else 0
    matches = list(_CONCLUSION_DECISION_RE.finditer(scope_text))
    if not matches:
        raise ValueError("No 'Conclusion: accept/reject' decision was found in the generated reasoning.")
    decision_match = matches[-1]
    decision = decision_match.group(1).lower()
    decision_char_start = scope_offset + decision_match.start(1)

    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        encoded = tokenizer(
            output_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        encoded_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        comparable_generated_ids = list(generated_ids[: len(encoded_ids)])
        if comparable_generated_ids == encoded_ids:
            for token_index, (char_start, char_end) in enumerate(offsets):
                if int(char_start) <= decision_char_start < int(char_end):
                    leading_text = output_text[int(char_start) : decision_char_start]
                    return token_index, leading_text, decision
    except Exception:
        pass

    token_index = 0
    decoded_prefix = ""
    for token_count in range(len(generated_ids) + 1):
        candidate_prefix = processor.batch_decode(
            [list(generated_ids[:token_count])],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if len(candidate_prefix) > decision_char_start:
            break
        token_index = token_count
        decoded_prefix = candidate_prefix
    if not output_text.startswith(decoded_prefix):
        raise ValueError("Unable to align the generated decision text with its token IDs.")
    leading_text = output_text[len(decoded_prefix) : decision_char_start]
    return token_index, leading_text, decision


def _score_conclusion_decision(
    qwen: Any,
    processor: Any,
    model_inputs: Dict[str, torch.Tensor],
    generated_ids: Sequence[int],
    generation_scores: Sequence[torch.Tensor],
    output_text: str,
) -> Dict[str, Any]:
    try:
        token_index, leading_text, generated_decision = _locate_conclusion_decision_token(
            processor=processor,
            generated_ids=generated_ids,
            output_text=output_text,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        resolved_leading_text = None
        leading_text_candidates = [leading_text, " ", "", "\n", "\n ", "\t"]
        for candidate_leading_text in dict.fromkeys(leading_text_candidates):
            actual_decision_ids = list(
                tokenizer(
                    f"{candidate_leading_text}{generated_decision}",
                    add_special_tokens=False,
                )["input_ids"]
            )
            generated_slice = list(
                generated_ids[token_index : token_index + len(actual_decision_ids)]
            )
            if actual_decision_ids and generated_slice == actual_decision_ids:
                resolved_leading_text = candidate_leading_text
                break
        if resolved_leading_text is None:
            raise ValueError(
                "Unable to reproduce the generated conclusion decision with the tokenizer."
            )

        accept_ids = list(
            tokenizer(
                f"{resolved_leading_text}accept",
                add_special_tokens=False,
            )["input_ids"]
        )
        reject_ids = list(
            tokenizer(
                f"{resolved_leading_text}reject",
                add_special_tokens=False,
            )["input_ids"]
        )
        if not accept_ids or not reject_ids:
            raise ValueError("The tokenizer produced an empty accept/reject candidate.")

        actual_token_id = int(generated_ids[token_index]) if token_index < len(generated_ids) else -1
        use_generation_logits = (
            len(accept_ids) == 1
            and len(reject_ids) == 1
            and token_index < len(generation_scores)
            and actual_token_id
            == (accept_ids[0] if generated_decision == "accept" else reject_ids[0])
        )
        if use_generation_logits:
            step_logits = generation_scores[token_index][0].float()
            full_logprobs = torch.log_softmax(step_logits, dim=-1)
            accept_logprob = float(full_logprobs[int(accept_ids[0])].item())
            reject_logprob = float(full_logprobs[int(reject_ids[0])].item())
            scoring_method = "generation_logits"
        else:
            generated_prefix_ids = list(generated_ids[:token_index])
            accept_logprob = _teacher_forced_sequence_logprob(
                qwen=qwen,
                model_inputs=model_inputs,
                generated_prefix_ids=generated_prefix_ids,
                candidate_ids=accept_ids,
            )
            reject_logprob = _teacher_forced_sequence_logprob(
                qwen=qwen,
                model_inputs=model_inputs,
                generated_prefix_ids=generated_prefix_ids,
                candidate_ids=reject_ids,
            )
            scoring_method = "teacher_forced_sequence_logprob"

        margin = accept_logprob - reject_logprob
        accept_probability = float(torch.sigmoid(torch.tensor(margin, dtype=torch.float64)).item())
        return {
            "valid": True,
            "generated_conclusion_decision": generated_decision,
            "decision_token_index": int(token_index),
            "accept_token_ids": [int(v) for v in accept_ids],
            "reject_token_ids": [int(v) for v in reject_ids],
            "accept_logprob": accept_logprob,
            "reject_logprob": reject_logprob,
            "logit_margin": margin,
            "accept_probability": accept_probability,
            "scoring_method": scoring_method,
            "error": "",
        }
    except Exception as exc:
        return {
            "valid": False,
            "generated_conclusion_decision": "",
            "decision_token_index": -1,
            "accept_token_ids": [],
            "reject_token_ids": [],
            "accept_logprob": 0.0,
            "reject_logprob": 0.0,
            "logit_margin": 0.0,
            "accept_probability": 0.0,
            "scoring_method": "",
            "error": str(exc),
        }


def _generate_reflect_output_with_decision_scores(
    reflect_model: Any,
    reflect_image: Image.Image,
    prompt_text: str,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    _restore_reflect_processor_pixel_limits(reflect_model)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": reflect_image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    qwen = reflect_model.qwen
    processor = reflect_model.processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([messages])
    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {
        key: value.to(qwen.device) if torch.is_tensor(value) else value
        for key, value in model_inputs.items()
    }
    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        raise KeyError(
            f"Processor outputs do not contain 'input_ids'. Available keys: {sorted(model_inputs.keys())}"
        )
    with torch.inference_mode():
        generated = qwen.generate(
            **model_inputs,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    input_length = int(input_ids.shape[1])
    trimmed_ids = generated.sequences[0, input_length:]
    output_text = processor.batch_decode(
        [trimmed_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    score_info = _score_conclusion_decision(
        qwen=qwen,
        processor=processor,
        model_inputs=model_inputs,
        generated_ids=[int(v) for v in trimmed_ids.tolist()],
        generation_scores=generated.scores,
        output_text=output_text,
    )
    return output_text, score_info


def _generate_reflect_output(
    reflect_model: Any,
    reflect_image: Image.Image,
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    _restore_reflect_processor_pixel_limits(reflect_model)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": reflect_image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    return _generate_text_from_messages(
        qwen=reflect_model.qwen,
        processor=reflect_model.processor,
        messages=messages,
        max_new_tokens=max_new_tokens,
    )

def _build_direct_reflect_context_model_inputs(
    model: Any,
    image: Image.Image,
    question: str,
    stage1_output_text: str,
    reflect_output_text: str,
    resize_size: int,
    max_pixels: int,
    min_pixels: int,
) -> Dict[str, torch.Tensor]:
    model.processor.image_processor.max_pixels = max_pixels
    model.processor.image_processor.min_pixels = min_pixels

    conversation = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": resize_longest_side(image, longest_side=resize_size)},
                {
                    "type": "text",
                    "text": GEOMETRIC_QUERY_TEMPLATE.format(
                        Question=question.lower().strip(".")
                    ),
                },
            ],
        },
    ]

    if str(stage1_output_text).strip():
        conversation.append({"role": "assistant", "content": str(stage1_output_text)})

    conversation.append(
        {
            "role": "user",
            "content": (
                "The previous proposal was evaluated by a reflection model and should be handled by the direct branch.\n"
                f"Reflection output:\n{reflect_output_text}\n"
                "Use the original image and the full context above to segment the referred object directly."
            ),
        }
    )

    text = model.processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info([conversation])

    return model.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

def _build_direct_query_reflect_model_inputs(
    model: Any,
    reflect_image: Image.Image,
    reflect_prompt: str,
    reflect_output_text: str,
    resize_size: int,
    max_pixels: int,
    min_pixels: int,
    prompt_mode: str,
) -> Tuple[Dict[str, torch.Tensor], str]:
    model.processor.image_processor.max_pixels = max_pixels
    model.processor.image_processor.min_pixels = min_pixels
    output_text = str(reflect_output_text or "").strip()
    if prompt_mode == "query_reflect_reason_only":
        output_text = _strip_reflect_output_to_reason_only(output_text)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": resize_longest_side(reflect_image, longest_side=resize_size)},
                {"type": "text", "text": str(reflect_prompt or "").strip()},
            ],
        },
    ]
    if output_text:
        conversation.append({"role": "assistant", "content": output_text})
    text = model.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([conversation])
    model_inputs = model.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return model_inputs, text


def _build_direct_branch_embedding(
    model: Any,
    base_model_inputs: Dict[str, torch.Tensor],
    image: Image.Image,
    question: str,
    reflect_image: Image.Image,
    reflect_prompt: str,
    stage1_output_text: str,
    reflect_output_text: str,
    resize_size: int,
    max_pixels: int,
    min_pixels: int,
    direct_branch_prompt_mode: str,
) -> Tuple[torch.Tensor, str, str]:
    prompt_mode = str(direct_branch_prompt_mode or "geometric").strip().lower()
    direct_prompt_text = ""
    if prompt_mode == "reflect_context":
        prompt_mode = "query_reflect"
    if prompt_mode in {"query_reflect", "query_reflect_reason_only"}:
        direct_inputs, direct_prompt_text = _build_direct_query_reflect_model_inputs(
            model=model,
            reflect_image=reflect_image,
            reflect_prompt=reflect_prompt,
            reflect_output_text=reflect_output_text,
            resize_size=resize_size,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            prompt_mode=prompt_mode,
        )
    elif prompt_mode == "geometric":
        direct_inputs = base_model_inputs
        direct_prompt_text = model.processor.tokenizer.batch_decode(
            direct_inputs["input_ids"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
    else:
        raise ValueError(f"Unsupported direct_branch_prompt_mode: {direct_branch_prompt_mode}")

    with model.direct_lora_enabled():
        query_hidden_batch, query_hidden_mask = model.extract_cached_query_hidden_for_inputs(
            direct_inputs,
            branch="direct",
        )
        _, direct_embedding_batch = model.build_connector_embeddings(
            query_hidden_batch,
            query_hidden_mask,
            branch="direct",
        )

    return direct_embedding_batch[0], prompt_mode, direct_prompt_text



def _safe_float(value: Any) -> Tuple[float, bool]:
    if value is None:
        return 0.0, False
    try:
        return float(value), True
    except Exception:
        return 0.0, False


def parse_reflect_output(output_text: str) -> Dict[str, Any]:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL)
    if not match:
        return {"valid": False, "decision": "reject", "confidence": 0.0, "bbox": None, "points": None}
    try:
        payload = json.loads(match.group(1).strip())
    except Exception:
        return {"valid": False, "decision": "reject", "confidence": 0.0, "bbox": None, "points": None}

    raw_decision = payload.get("decision", "")
    if isinstance(raw_decision, str):
        decision = raw_decision.lower().strip()
    elif raw_decision in {0, 1}:
        decision = "accept" if int(raw_decision) == 1 else "reject"
    else:
        decision = ""

    has_extra_keys = any(key not in {"decision"} for key in payload.keys()) if isinstance(payload, dict) else True
    valid = decision in {"accept", "reject"} and not has_extra_keys

    return {
        "valid": valid,
        "decision": decision or "reject",
        "confidence": 1.0 if decision == "accept" else 0.0,
        "bbox": None,
        "points": None,
    }

def _think_implies_reject(output_text: str) -> bool:
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", output_text, re.DOTALL)
    if not think_match:
        return False
    think_text = think_match.group(1).lower()
    return "does not match" in think_text


def _select_query_reflect_branch(
    parsed: Dict[str, Any],
    decision_mode: str,
    confidence_threshold: float,
) -> str:
    if not parsed.get("valid", False):
        return "parse_error_direct"

    decision = str(parsed.get("decision", "")).lower().strip()
    return "accept_aligned" if decision == "accept" else "reject_direct"


def _apply_sam_mask_threshold_postprocess(
    prediction: Dict[str, Any],
    *,
    enabled: bool,
    probability_threshold: float,
) -> Dict[str, Any]:
    """Optionally re-binarize one SAM2 prediction without changing model inference."""
    if not enabled:
        return prediction

    threshold = float(probability_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"SAM mask probability threshold must be in [0, 1], got {threshold}")

    pred_mask_logits = prediction.get("pred_mask_logits")
    if not torch.is_tensor(pred_mask_logits):
        raise ValueError("SAM mask threshold postprocess requires tensor pred_mask_logits")

    pred_mask = (
        torch.sigmoid(pred_mask_logits.detach().float()).cpu().numpy() > threshold
    ).astype(np.uint8)
    pred_mask = np.squeeze(pred_mask)
    if pred_mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D SAM mask after thresholding, got shape {tuple(pred_mask.shape)}"
        )

    updated = dict(prediction)
    original_mask = np.asarray(prediction.get("pred_mask", np.zeros_like(pred_mask)))
    updated["pred_mask"] = pred_mask
    updated["pred_box"] = mask_to_box(pred_mask)
    updated["sam_mask_threshold_metadata"] = {
        "sam_mask_threshold_postprocess_enabled": True,
        "sam_mask_probability_threshold": threshold,
        "sam_mask_positive_pixels_before": int(np.count_nonzero(original_mask)),
        "sam_mask_positive_pixels_after": int(np.count_nonzero(pred_mask)),
        "sam_mask_empty_after_threshold": bool(not np.any(pred_mask)),
    }
    return updated


def _center_point_from_mask(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0]
    return [int(round(float(xs.mean()))), int(round(float(ys.mean())))]


def _draw_point(draw: ImageDraw.ImageDraw, point: Sequence[int], color: Tuple[int, int, int], radius: int = 5) -> None:
    if len(point) < 2:
        return
    x, y = int(point[0]), int(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color)

def _safe_stem(text: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z._-]+", "_", str(text))
    safe = safe.strip("_")
    return safe or "sample"

def _render_stage1_proposal_visualization(
    image: Image.Image,
    pred_bbox: Sequence[int],
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = [int(v) for v in pred_bbox[:4]]
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
    return canvas


def _save_stage1_proposal_visualization(
    reflect_image: Image.Image,
    output_dir: str,
    part_idx: int,
    image_id: str,
    sample_id: str,
    question: str,
) -> str:
    vis_dir = Path(output_dir) / "stage1_proposal_visualizations" / f"part_{part_idx}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    image_label = _safe_stem(Path(image_id).stem)
    ann_label = _safe_stem(str(sample_id))
    question_label = _safe_stem(question)[:80]
    output_path = vis_dir / f"{image_label}__{ann_label}__{question_label}.png"
    reflect_image.save(output_path)
    return str(output_path)


def _save_reject_direct_prompt(
    prompt_text: str,
    output_dir: str,
    part_idx: int,
    image_id: str,
    sample_id: str,
    question: str,
) -> str:
    prompt_dir = Path(output_dir) / "reject_direct_prompts" / f"part_{part_idx}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    image_label = _safe_stem(Path(image_id).stem)
    ann_label = _safe_stem(str(sample_id))
    question_label = _safe_stem(question)[:80]
    output_path = prompt_dir / f"{image_label}__{ann_label}__{question_label}.txt"
    output_path.write_text(str(prompt_text or ""), encoding="utf-8")
    return str(output_path)


def _render_branch_visual_panel(
    image: Image.Image,
    pred_mask: np.ndarray,
    pred_bbox: Sequence[int],
    pred_points: Sequence[Sequence[int]],
    gt_mask: np.ndarray,
    gt_bbox: Sequence[int],
    title_text: str,
    show_gt_mask_point: bool = True,
) -> Image.Image:
    canvas = image.convert("RGBA")
    mask_overlay = np.zeros((canvas.height, canvas.width, 4), dtype=np.uint8)
    mask_overlay[pred_mask > 0] = [255, 255, 0, 64]
    overlay = Image.fromarray(mask_overlay, mode="RGBA")
    canvas = Image.alpha_composite(canvas, overlay)

    draw = ImageDraw.Draw(canvas)
    gt_box = [int(v) for v in gt_bbox]
    pd_box = [int(v) for v in pred_bbox]
    draw.rectangle(gt_box, outline=(255, 0, 0, 255), width=3)
    draw.rectangle(pd_box, outline=(0, 255, 0, 255), width=3)

    if show_gt_mask_point:
        gt_point = _center_point_from_mask(gt_mask)
        _draw_point(draw, gt_point, (255, 0, 0))
    for point in pred_points:
        _draw_point(draw, point, (0, 255, 0))

    panel = canvas.convert("RGB")
    draw = ImageDraw.Draw(panel)
    draw.rectangle([8, 8, min(panel.width - 8, 520), 52], fill=(0, 0, 0))
    draw.text((16, 16), title_text, fill=(255, 255, 255))
    return panel

def _save_reject_branch_side_by_side_visualization(
    image: Image.Image,
    aligned_pred_mask: np.ndarray,
    aligned_pred_bbox: Sequence[int],
    aligned_pred_points: Sequence[Sequence[int]],
    aligned_iou: float,
    aligned_bbox_iou: float,
    direct_pred_mask: np.ndarray,
    direct_pred_bbox: Sequence[int],
    direct_pred_points: Sequence[Sequence[int]],
    direct_iou: float,
    direct_bbox_iou: float,
    gt_mask: np.ndarray,
    gt_bbox: Sequence[int],
    output_dir: str,
    part_idx: int,
    sample_id: str,
    parsed_decision: str,
    parsed_confidence: float,
    direct_variant: str,
    image_id: str,
    question: str,
    bbox_only_metrics: bool = False,
) -> None:
    vis_dir = Path(output_dir) / "reject_branch_visualizations" / f"part_{part_idx}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if bbox_only_metrics:
        aligned_title = f"aligned candidate bbox_iou={aligned_bbox_iou:.4f}"
        direct_title = (
            f"reject direct bbox_iou={direct_bbox_iou:.4f} "
            f"decision={parsed_decision} conf={parsed_confidence:.4f} {direct_variant}"
        )
    else:
        aligned_title = f"aligned candidate iou={aligned_iou:.4f} bbox_iou={aligned_bbox_iou:.4f}"
        direct_title = (
            f"reject direct iou={direct_iou:.4f} bbox_iou={direct_bbox_iou:.4f} "
            f"decision={parsed_decision} conf={parsed_confidence:.4f} {direct_variant}"
        )

    left_panel = _render_branch_visual_panel(
        image=image,
        pred_mask=aligned_pred_mask,
        pred_bbox=aligned_pred_bbox,
        pred_points=aligned_pred_points,
        gt_mask=gt_mask,
        gt_bbox=gt_bbox,
        title_text=aligned_title,
        show_gt_mask_point=not bbox_only_metrics,
    )
    right_panel = _render_branch_visual_panel(
        image=image,
        pred_mask=direct_pred_mask,
        pred_bbox=direct_pred_bbox,
        pred_points=[] if bbox_only_metrics else direct_pred_points,
        gt_mask=gt_mask,
        gt_bbox=gt_bbox,
        title_text=direct_title,
        show_gt_mask_point=not bbox_only_metrics,
    )

    w, h = left_panel.size
    canvas = Image.new("RGB", (w * 2, h), (255, 255, 255))
    canvas.paste(left_panel, (0, 0))
    canvas.paste(right_panel, (w, 0))
    image_label = _safe_stem(Path(image_id).stem)
    ann_label = _safe_stem(str(sample_id))
    question_label = _safe_stem(question)[:80]
    canvas.save(vis_dir / f"{image_label}__{ann_label}__{question_label}.png")
    print(f"####{question_label}###",image_label)



def _clone_model_inputs(model_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cloned: Dict[str, torch.Tensor] = {}
    for key, value in model_inputs.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def _image_token_positions_for_box(
    model: Any,
    model_inputs: Dict[str, torch.Tensor],
    box_xyxy: Sequence[int],
    image_size: Tuple[int, int],
) -> torch.Tensor:
    input_ids = model_inputs["input_ids"]
    image_grid_thw = model_inputs["image_grid_thw"]

    if input_ids.size(0) != 1:
        raise RuntimeError("Attention masking currently expects a single-sample batch.")
    if image_grid_thw.size(0) != 1:
        raise RuntimeError("Attention masking currently expects one image grid per sample.")

    image_positions = torch.nonzero(
        input_ids[0] == model.qwen.config.image_token_id,
        as_tuple=False,
    ).squeeze(-1)
    if image_positions.numel() == 0:
        raise RuntimeError("No image tokens found while building reject-direct attention mask.")

    merge_size = int(getattr(model.processor.image_processor, "merge_size", 1))
    t, h, w = [int(v) for v in image_grid_thw[0].tolist()]
    grid_h = h // merge_size
    grid_w = w // merge_size
    expected_tokens = t * grid_h * grid_w
    if int(image_positions.numel()) != expected_tokens:
        raise RuntimeError(
            f"Image token count mismatch: found {image_positions.numel()} expected {expected_tokens}"
        )

    image_w, image_h = image_size
    x1, y1, x2, y2 = [float(v) for v in box_xyxy[:4]]
    x1, x2 = sorted((max(0.0, min(x1, float(image_w))), max(0.0, min(x2, float(image_w)))))
    y1, y2 = sorted((max(0.0, min(y1, float(image_h))), max(0.0, min(y2, float(image_h)))))

    if image_w <= 0 or image_h <= 0 or x2 <= x1 or y2 <= y1:
        return image_positions.new_empty((0,))

    grid_x1 = max(0, min(grid_w - 1, int(np.floor(x1 / float(image_w) * grid_w))))
    grid_x2 = max(0, min(grid_w - 1, int(np.ceil(x2 / float(image_w) * grid_w) - 1)))
    grid_y1 = max(0, min(grid_h - 1, int(np.floor(y1 / float(image_h) * grid_h))))
    grid_y2 = max(0, min(grid_h - 1, int(np.ceil(y2 / float(image_h) * grid_h) - 1)))

    drop_grid = torch.zeros((t, grid_h, grid_w), device=image_positions.device, dtype=torch.bool)
    drop_grid[:, grid_y1 : grid_y2 + 1, grid_x1 : grid_x2 + 1] = True
    return image_positions[drop_grid.reshape(-1)]


def _drop_box_image_token_attention(
    model: Any,
    model_inputs: Dict[str, torch.Tensor],
    box_xyxy: Sequence[int],
    image_size: Tuple[int, int],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    masked_inputs = _clone_model_inputs(model_inputs)
    drop_positions = _image_token_positions_for_box(model, masked_inputs, box_xyxy, image_size)
    masked_inputs["attention_mask"][0, drop_positions] = 0
    debug_info = {
        "used_reject_direct_attention_mask": bool(drop_positions.numel() > 0),
        "reject_direct_attention_mask_source": "stage1_bbox",
        "reject_direct_attention_mask_bbox": [int(v) for v in box_xyxy[:4]],
        "reject_direct_attention_mask_token_count": int(drop_positions.numel()),
    }
    return masked_inputs, debug_info


def _run_masked_direct_prediction(
    model: Any,
    collated_batch: Dict[str, Any],
    image: Image.Image,
    stage1_bbox: Sequence[int],
    gt_mask: np.ndarray,
    gt_box: Sequence[int],
    sam_image_size: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_inputs = collated_batch["model_inputs"]
    drop_positions = _image_token_positions_for_box(
        model=model,
        model_inputs=model_inputs,
        box_xyxy=stage1_bbox,
        image_size=(int(image.width), int(image.height)),
    )
    debug_info = {
        "used_reject_direct_attention_mask": bool(drop_positions.numel() > 0),
        "reject_direct_attention_mask_source": "stage1_bbox",
        "reject_direct_attention_mask_bbox": [int(v) for v in stage1_bbox[:4]],
        "reject_direct_attention_mask_token_count": int(drop_positions.numel()),
    }

    model._late_attention_drop = {
        "drop_positions": drop_positions,
        "start_layer": 5,
    }
    try:
        with model.direct_lora_enabled():
            query_hidden_batch, query_hidden_mask = model.extract_cached_query_hidden_for_inputs(
                model_inputs,
                branch="direct",
            )
            _, direct_embedding_batch = model.build_connector_embeddings(
                query_hidden_batch,
                query_hidden_mask,
                branch="direct",
            )
    finally:
        model._late_attention_drop = None
    gt_mask_tensor = torch.as_tensor(gt_mask, dtype=torch.float32)
    result = model.segment_with_direct_embedding(
        image=image,
        direct_embedding=direct_embedding_batch[0],
        gt_box=gt_box,
        gt_mask=gt_mask_tensor,
        sam_image_size=sam_image_size,
    )
    return result, debug_info

def run_single_prediction(
    geometry_model: Any,
    reflect_model: Any,
    sample: RefineSample,
    resize_size: int,
    sam_image_size: int,
    max_pixels: int,
    min_pixels: int,
    stage1_max_new_tokens: int,
    reflect_max_new_tokens: int,
    confidence_threshold: float,
    decision_mode: str,
    output_dir: str,
    part_idx: int,
    use_reject_direct_attention_mask: bool = False,
    save_reject_branch_visualizations: bool = False,
    reject_branch_visualization_limit: int = -1,
    save_stage1_proposal_visualizations: bool = False,
    save_reject_direct_prompts: bool = False,
    use_direct_query_for_direct_branch: bool = False,
    direct_branch_prompt_mode: str = "geometric",
    bbox_only_metrics: bool = False,
    enable_sam_mask_threshold_postprocess: bool = False,
    aligned_sam_mask_probability_threshold: float = 0.5,
    direct_sam_mask_probability_threshold: float = 0.5,
    enable_reflection: bool = True,
    force_direct_after_reflection: bool = False,
    enable_reflect_decision_logit_threshold: bool = False,
    reflect_accept_probability_threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    image = (
        Image.open(sample.image_path).convert("RGB")
        if not isinstance(sample.raw_item.get("image_obj"), Image.Image)
        else sample.raw_item["image_obj"].convert("RGB")
    )
    collated = _collate_samples_for_oracle(
        model=geometry_model,
        samples=[sample],
        resize_size=resize_size,
        sam_image_size=sam_image_size,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )
    model_inputs = collated["model_inputs"]

    # Stage1/aligned use base Qwen. Direct LoRA is enabled only when building the direct branch embedding.
    with geometry_model.direct_lora_disabled():
        cached_hidden, cached_mask = geometry_model.extract_cached_query_hidden_for_inputs(
            model_inputs,
            branch="aligned",
        )
        aligned_embedding, _ = geometry_model.build_connector_embeddings(
            cached_hidden,
            cached_mask,
            branch="aligned",
        )
    aligned_embedding = aligned_embedding[0]

    # 2026-04-25: 保持当前执行顺序不变；stage1 先生成 bbox / points / think /
    # output_text，然后立刻生成用于 reflect 的 crop。
    # try:
    with geometry_model.direct_lora_disabled():
        generation = geometry_model._generate_once(
            model_inputs,
            max_new_tokens=stage1_max_new_tokens,
            do_sample=False,
        )[0][0]
    geometry = _sanitize_bbox_points(
        generation.answer,
        width=image.width,
        height=image.height,
        resize_size=resize_size,
    )
    stage1 = {
        "output_text": generation.output_text,
        "think": generation.think,
        "answer": generation.answer,
        "bbox": geometry["bbox"],
        "points": geometry["points"],
    }
    stage1_mask = _run_branch_from_collated_batch(
        model=geometry_model,
        collated_batch=collated,
        idx=0,
        branch_embedding=aligned_embedding,
        input_box=stage1["bbox"],
        input_points=stage1["points"],
        refiner_points=stage1["points"],
        use_points=True,
        mode = "aligned"
    )
    final_bbox = stage1["bbox"] 
    final_points = stage1["points"]

    if not enable_reflection:
        aligned_result = _apply_sam_mask_threshold_postprocess(
            stage1_mask,
            enabled=enable_sam_mask_threshold_postprocess,
            probability_threshold=aligned_sam_mask_probability_threshold,
        )
        extra = {
            "selected_branch": "aligned_no_reflect",
            "reflection_enabled": False,
            "decision": "",
            "second_confidence": 0.0,
            "decision_mode_used": "disabled",
            "decision_confidence_threshold": float(confidence_threshold),
            "parsed_decision": "",
            "parsed_confidence": 0.0,
            "reflect_output_text": "",
            "stage1_output_text": stage1["output_text"],
            "stage1_bbox": stage1["bbox"],
            "stage1_points": stage1["points"],
            "stage1_proposal_visualization_path": "",
            "generated_bbox": stage1["bbox"],
            "generated_points": stage1["points"],
            "used_reject_direct_attention_mask": False,
            "reject_direct_attention_mask_source": "",
            "reject_direct_attention_mask_token_count": 0,
            "used_direct_query_for_direct_branch": False,
            "direct_prompt_mode": "disabled",
            "reject_direct_prompt_path": "",
        }
        if enable_sam_mask_threshold_postprocess:
            extra.update(
                {
                    "sam_mask_threshold_postprocess_enabled": True,
                    "aligned_sam_mask_probability_threshold": float(
                        aligned_sam_mask_probability_threshold
                    ),
                    "direct_sam_mask_probability_threshold": float(
                        direct_sam_mask_probability_threshold
                    ),
                }
            )
        extra.update(aligned_result.get("sam_mask_threshold_metadata", {}))
        final_record = _mask_branch_record(
            image_id=sample.image_id,
            sample_id=sample.sample_id,
            question=sample.refexp,
            output_text=stage1["output_text"],
            think=stage1["think"],
            pred_bbox=aligned_result["pred_box"],
            pred_points=final_points,
            pred_mask=aligned_result["pred_mask"],
            gt_mask=sample.gt_mask,
            branch_type="aligned",
            extra=extra,
        )
        final_record["selected_branch"] = "aligned_no_reflect"
        return {"final_record": final_record}

    # print("final_points",final_points)
    # aligned_result = _run_branch_from_collated_batch(
    #     model=geometry_model,
    #     collated_batch=collated,
    #     idx=0,
    #     branch_embedding=aligned_embedding,
    #     input_box=final_bbox,
    #     input_points=final_points,
    #     use_points=True,
    # )
    reflect_image = _render_stage1_proposal_visualization(
        image=image,
        pred_bbox=stage1["bbox"],
    )
    stage1_proposal_visualization_path = ""
    if save_stage1_proposal_visualizations:
        stage1_proposal_visualization_path = _save_stage1_proposal_visualization(
            reflect_image=reflect_image,
            output_dir=output_dir,
            part_idx=part_idx,
            image_id=sample.image_id,
            sample_id=sample.sample_id,
            question=sample.refexp,
        )
    # except Exception as exc:
    #     extra = {
    #         "selected_branch": "parse_error_direct",
    #         "decision": "reject",
    #         "second_confidence": 0.0,
    #         "reflect_output_text": "",
    #         "stage1_output_text": "",
    #         "stage1_bbox": [0, 0, 0, 0],
    #         "stage1_points": [[0, 0], [0, 0]],
    #     }
    #     direct_prediction = _run_branch_from_collated_batch(
    #         model=geometry_model,
    #         collated_batch=collated,
    #         idx=0,
    #         branch_embedding=direct_embedding,
    #         input_box=None,
    #         input_points=None,
    #         use_points=False,
    #     )
    #     final_record = _mask_branch_record(
    #         image_id=sample.image_id,
    #         sample_id=sample.sample_id,
    #         question=sample.refexp,
    #         output_text="",
    #         think="",
    #         pred_bbox=direct_prediction["pred_box"],
    #         pred_points=[[0, 0], [0, 0]],
    #         pred_mask=direct_prediction["pred_mask"],
    #         gt_mask=sample.gt_mask,
    #         branch_type="direct",
    #         extra=extra,
    #     )
    #     final_record["selected_branch"] = "parse_error_direct"
    #     final_record["error"] = f"stage1_parse_error: {exc}"
    #     return {"final_record": final_record}

    # 2026-04-25: reflect 阶段只基于 stage1 输出做接受/拒绝判断，不重新生成
    # 几何结果；是否替换语言模型只体现在 reflect_model 上。
    reflect_prompt = build_query_reflect_reflect_rl_prompt(
        question=sample.refexp,
    )
    reflect_lora_context = reflect_model.direct_lora_disabled() if hasattr(reflect_model, "direct_lora_disabled") else nullcontext()
    with reflect_lora_context:
        if enable_reflect_decision_logit_threshold:
            reflect_output_text, decision_score_info = _generate_reflect_output_with_decision_scores(
                reflect_model=reflect_model,
                reflect_image=reflect_image,
                prompt_text=reflect_prompt,
                max_new_tokens=reflect_max_new_tokens,
            )
        else:
            reflect_output_text = _generate_reflect_output(
                reflect_model=reflect_model,
                reflect_image=reflect_image,
                prompt_text=reflect_prompt,
                max_new_tokens=reflect_max_new_tokens,
            )
            decision_score_info = {}
    ## 基于模型输出的接受还是拒绝
    parsed = parse_reflect_output(reflect_output_text)
    if enable_reflect_decision_logit_threshold:
        score_valid = bool(decision_score_info.get("valid", False))
        if not parsed.get("valid", False) or not score_valid:
            reflection_selected_branch = "parse_error_direct"
        else:
            accept_probability = float(decision_score_info["accept_probability"])
            reflection_selected_branch = (
                "accept_aligned"
                if accept_probability >= float(reflect_accept_probability_threshold)
                else "reject_direct"
            )
    else:
        reflection_selected_branch = _select_query_reflect_branch(
            parsed=parsed,
            decision_mode=decision_mode,
            confidence_threshold=confidence_threshold,
        )
    chosen_branch = (
        "reject_direct"
        if force_direct_after_reflection
        else reflection_selected_branch
    )
    # print("reflect_output_text",reflect_output_text)
    # parsed = parse_reflect_output(reflect_output_text)
    # think_reject = _think_implies_reject(reflect_output_text)
    # chosen_branch = "reject_direct" if think_reject else "accept_aligned"

    use_aligned = chosen_branch == "accept_aligned"
    extra = {
        "selected_branch": chosen_branch,
        "decision": parsed["decision"],
        "second_confidence": float(parsed["confidence"]),
        "decision_mode_used": str(decision_mode),
        "decision_confidence_threshold": float(confidence_threshold),
        "parsed_decision": parsed["decision"],
        "parsed_confidence": float(parsed["confidence"]),
        "reflect_output_text": reflect_output_text,
        "stage1_output_text": stage1["output_text"],
        "stage1_bbox": stage1["bbox"],
        "stage1_points": stage1["points"],
        "stage1_proposal_visualization_path": stage1_proposal_visualization_path,
        "generated_bbox": stage1["bbox"],
        "generated_points": stage1["points"],
        "used_reject_direct_attention_mask": False,
        "reject_direct_attention_mask_source": "",
        "reject_direct_attention_mask_token_count": 0,
        "used_direct_query_for_direct_branch": bool(use_direct_query_for_direct_branch),
        "direct_prompt_mode": str(direct_branch_prompt_mode or "geometric"),
        "reject_direct_prompt_path": "",
    }
    if enable_reflect_decision_logit_threshold:
        extra.update(
            {
                "reflect_decision_logit_threshold_enabled": True,
                "reflect_accept_probability_threshold": float(
                    reflect_accept_probability_threshold
                ),
                "reflect_decision_score_valid": bool(
                    decision_score_info.get("valid", False)
                ),
                "reflect_generated_conclusion_decision": str(
                    decision_score_info.get("generated_conclusion_decision", "")
                ),
                "reflect_decision_token_index": int(
                    decision_score_info.get("decision_token_index", -1)
                ),
                "reflect_accept_token_ids": decision_score_info.get("accept_token_ids", []),
                "reflect_reject_token_ids": decision_score_info.get("reject_token_ids", []),
                "reflect_accept_logprob": float(
                    decision_score_info.get("accept_logprob", 0.0)
                ),
                "reflect_reject_logprob": float(
                    decision_score_info.get("reject_logprob", 0.0)
                ),
                "reflect_decision_logit_margin": float(
                    decision_score_info.get("logit_margin", 0.0)
                ),
                "reflect_accept_probability": float(
                    decision_score_info.get("accept_probability", 0.0)
                ),
                "reflect_decision_scoring_method": str(
                    decision_score_info.get("scoring_method", "")
                ),
                "reflect_decision_score_error": str(
                    decision_score_info.get("error", "")
                ),
            }
        )
    if force_direct_after_reflection:
        extra.update(
            {
                "force_direct_after_reflection": True,
                "reflection_selected_branch": reflection_selected_branch,
                "branch_selection_overridden": reflection_selected_branch != chosen_branch,
            }
        )
    if enable_sam_mask_threshold_postprocess:
        extra.update(
            {
                "sam_mask_threshold_postprocess_enabled": True,
                "aligned_sam_mask_probability_threshold": float(
                    aligned_sam_mask_probability_threshold
                ),
                "direct_sam_mask_probability_threshold": float(
                    direct_sam_mask_probability_threshold
                ),
            }
        )

    # 2026-04-25: 这里是 query-reflect 的最终分支选择，不是 fallback 逻辑。
    if use_aligned:
        aligned_result = _run_branch_from_collated_batch(
            model=geometry_model,
            collated_batch=collated,
            idx=0,
            branch_embedding=aligned_embedding,
            input_box=final_bbox,
            input_points=final_points,
            refiner_points=final_points,
            use_points=True,
            mode = "aligned",
        )
        aligned_result = _apply_sam_mask_threshold_postprocess(
            aligned_result,
            enabled=enable_sam_mask_threshold_postprocess,
            probability_threshold=aligned_sam_mask_probability_threshold,
        )
        extra.update(aligned_result.get("sam_mask_threshold_metadata", {}))
        final_record = _mask_branch_record(
            image_id=sample.image_id,
            sample_id=sample.sample_id,
            question=sample.refexp,
            output_text=reflect_output_text,
            think="",
            pred_bbox=aligned_result["pred_box"],
            pred_points=final_points,
            pred_mask=aligned_result["pred_mask"],
            gt_mask=sample.gt_mask,
            branch_type="aligned",
            extra=extra,
        )
    else:
        if force_direct_after_reflection:
            # Stage1 already produced the same aligned candidate required for reflection.
            # Reuse it only for the optional comparison visualization; the final result
            # below is always produced by the direct branch.
            aligned_candidate = stage1_mask
        else:
            aligned_candidate = _run_branch_from_collated_batch(
                model=geometry_model,
                collated_batch=collated,
                idx=0,
                branch_embedding=aligned_embedding,
                input_box=stage1["bbox"],
                input_points=stage1["points"],
                refiner_points=stage1["points"],
                use_points=True,
                mode = "aligned",
            )
        aligned_candidate = _apply_sam_mask_threshold_postprocess(
            aligned_candidate,
            enabled=enable_sam_mask_threshold_postprocess,
            probability_threshold=aligned_sam_mask_probability_threshold,
        )
        if use_direct_query_for_direct_branch:
            try:
                direct_refiner_points = geometry_model._require_two_direct_refiner_points(
                    final_points, (image.height, image.width)
                )
            except Exception as exc:
                final_record = _empty_mask_branch_record(
                    image_id=sample.image_id,
                    sample_id=sample.sample_id,
                    question=sample.refexp,
                    output_text=stage1["output_text"],
                    think=stage1["think"],
                    gt_mask=sample.gt_mask,
                    branch_type="direct",
                    extra=extra,
                    error=f"invalid_direct_refiner_points: {exc}",
                )
                final_record["selected_branch"] = chosen_branch
                return {"final_record": final_record}

            direct_embedding_for_branch, resolved_prompt_mode, direct_prompt_text = _build_direct_branch_embedding(
                model=geometry_model,
                base_model_inputs=model_inputs,
                image=image,
                question=sample.refexp,
                reflect_image=reflect_image,
                reflect_prompt=reflect_prompt,
                stage1_output_text=stage1["output_text"],
                reflect_output_text=reflect_output_text,
                resize_size=resize_size,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
                direct_branch_prompt_mode=direct_branch_prompt_mode,
            )
            direct_variant = f"direct_query_{resolved_prompt_mode}"
            extra["direct_prompt_mode"] = resolved_prompt_mode
            if save_reject_direct_prompts:
                extra["reject_direct_prompt_path"] = _save_reject_direct_prompt(
                    prompt_text=direct_prompt_text,
                    output_dir=output_dir,
                    part_idx=part_idx,
                    image_id=sample.image_id,
                    sample_id=sample.sample_id,
                    question=sample.refexp,
                )
                tqdm.write(
                    "[reject-direct-prompt] "
                    f"sample_id={sample.sample_id} path={extra['reject_direct_prompt_path']}"
                )

            direct_prediction = _run_branch_from_collated_batch(
                model=geometry_model,
                collated_batch=collated,
                idx=0,
                branch_embedding=direct_embedding_for_branch,
                input_box=final_bbox,
                input_points=final_points,
                refiner_points=direct_refiner_points,
                use_points=True,
                mode = "direct",
            )

        elif use_reject_direct_attention_mask:
            direct_variant = "hard_masked_direct"
            direct_prediction, mask_debug = _run_masked_direct_prediction(
                model=geometry_model,
                collated_batch=collated,
                image=image,
                stage1_bbox=stage1["bbox"],
                gt_mask=sample.gt_mask,
                gt_box=sample.gt_box_xyxy,
                sam_image_size=sam_image_size,
            )
            extra.update(mask_debug)

        else:
            direct_embedding_for_branch, resolved_prompt_mode, direct_prompt_text = _build_direct_branch_embedding(
                model=geometry_model,
                base_model_inputs=model_inputs,
                image=image,
                question=sample.refexp,
                reflect_image=reflect_image,
                reflect_prompt=reflect_prompt,
                stage1_output_text=stage1["output_text"],
                reflect_output_text=reflect_output_text,
                resize_size=resize_size,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
                direct_branch_prompt_mode="geometric",
            )
            direct_variant = f"baseline_direct_{resolved_prompt_mode}"
            direct_prediction = _run_branch_from_collated_batch(
                model=geometry_model,
                collated_batch=collated,
                idx=0,
                branch_embedding=direct_embedding_for_branch,
                input_box=stage1["bbox"],
                input_points=None,
                use_points=False,
                mode = "direct",
            )

        direct_prediction = _apply_sam_mask_threshold_postprocess(
            direct_prediction,
            enabled=enable_sam_mask_threshold_postprocess,
            probability_threshold=direct_sam_mask_probability_threshold,
        )
        extra.update(direct_prediction.get("sam_mask_threshold_metadata", {}))

        final_record = _mask_branch_record(
            image_id=sample.image_id,
            sample_id=sample.sample_id,
            question=sample.refexp,
            output_text=stage1["output_text"],
            think=stage1["think"],
            pred_bbox=direct_prediction["pred_box"],
            pred_points=[[0, 0], [0, 0]],
            pred_mask=direct_prediction["pred_mask"],
            gt_mask=sample.gt_mask,
            branch_type="direct",
            extra=extra,
        )
        should_save_reject_vis = bool(save_reject_branch_visualizations)
        if should_save_reject_vis:
            _save_reject_branch_side_by_side_visualization(
                image=image,
                aligned_pred_mask=aligned_candidate["pred_mask"],
                aligned_pred_bbox=aligned_candidate["pred_box"],
                aligned_pred_points=stage1["points"],
                aligned_iou=(
                    0.0
                    if bbox_only_metrics
                    else float(compute_iou(aligned_candidate["pred_mask"], sample.gt_mask)[2])
                ),
                aligned_bbox_iou=float(compute_box_iou(aligned_candidate["pred_box"], sample.gt_box_xyxy)),
                direct_pred_mask=direct_prediction["pred_mask"],
                direct_pred_bbox=direct_prediction["pred_box"],
                direct_pred_points=[[0, 0], [0, 0]],
                direct_iou=(
                    0.0
                    if bbox_only_metrics
                    else float(compute_iou(direct_prediction["pred_mask"], sample.gt_mask)[2])
                ),
                direct_bbox_iou=float(compute_box_iou(direct_prediction["pred_box"], sample.gt_box_xyxy)),
                gt_mask=sample.gt_mask,
                gt_bbox=sample.gt_box_xyxy,
                output_dir=output_dir,
                part_idx=part_idx,
                sample_id=sample.sample_id,
                parsed_decision=parsed["decision"],
                parsed_confidence=float(parsed["confidence"]),
                direct_variant=direct_variant,
                image_id=sample.image_id,
                question=sample.refexp,
                bbox_only_metrics=bbox_only_metrics,
            )

    final_record["selected_branch"] = chosen_branch
    return {"final_record": final_record}


def run_prediction_loop(
    dataset: QueryReflectDataset,
    output_dir: str,
    idx: int,
    num_parts: int,
    geometric_export_dir: str,
    shared_mllm_path: Optional[str],
    resize_size: int,
    sam_image_size: int,
    max_pixels: int,
    min_pixels: int,
    stage1_max_new_tokens: int,
    reflect_max_new_tokens: int,
    confidence_threshold: float,
    decision_mode: str,
    save_branch_breakdown: bool = False,
    limit: int = -1,
    use_reject_direct_attention_mask: bool = False,
    save_reject_branch_visualizations: bool = False,
    reject_branch_visualization_limit: int = -1,
    save_stage1_proposal_visualizations: bool = False,
    save_reject_direct_prompts: bool = False,
    use_direct_query_for_direct_branch: bool = False,
    direct_branch_prompt_mode: str = "geometric",
    bbox_only_metrics: bool = False,
    enable_sam_mask_threshold_postprocess: bool = False,
    aligned_sam_mask_probability_threshold: float = 0.5,
    direct_sam_mask_probability_threshold: float = 0.5,
    enable_reflection: bool = True,
    force_direct_after_reflection: bool = False,
    enable_reflect_decision_logit_threshold: bool = False,
    reflect_accept_probability_threshold: float = 0.5,

) -> Path:
    if force_direct_after_reflection and not enable_reflection:
        raise ValueError(
            "force_direct_after_reflection=True requires enable_reflection=True"
        )
    if enable_reflect_decision_logit_threshold and not enable_reflection:
        raise ValueError(
            "enable_reflect_decision_logit_threshold=True requires enable_reflection=True"
        )
    if not 0.0 <= float(reflect_accept_probability_threshold) <= 1.0:
        raise ValueError(
            "reflect_accept_probability_threshold must be between 0 and 1, got "
            f"{reflect_accept_probability_threshold}"
        )
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    reflect_mllm_path = shared_mllm_path if enable_reflection else ""
    geometry_model, reflect_model = _load_branch_models(geometric_export_dir, reflect_mllm_path, device)
    selected_indices = list(split_indices(len(dataset), idx, num_parts))
    if limit > 0:
        selected_indices = selected_indices[:limit]
    records: List[Dict[str, Any]] = []
    aligned_records: List[Dict[str, Any]] = []
    direct_records: List[Dict[str, Any]] = []

    progress = tqdm(
        selected_indices,
        desc=f"query-reflect part {idx}",
        position=idx,
        leave=True,
    )
    for sample_idx in progress:
        sample = dataset[sample_idx]
        refine_sample = RefineSample(
            image_path=getattr(sample, "image_path", ""),
            image_id=sample.image_id,
            sample_id=sample.ann_id,
            question_text=sample.question,
            refexp=sample.question,
            gt_mask=sample.gt_mask,
            gt_box_xyxy=sample.gt_box,
            width=sample.image.width if hasattr(sample, "image") else sample.gt_mask.shape[1],
            height=sample.image.height if hasattr(sample, "image") else sample.gt_mask.shape[0],
            raw_item={"image_obj": sample.image} if hasattr(sample, "image") else {},
        )
        if not refine_sample.image_path:
            temp_dir = Path(output_dir) / "_tmp_images"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{refine_sample.sample_id}.png"
            refine_sample.raw_item["image_obj"].save(temp_path)
            refine_sample.image_path = str(temp_path)

        payload = run_single_prediction(
            geometry_model=geometry_model,
            reflect_model=reflect_model,
            sample=refine_sample,
            resize_size=resize_size,
            sam_image_size=sam_image_size,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            stage1_max_new_tokens=stage1_max_new_tokens,
            reflect_max_new_tokens=reflect_max_new_tokens,
            enable_reflection=enable_reflection,
            force_direct_after_reflection=force_direct_after_reflection,
            confidence_threshold=confidence_threshold,
            use_reject_direct_attention_mask=use_reject_direct_attention_mask,
            decision_mode=decision_mode,
            output_dir=output_dir,
            part_idx=idx,
            save_reject_branch_visualizations=save_reject_branch_visualizations,
            reject_branch_visualization_limit=reject_branch_visualization_limit,
            save_stage1_proposal_visualizations=save_stage1_proposal_visualizations,
            save_reject_direct_prompts=save_reject_direct_prompts,
            use_direct_query_for_direct_branch=use_direct_query_for_direct_branch,
            direct_branch_prompt_mode=direct_branch_prompt_mode,
            bbox_only_metrics=bbox_only_metrics,
            enable_sam_mask_threshold_postprocess=enable_sam_mask_threshold_postprocess,
            aligned_sam_mask_probability_threshold=aligned_sam_mask_probability_threshold,
            direct_sam_mask_probability_threshold=direct_sam_mask_probability_threshold,
            enable_reflect_decision_logit_threshold=enable_reflect_decision_logit_threshold,
            reflect_accept_probability_threshold=reflect_accept_probability_threshold,
        )
        final_record = payload["final_record"]
        if bbox_only_metrics:
            mask_record_fields = {
                "image_id",
                "ann_id",
                "question",
                "model_output_text",
                "model_think",
                "pred_bbox",
                "pred_points",
                "gt_bbox",
                "bbox_iou",
                "pred_mask_rle",
                "gt_mask_rle",
                "intersection",
                "union",
                "iou",
                "branch_type",
                "error",
            }
            extra = {
                key: value
                for key, value in final_record.items()
                if key not in mask_record_fields
            }
            final_record = _bbox_branch_record(
                image_id=refine_sample.image_id,
                sample_id=refine_sample.sample_id,
                question=refine_sample.refexp,
                output_text=str(final_record.get("model_output_text", "")),
                think=str(final_record.get("model_think", "")),
                pred_bbox=[int(v) for v in final_record.get("pred_bbox", [0, 0, 0, 0])],
                pred_points=final_record.get("pred_points", [[0, 0], [0, 0]]),
                gt_bbox=refine_sample.gt_box_xyxy,
                branch_type=str(final_record.get("branch_type", "")),
                extra=extra,
                error=str(final_record.get("error", "")),
            )
        records.append(final_record)
        progress.set_postfix(
            sample_id=refine_sample.sample_id,
            branch=final_record.get("selected_branch", ""),
            iou=f"{float(final_record.get('iou', 0.0)):.4f}",
        )
        tqdm.write(
            "[query-reflect-eval] "
            f"sample_id={refine_sample.sample_id} "
            f"branch={final_record.get('selected_branch', '')} "
            f"iou={float(final_record.get('iou', 0.0)):.4f} "
            f"parsed_decision={final_record.get('parsed_decision', '')} "
            f"parsed_confidence={float(final_record.get('parsed_confidence', 0.0)):.4f}"
        )
        if save_branch_breakdown:
            branch = str(final_record.get("selected_branch", ""))
            if branch in {"accept_aligned", "aligned_no_reflect"}:
                aligned_records.append(final_record)
            else:
                direct_records.append(final_record)

    output_path = save_prediction_part(output_dir, idx, records)
    if save_branch_breakdown:
        save_prediction_part(str(Path(output_dir) / "aligned_only"), idx, aligned_records)
        save_prediction_part(str(Path(output_dir) / "direct_only"), idx, direct_records)
    return output_path
