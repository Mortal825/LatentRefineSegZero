from pathlib import Path

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_sam2_predictor(
    segmentation_model_path: str,
    sam_model_cfg: str,
    device: torch.device,
) -> SAM2ImagePredictor:
    """Load SAM2 from either a local ``.pt`` checkpoint or a model ID."""
    model_path = Path(segmentation_model_path)
    if model_path.suffix == ".pt":
        try:
            from sam2.build_sam import build_sam2
        except ImportError:
            try:
                from src.segment_anything_2.sam2.build_sam import build_sam2
            except ImportError as exc:
                raise ImportError(
                    "Failed to import build_sam2. Install SAM2 or make "
                    "src.segment_anything_2 importable."
                ) from exc
        sam_model = build_sam2(
            sam_model_cfg,
            segmentation_model_path,
            device=str(device),
        )
        return SAM2ImagePredictor(sam_model)
    return SAM2ImagePredictor.from_pretrained(segmentation_model_path)

