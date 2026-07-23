from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from training_scripts.eval.refcocog_common import get_refcocog_sample, load_refcocog_records
from training_scripts.eval.query_sam_lib import load_eval_dataset, mask_to_box


@dataclass
class ReflectSample:
    image: Image.Image
    question: str
    gt_mask: np.ndarray
    gt_box: List[int]
    image_id: str
    sample_id: str
    ann_id: str
    raw_item: Dict[str, Any]


class QueryReflectDataset(Dataset):
    def __init__(
        self,
        data_mode: str,
        image_root: str = "",
        train_json_path: str = "",
        dataset_path: str = "",
        ref_json_path: str = "",
        answer_resize: int = 840,
        sam_image_size: int = 1024,
    ) -> None:
        self.data_mode = data_mode
        self.image_root = image_root
        self.answer_resize = answer_resize
        self.sam_image_size = sam_image_size
        self.samples: List[Any]
        self._helper_dataset: Optional[Any] = None
        if data_mode in {"offline_json", "reasonseg_dataset"}:
            self._helper_dataset = load_eval_dataset(
                eval_mode=data_mode,
                eval_json_path=train_json_path if data_mode == "offline_json" else "",
                eval_dataset_path=dataset_path if data_mode == "reasonseg_dataset" else "",
                image_root=image_root,
                answer_resize=answer_resize,
                sam_image_size=sam_image_size,
            )
            self.samples = list(range(len(self._helper_dataset)))
        elif data_mode == "refcocog_json":
            self.samples = load_refcocog_records(ref_json_path)
        else:
            raise ValueError(f"Unsupported data_mode: {data_mode}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ReflectSample:
        if self.data_mode == "refcocog_json":
            sample = get_refcocog_sample(self.samples[index], self.image_root)
            return ReflectSample(
                image=sample["image"],
                question=sample["question"],
                gt_mask=np.asarray(sample["gt_mask"], dtype=np.uint8),
                gt_box=[int(v) for v in sample["gt_bbox"]],
                image_id=sample["image_id"],
                sample_id=str(sample["ann_id"]),
                ann_id=str(sample["ann_id"]),
                raw_item=sample,
            )

        assert self._helper_dataset is not None
        sample = self._helper_dataset[index]
        if hasattr(sample, "image"):
            image = sample.image.convert("RGB")
            gt_mask = np.asarray(sample.gt_mask, dtype=np.uint8)
            gt_box = mask_to_box(gt_mask)
            return ReflectSample(
                image=image,
                question=sample.question,
                gt_mask=gt_mask,
                gt_box=gt_box,
                image_id=sample.image_id,
                sample_id=str(index),
                ann_id=str(getattr(sample, "ann_id", index)),
                raw_item={"source": "reasonseg_dataset"},
            )
        image = Image.open(sample.image_path).convert("RGB")
        gt_mask = np.asarray(sample.gt_mask, dtype=np.uint8)
        gt_box = mask_to_box(gt_mask)
        return ReflectSample(
            image=image,
            question=sample.question,
            gt_mask=gt_mask,
            gt_box=gt_box,
            image_id=sample.image_id,
            sample_id=str(index),
            ann_id=str(sample.raw_item.get("ann_id", index)),
            raw_item=sample.raw_item,
        )


def collate_reflect_samples(batch: List[ReflectSample]) -> List[ReflectSample]:
    return batch
