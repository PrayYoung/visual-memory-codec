from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from visual_memory_benchmark.types import SceneSample


class CocoSubsetDataset:
    def __init__(
        self,
        images_dir: str,
        annotations_path: str,
        image_size: int,
        num_samples: int,
        min_annotations: int = 2,
        require_multiple_categories: bool = True,
        include_image_ids: list[int] | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.image_size = image_size
        self.num_samples = num_samples
        self.min_annotations = min_annotations
        self.require_multiple_categories = require_multiple_categories
        self.include_image_ids = set(include_image_ids or [])

        data = json.loads(self.annotations_path.read_text())
        self.images = {item["id"]: item for item in data["images"]}
        self.categories = {item["id"]: item["name"] for item in data["categories"]}
        anns_by_image: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
        self.anns_by_image = anns_by_image

    def iter_samples(self) -> list[SceneSample]:
        selected: list[SceneSample] = []
        image_ids = sorted(self.include_image_ids) if self.include_image_ids else sorted(self.images)
        for image_id in image_ids:
            if image_id not in self.images or image_id not in self.anns_by_image:
                continue
            anns = [ann for ann in self.anns_by_image[image_id] if ann.get("iscrowd", 0) == 0]
            if len(anns) < self.min_annotations:
                continue
            category_names = {self.categories[ann["category_id"]] for ann in anns}
            if self.require_multiple_categories and len(category_names) < 2:
                continue
            image_info = self.images[image_id]
            image_path = self.images_dir / image_info["file_name"]
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            selected.append(
                SceneSample(
                    sample_id=f"coco_{image_id}",
                    image=image,
                    objects=None,
                    metadata={
                        "image_id": image_id,
                        "file_name": image_info["file_name"],
                        "original_width": image_info["width"],
                        "original_height": image_info["height"],
                        "annotations": anns,
                        "categories": self.categories,
                    },
                    source_path=str(image_path),
                )
            )
            if len(selected) >= self.num_samples:
                break
        return selected
