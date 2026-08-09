from __future__ import annotations

import math
import random
from dataclasses import asdict

from PIL import Image, ImageDraw

from visual_memory_benchmark.types import SceneObject, SceneSample

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 60, 60),
    "blue": (70, 120, 230),
    "green": (60, 170, 90),
    "yellow": (230, 200, 70),
    "purple": (150, 90, 210),
    "orange": (240, 150, 60),
}

SHAPES = ("circle", "square", "triangle")


class SyntheticShapesDataset:
    def __init__(
        self,
        num_samples: int,
        image_size: int,
        min_objects: int,
        max_objects: int,
        seed: int,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.seed = seed

    def iter_samples(self) -> list[SceneSample]:
        rng = random.Random(self.seed)
        return [self._make_sample(rng, idx) for idx in range(self.num_samples)]

    def _make_sample(self, rng: random.Random, idx: int) -> SceneSample:
        image = Image.new("RGB", (self.image_size, self.image_size), color=(248, 248, 245))
        draw = ImageDraw.Draw(image)
        objects: list[SceneObject] = []

        num_objects = rng.randint(self.min_objects, self.max_objects)
        for _ in range(num_objects):
            shape = rng.choice(SHAPES)
            color_name = rng.choice(tuple(COLORS))
            box_size = rng.randint(self.image_size // 8, self.image_size // 4)
            left = rng.randint(8, self.image_size - box_size - 8)
            top = rng.randint(8, self.image_size - box_size - 8)
            bbox = (left, top, left + box_size, top + box_size)
            self._draw_shape(draw, shape, bbox, COLORS[color_name])
            objects.append(SceneObject(shape=shape, color=color_name, bbox=bbox))

        metadata = {
            "objects": [asdict(obj) for obj in objects],
            "counts": self._counts(objects),
        }
        return SceneSample(
            sample_id=f"scene_{idx:04d}",
            image=image,
            objects=objects,
            metadata=metadata,
        )

    @staticmethod
    def _draw_shape(
        draw: ImageDraw.ImageDraw,
        shape: str,
        bbox: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        if shape == "circle":
            draw.ellipse(bbox, fill=color, outline=(30, 30, 30), width=2)
        elif shape == "square":
            draw.rectangle(bbox, fill=color, outline=(30, 30, 30), width=2)
        else:
            left, top, right, bottom = bbox
            mid_x = (left + right) / 2
            points = [(mid_x, top), (right, bottom), (left, bottom)]
            draw.polygon(points, fill=color, outline=(30, 30, 30))

    @staticmethod
    def _counts(objects: list[SceneObject]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for obj in objects:
            key = f"{obj.color}_{obj.shape}"
            counts[key] = counts.get(key, 0) + 1
        return counts


def object_center(obj: SceneObject) -> tuple[float, float]:
    left, top, right, bottom = obj.bbox
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def relation(a: SceneObject, b: SceneObject) -> str:
    ax, ay = object_center(a)
    bx, by = object_center(b)
    dx = ax - bx
    dy = ay - by
    if abs(dx) > abs(dy):
        return "right_of" if dx > 0 else "left_of"
    return "below" if dy > 0 else "above"
