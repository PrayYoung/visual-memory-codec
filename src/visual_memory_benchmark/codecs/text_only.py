from __future__ import annotations

import json

from PIL import Image, ImageDraw

from visual_memory_benchmark.codecs.base import BaseCodec
from visual_memory_benchmark.types import EncodedArtifact, SceneObject, SceneSample
from visual_memory_benchmark.data.synthetic_shapes import COLORS


class TextOnlyCodec(BaseCodec):
    def __init__(self, method_name: str, image_size: int, max_objects_in_caption: int = 8) -> None:
        super().__init__(method_name=method_name, image_size=image_size)
        self.max_objects_in_caption = max_objects_in_caption

    def encode(self, sample: SceneSample, budget_bytes: int) -> EncodedArtifact:
        objects = sorted(sample.objects, key=lambda obj: obj.bbox[0])[: self.max_objects_in_caption]
        serializable = {
            "canvas": self.image_size,
            "objects": [
                {
                    "shape": obj.shape,
                    "color": obj.color,
                    "bbox": list(obj.bbox),
                }
                for obj in objects
            ],
        }
        text = json.dumps(serializable, separators=(",", ":"))
        payload = self._truncate_utf8(text, budget_bytes)
        return EncodedArtifact(method_name=self.method_name, payload=payload, aux={"format": "utf8_json"})

    def decode(self, artifact: EncodedArtifact) -> Image.Image:
        image = Image.new("RGB", (self.image_size, self.image_size), color=(248, 248, 245))
        draw = ImageDraw.Draw(image)
        try:
            data = json.loads(artifact.payload.decode("utf-8"))
        except Exception:
            return image

        for item in data.get("objects", []):
            shape = item.get("shape")
            color_name = item.get("color")
            bbox = tuple(item.get("bbox", []))
            if len(bbox) != 4 or color_name not in COLORS:
                continue
            self._draw_shape(draw, shape, bbox, COLORS[color_name])
        return image

    @staticmethod
    def _truncate_utf8(text: str, budget_bytes: int) -> bytes:
        raw = text.encode("utf-8")
        if len(raw) <= budget_bytes:
            return raw
        trimmed = raw[:budget_bytes]
        while trimmed:
            try:
                json.loads(trimmed.decode("utf-8"))
                return trimmed
            except Exception:
                trimmed = trimmed[:-1]
        return b"{}"

    @staticmethod
    def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, bbox: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
        if shape == "circle":
            draw.ellipse(bbox, fill=color, outline=(30, 30, 30), width=2)
        elif shape == "square":
            draw.rectangle(bbox, fill=color, outline=(30, 30, 30), width=2)
        elif shape == "triangle":
            left, top, right, bottom = bbox
            mid_x = (left + right) / 2
            draw.polygon([(mid_x, top), (right, bottom), (left, bottom)], fill=color, outline=(30, 30, 30))
