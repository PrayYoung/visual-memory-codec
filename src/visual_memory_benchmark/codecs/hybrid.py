from __future__ import annotations

from io import BytesIO
import json

import numpy as np
from PIL import Image, ImageDraw

from visual_memory_benchmark.codecs.base import BaseCodec
from visual_memory_benchmark.data.synthetic_shapes import COLORS
from visual_memory_benchmark.types import EncodedArtifact, SceneSample


class HybridCodec(BaseCodec):
    def __init__(
        self,
        method_name: str,
        image_size: int,
        text_fraction: float = 0.45,
        max_objects_in_caption: int = 6,
        max_grid_size: int = 24,
    ) -> None:
        super().__init__(method_name=method_name, image_size=image_size)
        self.text_fraction = text_fraction
        self.max_objects_in_caption = max_objects_in_caption
        self.max_grid_size = max_grid_size

    def encode(self, sample: SceneSample, budget_bytes: int) -> EncodedArtifact:
        text_budget = max(32, int(budget_bytes * self.text_fraction))
        latent_budget = max(32, budget_bytes - text_budget)

        objects = sorted(sample.objects, key=lambda obj: (obj.color, obj.shape))[: self.max_objects_in_caption]
        summary = {
            "objects": [
                {
                    "s": obj.shape[0],
                    "c": obj.color[0],
                    "b": list(obj.bbox),
                }
                for obj in objects
            ]
        }
        text_payload = self._fit_text(summary, text_budget)
        latent_payload, grid_size = self._fit_latent(sample.image, latent_budget)

        combined = self._pack_payload(text_payload, latent_payload)
        if len(combined) > budget_bytes:
            overflow = len(combined) - budget_bytes
            latent_payload = latent_payload[:-overflow] if overflow < len(latent_payload) else b""
            combined = self._pack_payload(text_payload, latent_payload)

        return EncodedArtifact(
            method_name=self.method_name,
            payload=combined,
            aux={"grid_size": grid_size, "format": "hybrid_v1"},
        )

    def decode(self, artifact: EncodedArtifact) -> Image.Image:
        text_payload, latent_payload = self._unpack_payload(artifact.payload)
        base = Image.new("RGB", (self.image_size, self.image_size), color=(248, 248, 245))

        if latent_payload:
            try:
                rgb = np.load(BytesIO(latent_payload))["rgb"].astype(np.uint8)
                base = Image.fromarray(rgb, mode="RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            except Exception:
                pass

        try:
            data = json.loads(text_payload.decode("utf-8"))
        except Exception:
            return base

        overlay = base.copy()
        draw = ImageDraw.Draw(overlay)
        color_lookup = {name[0]: rgb for name, rgb in COLORS.items()}
        shape_lookup = {"c": "circle", "s": "square", "t": "triangle"}

        for item in data.get("objects", []):
            bbox = tuple(item.get("b", []))
            color = color_lookup.get(item.get("c"))
            shape = shape_lookup.get(item.get("s"))
            if color and shape and len(bbox) == 4:
                self._draw_shape(draw, shape, bbox, color)
        return overlay

    def _fit_text(self, summary: dict, text_budget: int) -> bytes:
        raw = json.dumps(summary, separators=(",", ":")).encode("utf-8")
        if len(raw) <= text_budget:
            return raw
        trimmed_objects = list(summary["objects"])
        while trimmed_objects:
            trimmed_objects.pop()
            raw = json.dumps({"objects": trimmed_objects}, separators=(",", ":")).encode("utf-8")
            if len(raw) <= text_budget:
                return raw
        return b"{}"

    def _fit_latent(self, image: Image.Image, latent_budget: int) -> tuple[bytes, int]:
        for size in range(self.max_grid_size, 3, -2):
            thumb = image.resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(thumb, dtype=np.uint8)
            buffer = BytesIO()
            np.savez_compressed(buffer, rgb=array)
            payload = buffer.getvalue()
            if len(payload) <= latent_budget:
                return payload, size
        return b"", 0

    @staticmethod
    def _pack_payload(text_payload: bytes, latent_payload: bytes) -> bytes:
        header = len(text_payload).to_bytes(4, byteorder="big", signed=False)
        return header + text_payload + latent_payload

    @staticmethod
    def _unpack_payload(payload: bytes) -> tuple[bytes, bytes]:
        if len(payload) < 4:
            return b"{}", b""
        text_len = int.from_bytes(payload[:4], byteorder="big", signed=False)
        text_payload = payload[4 : 4 + text_len]
        latent_payload = payload[4 + text_len :]
        return text_payload, latent_payload

    @staticmethod
    def _draw_shape(draw, shape: str, bbox: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
        if shape == "circle":
            draw.ellipse(bbox, fill=color, outline=(30, 30, 30), width=2)
        elif shape == "square":
            draw.rectangle(bbox, fill=color, outline=(30, 30, 30), width=2)
        elif shape == "triangle":
            left, top, right, bottom = bbox
            mid_x = (left + right) / 2
            draw.polygon([(mid_x, top), (right, bottom), (left, bottom)], fill=color, outline=(30, 30, 30))
