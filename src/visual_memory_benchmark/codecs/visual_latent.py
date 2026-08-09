from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from visual_memory_benchmark.codecs.base import BaseCodec
from visual_memory_benchmark.types import EncodedArtifact, SceneSample


class VisualLatentCodec(BaseCodec):
    def __init__(self, method_name: str, image_size: int, max_grid_size: int = 32) -> None:
        super().__init__(method_name=method_name, image_size=image_size)
        self.max_grid_size = max_grid_size

    def encode(self, sample: SceneSample, budget_bytes: int) -> EncodedArtifact:
        grid_size = self._fit_grid_size(sample.image, budget_bytes)
        thumb = sample.image.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
        array = np.asarray(thumb, dtype=np.uint8)
        buffer = BytesIO()
        np.savez_compressed(buffer, rgb=array)
        payload = buffer.getvalue()
        if len(payload) > budget_bytes:
            payload = payload[:budget_bytes]
        return EncodedArtifact(
            method_name=self.method_name,
            payload=payload,
            aux={"grid_size": grid_size, "format": "npz_rgb_grid"},
        )

    def decode(self, artifact: EncodedArtifact) -> Image.Image:
        try:
            buffer = BytesIO(artifact.payload)
            data = np.load(buffer)
            rgb = data["rgb"].astype(np.uint8)
            image = Image.fromarray(rgb, mode="RGB")
            return image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        except Exception:
            return Image.new("RGB", (self.image_size, self.image_size), color=(248, 248, 245))

    def _fit_grid_size(self, image: Image.Image, budget_bytes: int) -> int:
        for size in range(self.max_grid_size, 3, -2):
            thumb = image.resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(thumb, dtype=np.uint8)
            buffer = BytesIO()
            np.savez_compressed(buffer, rgb=array)
            if len(buffer.getvalue()) <= budget_bytes:
                return size
        return 4
