from __future__ import annotations

from io import BytesIO

from PIL import Image

from visual_memory_benchmark.codecs.base import BaseCodec
from visual_memory_benchmark.types import EncodedArtifact, SceneSample


# Frozen before the pilot rerun. Fidelity is ordered lexicographically by this
# grid: larger decoded resolution first, then larger codec quality. No QA,
# metric, or per-image content signal participates in selection.
RATE_CONTROLLED_RESOLUTIONS = (256, 224, 192, 160, 128, 96, 64, 48, 32, 24, 16)
RATE_CONTROLLED_QUALITIES = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1)


class OriginalRawCodec(BaseCodec):
    def encode(self, sample: SceneSample, budget_bytes: int) -> EncodedArtifact:
        image = sample.image.convert("RGB")
        return EncodedArtifact(self.method_name, image.tobytes(), {"size": image.size, "mode": "RGB"})

    def decode(self, artifact: EncodedArtifact) -> Image.Image:
        return Image.frombytes(artifact.aux["mode"], artifact.aux["size"], artifact.payload)


class PillowImageCodec(BaseCodec):
    def __init__(
        self,
        method_name: str,
        image_size: int,
        fmt: str,
        resolutions: tuple[int, ...] = RATE_CONTROLLED_RESOLUTIONS,
        qualities: tuple[int, ...] = RATE_CONTROLLED_QUALITIES,
    ) -> None:
        super().__init__(method_name, image_size)
        self.fmt = fmt
        self.resolutions = tuple(size for size in resolutions if 0 < size <= image_size)
        self.qualities = tuple(qualities)
        if not self.resolutions or not self.qualities:
            raise ValueError("codec grid must contain at least one resolution and quality")
        self._candidate_cache: dict[str, list[tuple[int, int, bytes]]] = {}

    def encode(self, sample: SceneSample, budget_bytes: int) -> EncodedArtifact:
        if budget_bytes <= 0:
            raise ValueError("budget must be positive")
        candidates = self._candidate_cache.get(sample.sample_id)
        if candidates is None:
            candidates = self._encode_grid(sample.image)
            self._candidate_cache[sample.sample_id] = candidates
        for resolution, quality, payload in candidates:
            if len(payload) <= budget_bytes:
                with Image.open(BytesIO(payload)) as decoded:
                    decoded.convert("RGB").load()
                return EncodedArtifact(
                    self.method_name,
                    payload,
                    {"format": self.fmt, "quality": quality, "encoded_resolution": resolution,
                     "rate_controlled": len(self.resolutions) > 1},
                )
        raise ValueError(f"No {self.fmt} grid encoding fits {budget_bytes} B")

    def _encode_grid(self, source: Image.Image) -> list[tuple[int, int, bytes]]:
        image = source.convert("RGB")
        candidates: list[tuple[int, int, bytes]] = []
        for resolution in self.resolutions:
            resized = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
            for quality in self.qualities:
                buf = BytesIO()
                kwargs = {"format": self.fmt, "quality": quality}
                if self.fmt == "WEBP":
                    kwargs["method"] = 6
                resized.save(buf, **kwargs)
                candidates.append((resolution, quality, buf.getvalue()))
        return candidates

    def decode(self, artifact: EncodedArtifact) -> Image.Image:
        with Image.open(BytesIO(artifact.payload)) as image:
            return image.convert("RGB").copy()
