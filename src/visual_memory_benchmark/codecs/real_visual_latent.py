from __future__ import annotations

from visual_memory_benchmark.codecs.base import BaseCodec
from visual_memory_benchmark.models.hf_adapters import (
    SdvVaeAdapter,
    deserialize_quantized_latent,
    quantize_latent,
    serialize_quantized_latent,
)
from visual_memory_benchmark.types import EncodedArtifact, SceneSample


class RealVisualLatentCodec(BaseCodec):
    def __init__(
        self,
        method_name: str,
        image_size: int,
        vae_model_name: str = "stabilityai/sd-vae-ft-mse",
        min_spatial_size: int = 4,
    ) -> None:
        super().__init__(method_name=method_name, image_size=image_size)
        self.vae = SdvVaeAdapter(model_name=vae_model_name)
        self.min_spatial_size = min_spatial_size
        self._latent_cache = {}

    def encode(self, sample: SceneSample, budget_bytes: int) -> EncodedArtifact:
        latent = self._latent_cache.get(sample.sample_id)
        if latent is None:
            latent = self.vae.encode(sample.image, image_size=self.image_size)
            self._latent_cache[sample.sample_id] = latent

        latent_h = int(latent.shape[-2])
        best_payload = None
        best_shape = None
        for size in range(latent_h, self.min_spatial_size - 1, -2):
            q = quantize_latent(latent, target_hw=(size, size))
            payload = serialize_quantized_latent(q)
            if len(payload) <= budget_bytes:
                best_payload = payload
                best_shape = (size, size)
                break

        if best_payload is None:
            raise ValueError(
                f"No valid visual latent representation fits budget {budget_bytes} bytes for sample {sample.sample_id}."
            )

        return EncodedArtifact(
            method_name=self.method_name,
            payload=best_payload,
            aux={"latent_hw": best_shape},
        )

    def decode(self, artifact: EncodedArtifact):
        import torch.nn.functional as F

        latent = deserialize_quantized_latent(artifact.payload)
        target_h = self.image_size // 8
        latent = F.interpolate(latent, size=(target_h, target_h), mode="bilinear", align_corners=False)
        return self.vae.decode(latent)
