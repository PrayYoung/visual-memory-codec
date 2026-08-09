from __future__ import annotations

import math

import numpy as np
from PIL import Image

from visual_memory_benchmark.models.hf_adapters import DinoSimilarity, SemanticImageSimilarity


def psnr(original: Image.Image, reconstructed: Image.Image) -> float:
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(reconstructed.resize(original.size), dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return 99.0
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_like(original: Image.Image, reconstructed: Image.Image) -> float:
    a = np.asarray(original.convert("L"), dtype=np.float32)
    b = np.asarray(reconstructed.resize(original.size).convert("L"), dtype=np.float32)
    mu_a = float(np.mean(a))
    mu_b = float(np.mean(b))
    sigma_a = float(np.var(a))
    sigma_b = float(np.var(b))
    sigma_ab = float(np.mean((a - mu_a) * (b - mu_b)))
    c1 = 6.5025
    c2 = 58.5225
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return numerator / denominator if denominator else 0.0


class NaturalMetricEvaluator:
    def __init__(
        self,
        semantic_model_name: str = "google/siglip-base-patch16-224",
        dino_model_name: str = "facebook/dinov2-base",
    ) -> None:
        self.semantic = SemanticImageSimilarity(model_name=semantic_model_name)
        self.dino = DinoSimilarity(model_name=dino_model_name)

    def evaluate(self, original: Image.Image, reconstructed: Image.Image) -> dict[str, float]:
        return {
            "semantic_similarity": self.semantic.image_similarity(original, reconstructed),
            "dino_similarity": self.dino.image_similarity(original, reconstructed),
            "psnr": psnr(original, reconstructed),
            "ssim_like": ssim_like(original, reconstructed),
        }
