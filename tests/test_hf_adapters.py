from __future__ import annotations

import math
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

from visual_memory_benchmark.models.hf_adapters import SemanticImageSimilarity


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __call__(self, images, return_tensors="pt"):
        return _FakeBatch({"pixel_values": torch.ones((len(images), 3, 4, 4), dtype=torch.float32)})


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def get_image_features(self, **inputs):
        image_embeds = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float32)
        return SimpleNamespace(image_embeds=image_embeds)


def test_semantic_image_similarity_returns_finite_scalar():
    image_a = Image.new("RGB", (8, 8), color=(255, 0, 0))
    image_b = Image.new("RGB", (8, 8), color=(0, 0, 255))

    with mock.patch(
        "visual_memory_benchmark.models.hf_adapters._load_semantic_model",
        return_value=(_FakeProcessor(), _FakeModel()),
    ):
        similarity = SemanticImageSimilarity(model_name="google/siglip-base-patch16-224").image_similarity(image_a, image_b)

    assert isinstance(similarity, float)
    assert math.isfinite(similarity)
