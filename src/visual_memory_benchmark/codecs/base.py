from __future__ import annotations

from abc import ABC, abstractmethod

from visual_memory_benchmark.types import EncodedArtifact, SceneSample


class BaseCodec(ABC):
    def __init__(self, method_name: str, image_size: int) -> None:
        self.method_name = method_name
        self.image_size = image_size

    @abstractmethod
    def encode(self, sample: SceneSample, budget_bytes: int) -> EncodedArtifact:
        raise NotImplementedError

    @abstractmethod
    def decode(self, artifact: EncodedArtifact):
        raise NotImplementedError
