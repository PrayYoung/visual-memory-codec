from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class SceneObject:
    shape: str
    color: str
    bbox: tuple[int, int, int, int]


@dataclass
class SceneSample:
    sample_id: str
    image: Image.Image
    objects: list[SceneObject] | None
    metadata: dict[str, Any]
    source_path: str | None = None


@dataclass
class EncodedArtifact:
    method_name: str
    payload: bytes
    aux: dict[str, Any]

    @property
    def stored_bytes(self) -> int:
        return len(self.payload)


@dataclass
class ReconstructionResult:
    sample_id: str
    method_name: str
    budget_bytes: int
    stored_bytes: int
    reconstruction: Image.Image
    artifact: EncodedArtifact


Array = np.ndarray
PathLike = str | Path
