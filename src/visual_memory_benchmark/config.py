from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MethodConfig:
    name: str
    kind: str
    params: dict[str, Any]


@dataclass
class DatasetConfig:
    type: str
    params: dict[str, Any]


@dataclass
class EvalConfig:
    type: str
    params: dict[str, Any]


@dataclass
class ExperimentConfig:
    run_name: str
    output_dir: str
    dataset: DatasetConfig
    budgets_bytes: list[int]
    methods: list[MethodConfig]
    evaluation: EvalConfig | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text())
        dataset = DatasetConfig(
            type=data["dataset"]["type"],
            params={k: v for k, v in data["dataset"].items() if k != "type"},
        )
        methods = [MethodConfig(**item) for item in data["methods"]]
        evaluation = None
        if "evaluation" in data:
            evaluation = EvalConfig(
                type=data["evaluation"]["type"],
                params={k: v for k, v in data["evaluation"].items() if k != "type"},
            )
        return cls(
            run_name=data["run_name"],
            output_dir=data["output_dir"],
            dataset=dataset,
            budgets_bytes=list(data["budgets_bytes"]),
            methods=methods,
            evaluation=evaluation,
        )
