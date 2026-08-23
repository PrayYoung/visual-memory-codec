# Visual Memory Codec

This repository benchmarks persistent representations for machine visual
memory under fixed stored-byte budgets.

## Frozen benchmark record

The formal N=288 GQA/Visual Genome benchmark and two predeclared robustness
checks are complete and frozen. At the nominal 4 KB arm, visual latent memory
exceeded rate-controlled WebP by +0.034 scene-QA accuracy (paired 95% CI
[+0.009, +0.059]). Under a second evaluator, the point estimate remained
positive but the interval touched zero; the stricter evaluator-robustness rule
is therefore not met. A stronger text encoder did not produce a paired
confidence interval excluding zero at any byte cap on the frozen N=144 subset.

- [Formal benchmark analysis](docs/formal_n288_analysis.md)
- [Evaluator robustness](docs/evaluator_robustness.md)
- [Text-encoder robustness](docs/text_encoder_robustness.md)
- [Frozen result tables and provenance](results/formal_benchmark/README.md)

The tracked result bundle contains portable tables, protocols, and manifest
provenance. Raw images, encoded payloads, model weights, credentials, caches,
and machine-specific execution state are intentionally excluded.

## Repository layout

- `configs/`: frozen benchmark configurations and baseline examples.
- `docs/`: method, protocol, and result analyses.
- `scripts/`: dataset preparation and benchmark entry points.
- `src/visual_memory_benchmark/`: reusable benchmark implementation.
- `results/formal_benchmark/`: archived, platform-independent result tables.

## Quickstart

Install the project dependencies, then run the lightweight synthetic regression
benchmark:

```bash
PYTHONPATH=src python3 -m visual_memory_benchmark.run --config configs/synthetic_baseline.json
```

Outputs are written to `outputs/<run_name>/` and are not tracked.

## Scope

The synthetic benchmark is a regression path. The formal natural-image result
is a fixed measurement under its documented manifest, byte accounting, and VLM
evaluation protocol. Its conclusions should not be generalized beyond those
conditions.
