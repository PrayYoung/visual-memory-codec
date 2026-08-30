# Visual Memory Codec

> **Persistent visual memory, compared at the same stored-byte budget.**

Visual Memory Codec asks a practical question for multimodal systems: after an
image must be stored as a compact memory, which representation preserves the
information a vision-language model can still use? Paper 1 is a frozen,
natural-image benchmark comparing text-only memory, visual latent memory,
hybrid memory, and rate-controlled WebP under exact byte accounting.

| Explore | Verify | Reproduce |
|:--|:--|:--|
| [Main figure](#main-figure) | [Key results](paper_assets/table_1_key_results.csv) | [Figure generator](scripts/generate_paper1_release_artifacts.py) |
| [Formal analysis](docs/formal_n288_analysis.md) | [Frozen protocol](docs/robustness_protocol.md) | [Result bundle](results/formal_benchmark/README.md) |

## Why this project matters

Many visual-memory systems compare representations by nominal size or image
quality alone. This benchmark instead measures what a fixed evaluator can
recover from the stored representation, while recording the actual bytes used
for every image and method. That makes the tradeoff between compressed visual
state and downstream question answering explicit.

## Main takeaway

> [!NOTE]
> In the frozen N=288 benchmark, visual latent memory beat rate-controlled
> WebP at the nominal 4 KB arm (+0.034 paired scene-QA; 95% CI
> [+0.009, +0.059]). At the nominal 8 KB arm, that advantage disappeared:
> latent underfilled the cap and plateaued at 3.93 KB.

## Main figure

![Paper 1 rate–QA curve: visual latent memory has a positive 4 KB contrast with WebP, then plateaus while WebP continues to improve.](paper_assets/figure_1_rate_qa.png)

*Figure 1. Scene-QA accuracy versus actual mean stored bytes. Error bars are
95% image-bootstrap intervals. The vertical guides denote nominal 4 KB and
8 KB arms; the plotted x-axis always uses actual stored bytes.*

## Key takeaways

- **4 KB:** latent reaches 0.503 QA at 3,928 mean bytes; WebP reaches 0.469
  at 3,452 bytes. The paired latent − WebP difference is +0.034
  ([+0.009, +0.059]).
- **8 KB:** latent remains at 3,928 bytes and 0.503 QA, while WebP reaches
  7,320 bytes and 0.504 QA. The paired difference is −0.002
  ([−0.024, +0.021]).
- **Text and hybrid:** neither matches WebP at the reported 4 KB or 8 KB
  arms in the canonical benchmark.
- **Interpretation:** the 4 KB result is a canonical measurement, not a claim
  of perfectly equal actual rate or a general ranking of image codecs.

## What Paper 1 studied

| Component | Frozen design |
|:--|:--|
| Images and questions | 288 GQA/Visual Genome images; 1,152 frozen questions across six functional groups |
| Representations | Text only, visual latent, hybrid, rate-controlled WebP; AVIF where feasible; unbudgeted raw-image oracle |
| Budgets | 256 B, 512 B, 1 KB, 2 KB, 4 KB, and 8 KB nominal caps with actual stored-byte accounting |
| Primary inference | 10,000-resample paired image bootstrap for method contrasts |
| Manifest | SHA-256 `2cd66fcc014c97c51cb07e58407f9aa43e8f124bb59224146ed8b745a407f12e` |

The full methodological record is in the [formal analysis](docs/formal_n288_analysis.md)
and [frozen robustness protocol](docs/robustness_protocol.md).

## Main findings

| Comparison | Result | What it supports |
|:--|:--|:--|
| Latent − WebP at nominal 4 KB | +0.034, 95% CI [+0.009, +0.059] | Positive canonical contrast under the original evaluator |
| Latent − WebP at nominal 8 KB | −0.002, 95% CI [−0.024, +0.021] | No higher 8 KB QA ceiling for latent; it underfills and plateaus |
| Raw-image oracle | 0.539 QA | An evaluator/manifest reference point, not a human-fidelity ceiling |

The compact [key-results table](paper_assets/table_1_key_results.csv) is the
machine-readable summary. The canonical
[aggregate table](results/formal_benchmark/canonical_n288/aggregate_metrics.csv)
and [per-scene table](results/formal_benchmark/canonical_n288/per_scene_metrics.csv)
support inspection of the frozen measurement.

## Robustness controls

| Control | Frozen change | Result |
|:--|:--|:--|
| **A — evaluator** | Re-evaluate the same latent, WebP, and raw-oracle artifacts with Qwen2.5-VL-7B | 4 KB direction remains positive (+0.0217), but CI [0.0000, +0.0425] touches zero. The strict evaluator-robustness rule is **not met**. |
| **B — text encoder** | Replace only the text encoder with Qwen2.5-VL-7B on a deterministic N=144 subset | No paired 95% CI excludes zero at any cap. No clear text-encoder-capacity effect; this does not test the latent-versus-WebP contrast. |

Read the [evaluator report](docs/evaluator_robustness.md) and
[text-encoder report](docs/text_encoder_robustness.md) for the complete,
predeclared interpretations.

## Limitations

- Results are specific to this frozen GQA/Visual Genome manifest and VLM
  evaluation pipeline; they are not claims about general human-perceived image
  fidelity.
- The raw-image oracle is 0.539 QA under the canonical evaluator and 0.507
  under Control A, which bounds absolute interpretation of the reported scores.
- Actual stored rates differ within nominal-budget arms. In particular, the
  latent 4 KB result uses 476 more mean bytes than WebP, while the 8 KB latent
  result remains at 3,928 bytes.
- Control A checks a second evaluator within the Qwen2.5-VL family; it is not
  cross-family evaluator independence.

## Repository structure

```text
paper_assets/                 Main figure and compact key-results table
configs/                      Frozen benchmark and control configurations
docs/                         Analysis, limitations, and robustness reports
results/formal_benchmark/     Protocols, provenance, and curated result tables
scripts/                      Benchmark entry points and figure generator
src/visual_memory_benchmark/  Reusable benchmark implementation
```

## Reproduce or inspect

The Paper 1 results are frozen. The release includes the code and compact
tables needed to inspect them; raw source images, model weights, encoded
payloads, and execution environments are intentionally not redistributed.

Regenerate the tracked main figure from the frozen aggregate table:

```bash
PYTHONPATH=src python3 scripts/generate_paper1_release_artifacts.py
```

Run the lightweight synthetic regression benchmark:

```bash
PYTHONPATH=src python3 -m visual_memory_benchmark.run --config configs/synthetic_baseline.json
```

For the complete frozen record, start with the [results bundle](results/formal_benchmark/README.md).
Credentials, model caches, raw images, remote execution state, local research
notes, and temporary outputs remain deliberately untracked.
