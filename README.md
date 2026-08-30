# Visual Memory Codec — Paper 1

## Research question

At the same constrained storage budget, which persistent representation best
preserves the information that a vision-language QA evaluator can recover from
an image? Paper 1 compares text-only memory, visual latent memory, hybrid
memory, and rate-controlled WebP on a fixed natural-image benchmark.

## Frozen N=288 benchmark

The benchmark uses 288 GQA/Visual Genome images and 1,152 frozen questions
across six functional groups. All methods use actual stored-byte accounting;
the primary inference is a 10,000-resample paired image bootstrap. The frozen
manifest SHA-256 is
`2cd66fcc014c97c51cb07e58407f9aa43e8f124bb59224146ed8b745a407f12e`.

### Main result

At the nominal 4 KB arm, visual latent memory scored 0.503 QA at 3,928 mean
stored bytes versus WebP's 0.469 at 3,452 bytes: latent − WebP = **+0.034**
(paired 95% CI **[+0.009, +0.059]**). This is the canonical result, not a
claim of perfectly equal actual rate.

At the nominal 8 KB arm, latent remained at 3,928 bytes and 0.503 QA while
WebP reached 7,320 bytes and 0.504 QA. The paired difference is −0.002
([−0.024, +0.021]): latent underfills the cap and plateaus, so the benchmark
does not establish a higher 8 KB QA ceiling for latent memory.

### Predeclared robustness controls

- **Control A — evaluator:** Re-evaluating the frozen latent/WebP/raw-oracle
  artifacts with Qwen2.5-VL-7B gave a positive 4 KB point estimate (+0.0217),
  but its 95% CI [0.0000, +0.0425] touches zero. Under the frozen rule, the
  canonical contrast is **not established as evaluator-robust**.
- **Control B — text encoder:** Replacing only the text encoder with
  Qwen2.5-VL-7B on the deterministic N=144 subset produced no paired 95% CI
  excluding zero at any cap. It does not materially change the observed
  text-only weakness and does not test the latent-versus-WebP contrast.

### Scope and limitation

The raw-image oracle reaches 0.539 QA under the canonical evaluator (0.507
under Control A). Results therefore describe this manifest and evaluator
pipeline, not human-recognizable fidelity or a general ranking of codecs.

## Inspecting the release

- [Key claims table](paper_assets/table_1_key_results.csv)
- [Main rate–QA figure](paper_assets/figure_1_rate_qa.png)
- [Formal analysis](docs/formal_n288_analysis.md)
- [Evaluator robustness](docs/evaluator_robustness.md)
- [Text-encoder robustness](docs/text_encoder_robustness.md)
- [Frozen protocol](docs/robustness_protocol.md)
- [Portable result bundle](results/formal_benchmark/README.md)

The figure can be regenerated solely from tracked results:

```bash
PYTHONPATH=src python3 scripts/generate_paper1_release_artifacts.py
```

The complete frozen benchmark requires separately obtained GQA/Visual Genome
source data and model weights, which are not redistributed here. The tracked
synthetic benchmark remains a lightweight regression check:

```bash
PYTHONPATH=src python3 -m visual_memory_benchmark.run --config configs/synthetic_baseline.json
```

## Public contents

- `configs/`: frozen benchmark/control configurations and lightweight examples.
- `src/` and `scripts/`: benchmark and result-figure code.
- `results/formal_benchmark/`: compact machine-readable results, protocols,
  and manifest provenance.
- `paper_assets/`: the release figure and key-results table.
- `docs/`: concise methods, analysis, limitations, and robustness reports.

Raw images, encoded payloads, model caches, credentials, remote execution
records, local research state, and temporary outputs are intentionally excluded.
