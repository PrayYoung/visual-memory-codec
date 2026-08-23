# Text-encoder robustness: stronger text encoder

## Status and frozen protocol

The completed control used the frozen deterministic N=144/Q=576 balanced
subset (seed `20260820`) tied to canonical manifest SHA-256
`2cd66fcc014c97c51cb07e58407f9aa43e8f124bb59224146ed8b745a407f12e`.

Only the text encoder changed: new text memories use
`Qwen/Qwen2.5-VL-7B-Instruct` in the unchanged `RealTextCodec`; frozen
Qwen2.5-VL-3B text memories were read without alteration.  Both arms were
answered by the same Qwen2.5-VL-7B direct-text evaluator using frozen
questions, prompts, normalization, and scoring.  The evaluator received text
only.  Analysis uses the predeclared 10,000-resample paired image bootstrap.

All 864 newly encoded text artifacts (144 images × 6 caps) were persisted and
fit their original caps (range 200--3,753 B). The result bundle has 1,728
per-scene arm rows and 6,912 QA answers.

## Results

The table reports scene-level QA.  Differences are Qwen2.5-VL-7B text encoder
minus frozen Qwen2.5-VL-3B text encoder, paired by image.

| nominal cap | frozen 3B QA | new 7B QA | paired difference (95% CI) |
| ---: | ---: | ---: | ---: |
| 256 B | 0.167 | 0.163 | -0.003 [-0.033, +0.026] |
| 512 B | 0.196 | 0.196 | +0.000 [-0.033, +0.033] |
| 1 KB | 0.191 | 0.194 | +0.003 [-0.030, +0.036] |
| 2 KB | 0.191 | 0.194 | +0.003 [-0.030, +0.036] |
| 4 KB | 0.200 | 0.201 | +0.002 [-0.030, +0.033] |
| 8 KB | 0.196 | 0.229 | +0.033 [-0.002, +0.069] |

The new encoder underfills the nominal caps, as did the frozen text encoder:
mean stored bytes for the 7B arm are 233, 415, 423, 423, 944, and 1,630 B
across the six caps.  These are reported rather than treated as equal-rate
payloads.

## Interpretation

Under the frozen comparison, there is no statistically clear evidence that
the canonical text-only weakness is materially sensitive to text-encoder
capacity.  Every paired 95% CI includes zero.  The largest point estimate is
at the nominal 8 KB cap (+0.033), but its interval still includes zero and the
absolute QA remains only 0.229 on this subset/evaluator.

This is robustness evidence about the text-only arm only.  It neither
strengthens nor overturns the frozen N=288 latent-versus-WebP claim, which was
not rerun or modified by Control B.

## Independent critique

The result cleanly tests an encoder-capacity contrast within the fixed Qwen 7B
evaluator, but not independence across evaluator families: the 7B model is
both new encoder and evaluator.  The N=144 subset supplies paired precision
but is not a replacement for the canonical N=288 benchmark.  The 8 KB
directional increase may be worth noting descriptively, but the predeclared CI
does not support declaring an encoder-capacity improvement.  No follow-up was
started.

## Artifacts

- Completion metadata: `results/formal_benchmark/control_b/run_provenance.json`
- Protocol and output bundle: `results/formal_benchmark/control_b/`
- Aggregate and paired tables:
  `text_encoder_summary/aggregate_metrics.csv` and `text_encoder_summary/pairwise.csv`
