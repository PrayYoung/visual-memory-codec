# Evaluator robustness: second VLM evaluator

Portable result bundle: `results/formal_benchmark/control_a/`

## Frozen-protocol checks

- The archived result record is complete and includes all required tables.
- `protocol.json` records only Control A, the canonical N=288 manifest hash
  `2cd66fcc014c97c51cb07e58407f9aa43e8f124bb59224146ed8b745a407f12e`,
  Qwen2.5-VL-7B-Instruct as the evaluator, and the frozen 10,000-resample
  image bootstrap.
- All 1,728 required evaluator rows are present (288 images x 3 methods x 2
  budgets), with 6,912 answered frozen-QA items. No text-control record or
  newly encoded artifact is present.
- Every source reconstruction SHA-256 matches the staged frozen-input
  provenance, and every copied stored-byte record matches the canonical
  `per_scene_metrics.csv`.

Only the evaluator model changed. Questions, frozen reconstruction files,
normalization, scoring, actual byte records, and bootstrap procedure remained
unchanged.

## Results under Qwen2.5-VL-7B

| Method | 4 KB: actual mean bytes / QA | 8 KB: actual mean bytes / QA |
|---|---:|---:|
| Visual latent | 3,928 / 0.451 | 3,928 / 0.451 |
| Rate-controlled WebP | 3,452 / 0.429 | 7,320 / 0.466 |
| Raw-image oracle | 196,608 / 0.507 | 196,608 / 0.507 |

| Paired comparison (method - WebP) | 4 KB difference (95% CI) | 8 KB difference (95% CI) |
|---|---:|---:|
| Visual latent | +0.0217 ([0.0000, +0.0425]) | -0.0156 ([-0.0365, +0.0061]) |
| Raw-image oracle | +0.0781 ([+0.0556, +0.1007]) | +0.0408 ([+0.0191, +0.0616]) |

## Predeclared decision

The 4 KB point estimate remains positive, but its paired image-bootstrap CI
touches zero. Under the frozen rule—positive direction **and** a CI excluding
zero—the canonical latent-over-WebP contrast is **not established as
evaluator-robust** under Qwen2.5-VL-7B.

The planned 8 KB plateau check is also consistent with no latent advantage:
the difference is -0.0156 with CI [-0.0365, +0.0061]. The raw oracle remains
above WebP at both rates, while still reflecting this evaluator and manifest
rather than a human-recognition ceiling.

## Independent critique

This control directly tests evaluator sensitivity on byte-identical inputs and
the frozen question set. It does not retest codec quality, equalize the small
actual-rate difference at 4 KB (3,928 versus 3,452 B), or establish
independence across model families: both evaluators are Qwen2.5-VL variants.
Therefore the appropriate conclusion is limited to failure of the strict
second-evaluator robustness criterion, not a reversal of the frozen canonical
3B-evaluator result.
