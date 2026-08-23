# Formal N=288 analysis

Portable result bundle: `results/formal_benchmark/canonical_n288/`

## Protocol and integrity checks

- The result bundle's `protocol.json` records frozen manifest SHA-256
  `2cd66fcc014c97c51cb07e58407f9aa43e8f124bb59224146ed8b745a407f12e`.
- The frozen QA has 288 images and 1,152 questions: existence 216, count 216,
  attribute 144, spatial 216, state 144, and relational reasoning 216.
- The benchmark wrote 12,096 method/image/budget rows.  There are 1,610
  explicitly infeasible codec rows: 627 AVIF and 983 fixed-resolution-WebP
  diagnostic rows.  Rate-controlled WebP is feasible on all 288 images at all
  six budgets.
- Every feasible constrained artifact has
  `artifact_file_bytes == stored_bytes <= budget_bytes`.  The raw oracle is
  intentionally unbudgeted at 196,608 bytes.
- `qa_predictions.json` has 10,486 feasible method/image records and 41,944
  answered frozen-QA items; every item has a prediction and correctness flag.

## Overall rate--QA results

The primary comparison is direct QA against actual mean stored bytes.  CIs in
the raw aggregate table are 95% image-bootstrap CIs with 10,000 resamples.

| Method | 4 KB: mean bytes / QA | 8 KB: mean bytes / QA |
|---|---:|---:|
| Text only | 1,394 / 0.191 | 2,145 / 0.188 |
| Visual latent | 3,928 / 0.503 | 3,928 / 0.503 |
| Hybrid | 2,626 / 0.231 | 5,326 / 0.337 |
| Rate-controlled WebP | 3,452 / 0.469 | 7,320 / 0.504 |
| AVIF | 3,614 / 0.442 (282/288 feasible) | 7,020 / 0.509 (284/288 feasible) |
| Original raw oracle | 196,608 / 0.539 | 196,608 / 0.539 |

WebP's complete curve is 0.272, 0.326, 0.411, 0.444, 0.469, and 0.504 QA at
221, 424, 879, 1,750, 3,452, and 7,320 mean stored bytes respectively.  Visual
latent rises from 0.168 at 174 B to 0.503 at 3,928 B, then does not increase at
the nominal 8 KB arm because its actual stored bytes remain 3,928 B.

## Paired comparisons with rate-controlled WebP

The following post-run checks use paired image bootstrap (10,000 resamples;
the same image intersection for each method) on per-image scene-QA accuracy.
They are complementary to the predeclared per-method CIs.

| Comparison, difference = method − WebP | 4 KB paired images / difference (95% CI) | 8 KB paired images / difference (95% CI) |
|---|---:|---:|
| Text only | 288 / −0.278 (−0.307, −0.248) | 288 / −0.317 (−0.347, −0.287) |
| Visual latent | 288 / +0.034 (+0.009, +0.059) | 288 / −0.002 (−0.024, +0.021) |
| Hybrid | 288 / −0.238 (−0.269, −0.205) | 288 / −0.168 (−0.197, −0.139) |
| AVIF | 282 / −0.027 (−0.054, 0.000) | 284 / +0.005 (−0.012, +0.023) |
| Original raw oracle | 288 / +0.070 (+0.045, +0.096) | 288 / +0.035 (+0.014, +0.056) |

At 4 KB, latent has a 3.4-point advantage over WebP, though it also uses 476
more mean bytes.  At 8 KB, latent and WebP are statistically indistinguishable
in QA while latent uses 3,392 fewer mean bytes.  Thus the evidence supports a
latent rate-efficiency advantage only at the high nominal arm where latent
underfills its budget; it does not support a higher 8 KB QA ceiling than WebP.

Text and hybrid are substantially below WebP at both rates.  AVIF is comparable
where feasible, but its estimates condition on a changing feasible subset and
cannot replace the all-image WebP comparison.

## Functional groups

At 4 KB, latent is credibly above WebP only for the predeclared spatial group:
`+0.065` over 216 paired images (95% CI `+0.005` to `+0.120`).  No latent-versus-
WebP functional-group difference excludes zero at 8 KB.  The raw oracle exceeds
WebP for spatial questions at both 4 KB (`+0.102`, CI `+0.051` to `+0.153`) and
8 KB (`+0.051`, CI `+0.009` to `+0.093`).  These are predeclared subgroup
estimates, but should be treated as descriptive because multiple groups and
rates are inspected.

## Interpretation and independent critique

The frozen formal result establishes a direct-QA rate tradeoff under this
GQA/Visual-Genome functional manifest: WebP produces the strongest monotonic
all-image rate curve, visual latent reaches comparable high-rate QA with fewer
actual bytes but no higher 8 KB QA, and text/hybrid do not match WebP here.

Important limits:

1. The original-image oracle reaches only 0.539 QA.  The small 8 KB gap between
   WebP and the oracle therefore describes this VLM evaluator and frozen QA
   setup, not human-recognizable image fidelity or an absolute memory ceiling.
2. Actual stored rates differ within each nominal-budget arm.  The 4 KB latent
   versus WebP advantage is not a perfectly equal-rate comparison (3,928 versus
   3,452 mean bytes), while the 8 KB latent result is a lower-rate plateau.
3. AVIF comparisons condition on feasible images; infeasible arms are correctly
   retained rather than silently substituted, but AVIF is not directly
   all-image comparable at every rate.
4. The formal run preserved the manifest hash, QA counts, codec accounting,
   evaluator isolation, and image-clustered inference protocol.  No evidence of
   protocol drift appears in the synchronized artifacts.  This review accepts
   the result as a valid benchmark measurement, not as proof that any codec is
   universally superior outside this benchmark composition.
