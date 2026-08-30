# Small GQA text-only forensic diagnostic

## Scope and status

This is a completed, separate diagnostic, not a rerun or modification of the
frozen N=288 benchmark. It used the frozen 64-image / 256-question sanitized
GQA manifest, Qwen2.5-VL-3B `RealTextCodec`, direct stored-text QA, and
actual-byte caps of 256, 512, and 1024 bytes.

The encoder received only the source image and its fixed factual-extraction prompt. The evaluator received only stored text and the evaluation question in the persisted exact evaluator prompt. The manifest excludes captions and scene graphs. The per-item audit has 768 records and includes every requested memory, byte count, question, gold answer, prompt, raw answer, normalized answer, and score.

## Primary result

| Actual-byte cap | Mean stored bytes | Direct text-memory QA | Image-bootstrap 95% CI | `unknown` answers |
| --- | ---: | ---: | ---: | ---: |
| 256 B | 219.4 | 0.117 (30/256) | [0.078, 0.156] | 70.3% |
| 512 B | 465.6 | 0.156 (40/256) | [0.109, 0.203] | 64.1% |
| 1 KB | 673.3 | 0.148 (38/256) | [0.105, 0.195] | 65.2% |

The 512 B arm gains 17 items and loses 7 relative to 256 B. The 1 KB arm gains only 4 and loses 6 relative to 512 B. Thus this forensic run does not reproduce strong low-rate direct-text QA, and it has no reliable monotonic gain from 512 B to 1 KB.

The same pattern holds across the six frozen groups. At 512 B, the highest group score is attribute (7/32, 0.219); relational reasoning is 3/48 (0.062). At 1 KB, the best group is spatial (9/48, 0.188); relational reasoning remains 4/48 (0.083).

## Integrity checks

- All 192 distinct persisted text memories (64 images x 3 budgets) are nonempty and satisfy their actual-byte caps. Observed ranges are 115--255 B, 115--511 B, and 115--864 B.
- There are 64 unique memories per arm. Only 2 images have identical 256/512-B memories, 4 have identical 512-B/1-KB memories, and 2 have identical 256-B/1-KB memories. The plateau is therefore not explained by wholesale reuse of a single low-budget payload.
- VQA-style normalization changes only 3, 4, and 3 raw answers in the respective arms, so the result is not materially a normalization artifact. Because GQA has one reference answer, final scoring is normalized exact match rather than VQAv2's ten-human soft score.

## Interpretation and limit

The observed result is low and dominated by the evaluator choosing `unknown`,
not strong low-rate text performance. This is evidence against an
implementation or scoring explanation that would turn this exact setup into a
strong result.

It is not a direct adjudication of results obtained under different examples,
encoder settings, or scoring protocols. Nor does this run prove that text
memory is intrinsically weak under every encoder: the stronger-encoder control
is the appropriate predeclared test of that narrower question. The formal N=288
result remains unchanged.

## Evidence

- `data/gqa/text_forensic_gqa64_manifest.json`
- `results/formal_benchmark/provenance/frozen_run_provenance.json`
- `results/formal_benchmark/text_forensic_gqa64/aggregate_metrics.csv`
- `results/formal_benchmark/text_forensic_gqa64/protocol.json`
