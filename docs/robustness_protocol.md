# Robustness protocol (frozen before execution)

## Canonical result

The formal N=288 benchmark has GQA/Visual Genome manifest SHA-256
`2cd66fcc014c97c51cb07e58407f9aa43e8f124bb59224146ed8b745a407f12e`.
No codec, main-benchmark evaluator, QA item, dataset, byte budget, or canonical
artifact will be changed or rerun for these controls.

## Control A: second VLM evaluator

Use `Qwen/Qwen2.5-VL-7B-Instruct` to answer the same frozen questions from
only the already-decoded, frozen visual-memory representations.  Evaluate all
288 images for visual latent, rate-controlled WebP, and raw-image oracle at
4,096 and 8,192 nominal bytes.  Store the original artifact byte count from
the canonical `per_scene_metrics.csv`; do not encode, decode, resize, or alter
the stored representation during this control.  Re-evaluate the predeclared
4 KB latent-minus-WebP contrast by paired image bootstrap (10,000 resamples),
with 8 KB as the plateau check.

Success criterion: the result reports the exact frozen-input provenance and a
paired 4 KB latent-minus-WebP CI under the second evaluator.  It is evaluator-
robust only if the direction remains positive and its CI excludes zero.

## Control B: stronger text encoder on a frozen representative subset

Use the same `Qwen/Qwen2.5-VL-7B-Instruct` model as the replacement text
encoder, with the existing `RealTextCodec` prompts, unit parsing, packing, and
six byte caps unchanged.  The subset is frozen before encoding: 144 images and
576 of the original questions, selected with seed `20260820` as 72 images from
the 144-image count/existence/relational/spatial pattern and 36 images from
each of the two 72-image patterns.  This yields exactly half of every original
functional-QA quota.

For every subset image and budget, compare the existing frozen Qwen2.5-VL-3B
text payload with the newly encoded 7B payload under the same strong evaluator
and exact artifact-file byte accounting.  The source image is provided only to
the new text encoder, never to the evaluator; the evaluator receives only the
stored text and frozen question.  The paired image-bootstrap comparison is 7B
text encoder minus frozen 3B text encoder at each budget.

Success criterion: every new text artifact fits its original byte cap, and the
paired results quantify whether a stronger encoder materially changes direct
text-memory QA.  A result near the 3B baseline supports an encoder-robust poor
text outcome; an improvement identifies encoder capacity as a contributing
factor.  Neither outcome changes the canonical N=288 main result.

## Critique guardrails

The same 7B model is used as the alternate evaluator and replacement encoder,
which makes Control B's within-evaluator, paired encoder comparison valid but
does not establish cross-family evaluator independence.  The controls are
robustness evidence, not opportunities to choose a new main codec or modify
the canonical result after seeing outcomes.
