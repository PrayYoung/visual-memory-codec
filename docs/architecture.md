# Architecture Overview

## Goal

Evaluate which representation best preserves visual memory under a fixed stored-byte budget:

- text-only
- visual-latent-only
- hybrid text + visual residual

The experiment is neutral: the framework should reveal where each method dominates rather than assume a winner.

## Execution strategy

The natural-image benchmark is GPU-first.

Reason:

- the scientifically meaningful captioning, diffusion reconstruction, learned-latent, and evaluation models are not well matched to this local CPU-only machine
- forcing local-only operation would bias model choice downward and weaken the experiment

Therefore:

- local CPU remains the control/orchestration path
- A GPU-capable environment is the intended natural-image execution path
- the repository must remain runnable locally for regression and smoke testing without implying that local CPU results define the scientific benchmark

## First-version scope

- static images only
- reproducible on a small sample
- existing models later, simple local baseline now
- exact stored-byte accounting per scene
- reconstruction cost reported separately from storage cost

## Experiment flow

1. Load a dataset sample.
2. For each method and each byte budget:
   - encode the scene into a representation
   - enforce the total stored-byte limit
   - log exact stored bytes
   - reconstruct an image from only the stored representation
3. Evaluate each reconstruction with:
   - semantic recall
   - scene fidelity
   - image similarity metrics
4. Aggregate per-budget results.
5. Plot Pareto curves.

## Clean repository structure

```text
configs/
  synthetic_baseline.json
docs/
  architecture.md
src/visual_memory_benchmark/
  codecs/
  data/
  eval/
  utils/
  config.py
  run.py
outputs/
```

## Core interfaces

### Dataset loader

Responsibilities:

- enumerate samples
- load image and sample metadata
- expose stable sample IDs

Interface:

- `iter_samples() -> Iterable[SceneSample]`

### Text encoder / caption pipeline

Responsibilities:

- convert an image to a textual representation
- fit that representation into the byte budget
- reconstruct from stored text only

Interface:

- `encode(sample, budget_bytes) -> EncodedArtifact`
- `decode(artifact) -> Image`

### Visual latent encoder / decoder

Responsibilities:

- map image to a compact visual representation
- quantize or compress to fit the same byte budget
- decode to a reconstruction

Interface matches the text codec.

### Hybrid encoder / decoder

Responsibilities:

- split the total budget between semantic text and visual residual
- decode with both sources

Interface matches the other codecs.

### Byte-budget controller

Responsibilities:

- define allowed stored bytes
- reject or downscale representations that overflow
- record exact bytes used

### Reconstruction pipeline

Responsibilities:

- run all methods across all budgets
- save reconstructions
- log per-scene artifacts and metrics

### Metric evaluator

Responsibilities:

- compute semantic recall
- compute scene fidelity
- compute PSNR / SSIM
- allow optional LPIPS / DINO / CLIP adapters later

### VLM-based scene QA evaluator

Responsibilities:

- generate objective grounded questions
- answer them on original and reconstructed images
- compare answers automatically
- save QA pairs for reproducibility

First baseline note:

- the interface exists now
- the runnable local implementation uses deterministic scene-graph QA on synthetic scenes
- a real VLM adapter is the next swap-in for natural images

## Exact definition of stored bytes

The comparison hinges on byte accounting, so the definition must be explicit.

### Shared rule

Stored bytes are only the bytes required to persist the representation for one scene.

Do include:

- UTF-8 text bytes
- latent payload bytes
- quantized code bytes
- per-scene metadata required to decode that specific sample

Do not include:

- model weights
- tokenizer vocabulary
- shared prompt templates
- code
- runtime memory
- reconstruction inference cost

### Text-only

Stored bytes:

- exact UTF-8 byte length of the saved text representation

Examples:

- caption
- structured scene description
- JSON if the JSON itself is stored

### Visual-latent-only

Stored bytes:

- serialized latent/code payload
- per-scene shape/dtype headers required for decoding

Do not count:

- decoder weights
- shared quantizer codebook if fixed globally

If a learned codebook is scene-specific, count it. If it is a fixed pretrained component, do not.

### Hybrid

Stored bytes:

- text bytes
- latent/residual bytes
- any per-scene split metadata

Constraint:

- total bytes across both parts must not exceed the same budget used by the other methods

## Realistic model choices for the first natural-image implementation

### Text-only

- captions / structured descriptions:
  - `Qwen2.5-VL`
  - `LLaVA-NeXT`
  - `Idefics2`
  - `InternVL`
- reconstruction:
  - `Stable Diffusion XL`
  - `FLUX` variants if available on the GPU environment

### Visual-latent-only

- representation encoders / autoencoders:
  - `SDXL VAE` as a weak starting point
  - `Representation Autoencoder (RAE)` style public checkpoints if available
  - tokenizer-based image models such as `TiTok` or `MAGVIT` style public releases

### Hybrid

- caption + low-rate latent residual
- caption + quantized spatial tokens
- caption + small image patch grid

### Evaluation backbones

- semantic similarity: `SigLIP`
- feature fidelity: `DINOv2`
- perceptual: `LPIPS`
- optional dataset-level: `FID`

## What runs locally vs may require APIs

Can run fully locally:

- synthetic dataset baseline
- byte accounting
- plotting
- PSNR / SSIM
- small CPU smoke tests
- config validation
- artifact serialization checks

Should run on remote GPU or APIs for the intended natural-image benchmark:

- strong VLM caption generation
- diffusion reconstruction
- learned visual latent encode/decode at useful throughput
- strong QA evaluation
- DINO / SigLIP at moderate scale

The framework separates model adapters so local and API-backed implementations can coexist.

## Likely technical risks

- equal-byte comparisons can become unfair if hidden side information leaks into one method
- caption truncation under very small budgets may collapse semantics
- diffusion reconstruction may add plausible but false scene details
- natural-image QA quality may depend heavily on the evaluator VLM
- low-budget latent decoders may preserve texture but lose instance identity
- hybrid methods need a principled budget split

## Assumptions in the first runnable baseline

- use synthetic static scenes to validate the full experiment loop
- treat scene-graph metadata as ground truth semantics
- use simple deterministic reconstructions rather than large generative models
- report PSNR / SSIM now and leave LPIPS / DINO adapters as upgrade points

## Why synthetic scenes first

Synthetic scenes make the evaluation exact:

- object counts are known
- colors are known
- spatial relations are known
- scene fidelity can be computed without a noisy evaluator

This is the fastest way to verify that the stored-byte accounting, metrics, and plotting are correct before scaling to natural images and pretrained models.
