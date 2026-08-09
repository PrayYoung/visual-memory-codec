# Visual Memory Codec

Efficient persistent representations for machine visual memory.

Small research/engineering prototype for evaluating persistent visual memory representations under a fixed storage budget.

This first milestone is intentionally narrow:

- static images only
- equal-byte comparison across methods
- reproducible local baseline
- modular interfaces for later swap-in of stronger pretrained models

As of Sunday, August 9, 2026, the execution strategy is:

- local CPU:
  - orchestration
  - unit tests / smoke tests
  - byte accounting
  - synthetic regression benchmark
  - plotting and result packaging
- remote GPU:
  - natural-image captioning / VLM evaluation
  - diffusion reconstruction
  - learned visual latent encode/decode
  - DINO / SigLIP-scale natural-image metrics at useful throughput

The included runnable baseline uses a synthetic shapes dataset so the framework can produce:

- reconstructed images
- exact stored-byte counts
- semantic recall
- scene fidelity
- PSNR / SSIM
- Pareto plots

It is designed to validate the experiment loop before and alongside heavier natural-image runs.

## Methods in the baseline

- `text_only`: stores a UTF-8 scene description string and reconstructs by redrawing parsed shapes
- `visual_latent_only`: stores a quantized low-resolution RGB grid under the same byte budget
- `hybrid`: stores a compact text scene summary plus a small visual grid residual

## Natural-image path

The natural-image MVP is now set up as a Colab/GPU path rather than a local-CPU path.

Current repo support:

- COCO subset dataset adapter
- frozen QA generation from COCO annotations
- detector-based QA answering on reconstructions
- real-text and learned-latent codec scaffolding
- Colab notebook bootstrap for GPU execution

Recommended workflow:

1. Run the synthetic benchmark locally for regression.
2. Launch the Colab notebook in [notebooks/visual_memory_codec_colab.ipynb](/Users/peiyan1/Desktop/ai_agent_memory/notebooks/visual_memory_codec_colab.ipynb).
3. Prepare a curated COCO subset in Colab.
4. Run the natural-image config on GPU.
5. Copy outputs back to Google Drive or download them.

## Repository layout

- [docs/architecture.md](/Users/peiyan1/Desktop/ai_agent_memory/docs/architecture.md)
- [configs/synthetic_baseline.json](/Users/peiyan1/Desktop/ai_agent_memory/configs/synthetic_baseline.json)
- [configs/natural_coco_mvp.json](/Users/peiyan1/Desktop/ai_agent_memory/configs/natural_coco_mvp.json)
- [requirements-colab.txt](/Users/peiyan1/Desktop/ai_agent_memory/requirements-colab.txt)
- [notebooks/visual_memory_codec_colab.ipynb](/Users/peiyan1/Desktop/ai_agent_memory/notebooks/visual_memory_codec_colab.ipynb)
- [src/visual_memory_benchmark](/Users/peiyan1/Desktop/ai_agent_memory/src/visual_memory_benchmark)

## Quickstart

```bash
python3 -m visual_memory_benchmark.run --config configs/synthetic_baseline.json
```

If you do not want to install the package, use:

```bash
PYTHONPATH=src python3 -m visual_memory_benchmark.run --config configs/synthetic_baseline.json
```

Outputs are written to `outputs/<run_name>/`.

For Colab/GPU natural-image runs, use the notebook and `requirements-colab.txt`.

## Local Dependencies

The baseline uses only:

- `numpy`
- `Pillow`
- `matplotlib`

Natural-image GPU runs additionally require:

- `torch`
- `torchvision`
- `transformers`
- `diffusers`
- `accelerate`
- `safetensors`

## What this baseline is and is not

This is not yet a finished scientific conclusion. The synthetic benchmark remains a regression path, and the natural-image benchmark is intentionally routed to GPU execution so the scientific design is not distorted by local CPU constraints.
