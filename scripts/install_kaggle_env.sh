#!/usr/bin/env bash
set -euo pipefail

TORCH_VERSION="2.3.1"
TORCHVISION_VERSION="0.18.1"
TORCHAUDIO_VERSION="2.3.1"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"

python3 -m pip uninstall -y torch torchvision torchaudio || true
python3 -m pip install --no-cache-dir \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

# Project dependencies deliberately exclude torch-family packages so they cannot
# silently replace the pinned P100-compatible CUDA wheel.
python3 -m pip install --no-cache-dir -r requirements-kaggle.txt
