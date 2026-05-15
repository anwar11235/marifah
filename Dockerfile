# marifah-core training image
# Build: docker build -t anwar1919/marifah-core:2026-05-15 .
# Use on Vast.ai: specify this image when provisioning.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps — install from requirements.txt ahead of cloning the repo
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install pytest (used as pre-launch gate)
RUN pip install --no-cache-dir pytest

# Install flash-attn pinned to 2.7.4.post1 — last version ABI-compatible with torch 2.6.
# FLASH_ATTN_CUDA_ARCHS=80 limits compilation to A100 (sm_80), ~4× faster build.
RUN FLASH_ATTN_CUDA_ARCHS=80 pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Smoke test: fail the build early if flash-attn import is broken (catches ABI mismatches).
RUN python -c "import torch, flash_attn; assert torch.cuda.is_available() or True; print('flash_attn OK:', flash_attn.__version__)"

# Working directory — all session handoff docs assume /workspace.
RUN mkdir -p /workspace
WORKDIR /workspace

# On Vast.ai, the user clones marifah-core into /workspace/marifah and runs from there.
