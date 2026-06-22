# ── Base image: CUDA 12.3.1 + cuDNN 8.9 (required by TensorFlow 2.16) ─────────
# The nvidia/cuda image provides the GPU driver bridge so that TensorFlow can
# see and use the host GPU when the NVIDIA Container Toolkit is installed.
# Swap to the -devel variant only if you need nvcc for custom CUDA ops.
FROM nvidia/cuda:12.3.1-cudnn8-runtime-ubuntu22.04

# ── Avoid interactive prompts during apt installs ───────────────────────────
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ── System deps ─────────────────────────────────────────────────────────────
# python3.11 + pip are not in the CUDA base image, so we add them here.
# libgl1 / libglib2.0-0 are needed by OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python / python3
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip     pip     /usr/bin/pip3       1

# ── TensorFlow GPU memory growth (overridable via env) ──────────────────────
# TF_FORCE_GPU_ALLOW_GROWTH prevents TF from allocating the entire VRAM on
# startup, which allows other processes to share the GPU.
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# ── Python deps ─────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── App code ─────────────────────────────────────────────────────────────────
COPY ./app ./app
COPY ./tests ./tests
COPY ./pyproject.toml .
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
