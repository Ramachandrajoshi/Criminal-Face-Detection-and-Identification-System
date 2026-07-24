# ── Base image ────────────────────────────────────────────────────────────────
# tensorflow/tensorflow:2.16.1-gpu is the official Google image that bundles:
#   • CUDA 12.3  +  cuDNN 8.9   (the versions TensorFlow 2.16 requires)
#   • Python 3.11  +  pip       (no manual install needed)
#   • Ubuntu 22.04 LTS
#
# This image tag is confirmed to exist on Docker Hub.  It provides the GPU
# driver bridge when the NVIDIA Container Toolkit is installed on the host.
#
# To use CPU-only (no GPU), switch to:
#   FROM tensorflow/tensorflow:2.16.1
FROM tensorflow/tensorflow:2.16.1-gpu

# ── Avoid interactive prompts during apt installs ─────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
# libgl1 and libglib2.0-0 are required by OpenCV.
# The TF base image already has Python 3.11, pip, and CUDA; we only add
# the OpenCV system libraries here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Remove system Python packages that conflict with pip ─────────────────────
# The TF base image pre-installs several packages via apt/distutils
# (blinker, Flask, Werkzeug, …).  pip cannot uninstall distutils-managed
# packages, so it fails with "Cannot uninstall blinker 1.4".
# We purge them here so pip can install the correct versions cleanly.
RUN apt-get update && apt-get purge -y \
    python3-blinker \
    python3-flask \
    python3-werkzeug \
    python3-click \
    python3-itsdangerous \
    python3-jinja2 \
    2>/dev/null || true \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ── TensorFlow GPU memory growth (overridable via env) ────────────────────────
# TF_FORCE_GPU_ALLOW_GROWTH=true prevents TensorFlow from pre-allocating the
# entire GPU VRAM on startup, allowing other processes to share the card.
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# ── CUDA / cuDNN library paths ────────────────────────────────────────────────
# The dynamic linker must be able to find libcudnn.so.8, libcublas.so.12, etc.
# These are the standard paths for the tensorflow/tensorflow:2.16.1-gpu image.
# Without this, TF silently fails to dlopen the GPU libs and falls back to CPU.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda-12.3/lib64:${LD_LIBRARY_PATH}

# Suppress verbose TF/XLA C++ info logs (keep warnings and errors).
# 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR
ENV TF_CPP_MIN_LOG_LEVEL=1

# ── Python deps ───────────────────────────────────────────────────────────────
# tensorflow>=2.16.0 is already installed in the base image; pip will skip it
# or upgrade in-place if requirements.txt specifies a newer patch.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── App code ──────────────────────────────────────────────────────────────────
COPY ./app ./app
COPY ./tests ./tests
COPY ./pyproject.toml .
ENV PYTHONPATH=/app

EXPOSE 8000

# Production default: single worker — the GPU is shared (see docker-compose.yml)
# and each additional worker process would load its own copy of the
# RetinaFace/ArcFace weights into VRAM; concurrency instead comes from the
# in-process inference thread pool (app/core/inference_executor.py). No
# --reload (dev-only; overridden locally via docker-compose.override.yml).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
