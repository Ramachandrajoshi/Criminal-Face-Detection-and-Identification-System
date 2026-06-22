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

# ── TensorFlow GPU memory growth (overridable via env) ────────────────────────
# TF_FORCE_GPU_ALLOW_GROWTH=true prevents TensorFlow from pre-allocating the
# entire GPU VRAM on startup, allowing other processes to share the card.
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

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

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
