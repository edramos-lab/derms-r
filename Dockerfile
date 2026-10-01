# syntax=docker/dockerfile:1
# CUDA-enabled image for prepare_dataset.py / train.py / test.py / analyze.py / export_models.py.
# Works on CPU-only hosts too: just omit `--gpus all` and pass --device cpu to train.py/test.py.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV NO_ALBUMENTATIONS_UPDATE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# opencv-python-headless still dlopens libGL/libglib on some distros despite the "headless" name.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision already match the base image's CUDA build, so this only adds
# the remaining project + ONNX export dependencies on top of it.
COPY requirements.txt requirements_export.txt ./
RUN pip install -r requirements.txt -r requirements_export.txt

COPY . .

# Bind-mount real data/results at `docker run` time; these are placeholders so the
# paths exist even if a mount is omitted (e.g. for --verify-models).
RUN mkdir -p datasets results
VOLUME ["/workspace/datasets", "/workspace/results"]

ENTRYPOINT ["python"]
CMD ["train.py", "--verify-models"]
