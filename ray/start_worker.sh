#!/usr/bin/env bash
# Ray GPU worker, run ON the RTX 5000 Ada or RTX 5090 box, INSIDE the same
# training image used by the Docker/K8s paths (so train.py's environment is
# identical everywhere). Tags itself with a custom Ray resource ("ada" or
# "blackwell") so driver.py can pin each task to the matching GPU family --
# the two CUDA/PyTorch builds are not interchangeable (see
# ../k8s/40-images/Dockerfile.blackwell).
#
# Usage, on the RTX 5000 Ada box:
#   RAY_HEAD_IP=<head-ip> GPU_ARCH=ada ./start_worker.sh
# On the RTX 5090 box:
#   RAY_HEAD_IP=<head-ip> GPU_ARCH=blackwell ./start_worker.sh
set -euo pipefail

RAY_HEAD_IP="${RAY_HEAD_IP:?Set RAY_HEAD_IP to the head node's address}"
RAY_PORT="${RAY_PORT:-6379}"
GPU_ARCH="${GPU_ARCH:?Set GPU_ARCH=ada or GPU_ARCH=blackwell}"
IMAGE="${IMAGE:-derms-r-train:${GPU_ARCH}}" # built by ../k8s/40-images/build-and-push.sh,
                                             # or build locally: docker build -f Dockerfile (ada) / k8s/40-images/Dockerfile.blackwell -t "$IMAGE" .
DATASETS_DIR="${DATASETS_DIR:-$(pwd)/datasets}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$(pwd)/hf-cache}"

if [ "$GPU_ARCH" != "ada" ] && [ "$GPU_ARCH" != "blackwell" ]; then
  echo "GPU_ARCH must be 'ada' or 'blackwell', got: $GPU_ARCH" >&2
  exit 1
fi

docker run -d --name "derms-ray-worker-${GPU_ARCH}" \
  --gpus all \
  --network host \
  --shm-size 2g \
  -v "${DATASETS_DIR}:/workspace/datasets" \
  -v "${RESULTS_DIR}:/workspace/results" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  --entrypoint bash \
  "${IMAGE}" -c "
    pip install -q 'ray[default]>=2.9,<3' &&
    ray start --address=${RAY_HEAD_IP}:${RAY_PORT} \
      --num-gpus=1 \
      --resources='{\"${GPU_ARCH}\": 1}' \
      --block
  "

echo "Started derms-ray-worker-${GPU_ARCH}, tagged with Ray custom resource '${GPU_ARCH}'."
echo "Check it joined: docker logs -f derms-ray-worker-${GPU_ARCH}"
echo "Stop with: docker stop derms-ray-worker-${GPU_ARCH} && docker rm derms-ray-worker-${GPU_ARCH}"
