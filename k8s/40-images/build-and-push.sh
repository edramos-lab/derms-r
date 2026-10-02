#!/usr/bin/env bash
# Build both training images and push to a registry every cluster node can
# pull from. Run from the repo root:
#   REGISTRY=ghcr.io/you k8s/40-images/build-and-push.sh
set -euo pipefail

REGISTRY="${REGISTRY:?Set REGISTRY, e.g. REGISTRY=ghcr.io/you or REGISTRY=localhost:5000}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"

echo "== Building Ada (RTX 5000 Ada, sm_89) image from the repo's root Dockerfile =="
docker build -f Dockerfile -t "${REGISTRY}/derms-r-train:ada-${TAG}" .
docker tag "${REGISTRY}/derms-r-train:ada-${TAG}" "${REGISTRY}/derms-r-train:ada"

echo "== Building Blackwell (RTX 5090, sm_120) image =="
docker build -f k8s/40-images/Dockerfile.blackwell -t "${REGISTRY}/derms-r-train:blackwell-${TAG}" .
docker tag "${REGISTRY}/derms-r-train:blackwell-${TAG}" "${REGISTRY}/derms-r-train:blackwell"

echo "== Pushing =="
docker push "${REGISTRY}/derms-r-train:ada-${TAG}"
docker push "${REGISTRY}/derms-r-train:ada"
docker push "${REGISTRY}/derms-r-train:blackwell-${TAG}"
docker push "${REGISTRY}/derms-r-train:blackwell"

cat <<EOF

Done. Set REGISTRY=${REGISTRY} when running generate_manifests.py
(../60-jobs/generate_manifests.py) so the generated Job specs reference
${REGISTRY}/derms-r-train:ada and ${REGISTRY}/derms-r-train:blackwell.

Sanity-check each image on its real node before generating the full job
matrix (this is the check from the Dockerfile.blackwell header comment):
  docker run --rm --gpus all ${REGISTRY}/derms-r-train:ada \\
    python -c "import torch; print(torch.cuda.get_device_capability(0))"
  docker run --rm --gpus all ${REGISTRY}/derms-r-train:blackwell \\
    python -c "import torch; print(torch.cuda.get_device_capability(0))"
EOF
