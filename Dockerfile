# Clean-environment image for REP-01 (CPU). Build from the exported committed tree:
#   docker build -t lipid-v12 .
#   docker run --rm -v "$PWD:/work" -w /work lipid-v12 bash scripts/reproduce_revision.sh
# GPU reruns: replace the base image with an official pytorch/pytorch CUDA runtime image and pass --gpus all.
FROM python:3.10-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" && \
    python -m pip install -r /tmp/requirements.txt
