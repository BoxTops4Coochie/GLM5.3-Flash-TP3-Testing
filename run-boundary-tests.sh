#!/usr/bin/env bash
set -euo pipefail
ln -s /opt/venv /tmp/.venv
export TORCHINDUCTOR_CACHE_DIR=/tmp/tp3/inductor
export TRITON_CACHE_DIR=/tmp/tp3/triton
export XDG_CACHE_HOME=/tmp/tp3
cp /port/vllm/tests/v1/worker/test_gpu_boundary_checkpoint.py /tmp/test_gpu_boundary_checkpoint.py
cd /tmp
.venv/bin/python -m pytest -p no:cacheprovider test_gpu_boundary_checkpoint.py -q
