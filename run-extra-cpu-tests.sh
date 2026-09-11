#!/usr/bin/env bash
set -euo pipefail
ln -s /opt/venv /tmp/.venv
mkdir -p /tmp/extra
cp /port/vllm/tests/v1/core/test_mamba_sparse_cleanup.py /tmp/extra/
cp /port/cpu-conftest.py /tmp/extra/conftest.py
cd /tmp
.venv/bin/python -m pytest -p no:cacheprovider extra -q
.venv/bin/python -m pytest -p no:cacheprovider /port/b12x/tests/comm/test_pcie_dma_kernels.py /port/b12x/tests/comm/test_pcie_hierarchical.py -q
