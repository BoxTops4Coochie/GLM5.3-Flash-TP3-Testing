#!/usr/bin/env bash
set -euo pipefail
ln -s /opt/venv /tmp/.venv
export XDG_CACHE_HOME=/tmp/r27-tp3-comm
export TORCHINDUCTOR_CACHE_DIR=${XDG_CACHE_HOME}/inductor
export CUTE_DSL_CACHE_DIR=${XDG_CACHE_HOME}/cute
export B12X_CUTE_COMPILE_CACHE_DIR=${XDG_CACHE_HOME}/b12x-cute
export B12X_COMPILE_CACHE_DIR=${XDG_CACHE_HOME}/b12x
export SPARKINFER_COMPILE_CACHE_DIR=${XDG_CACHE_HOME}/b12x
export CUDA_CACHE_PATH=${XDG_CACHE_HOME}/cuda
export B12X_RUN_PCIE_DMA_TEST=1 B12X_PCIE_DMA_WORLD_SIZE=3
export B12X_RUN_PCIE_ONESHOT_TORTURE=1 B12X_PCIE_ONESHOT_TORTURE_WORLD_SIZE=3
cd /tmp
.venv/bin/python -m pytest -p no:cacheprovider \
  /port/b12x/tests/comm/test_pcie_dma_gpu.py::test_pcie_dma_all_reduce_eager_and_graph \
  /port/b12x/tests/comm/test_pcie_oneshot_torture.py -q -rs
