# DeepSeek-V4.1-Flash-1017 at TP3 on 3x RTX PRO 6000 Blackwell (sm_120).
#
# Base: Local Inference Lab karmic-kraken-beta-20261016.2 (vLLM 6d66eed35,
# FlashInfer 0.7.1 at 79819b11). b12x is replaced by FlashInfer main at
# FLASHINFER_COMMIT (Python only: the b12x kernels compile at startup), and the
# files in src/ add the TP3 expert placement, Engram host-RAM options and the
# compatibility shims listed in source.patch.
FROM ghcr.io/local-inference-lab/vllm@sha256:f5f9efe1b10ec7b853a495e66b7dbf15ff54f62741e444dea0b44e16736a10a9

ARG FLASHINFER_COMMIT=5b065abe0ce787364d26f01d753c7b73d2c0d57e
ADD https://github.com/local-inference-lab/flashinfer/archive/${FLASHINFER_COMMIT}.tar.gz /tmp/flashinfer.tar.gz
RUN set -e; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    mkdir /tmp/fi; tar -xzf /tmp/flashinfer.tar.gz -C /tmp/fi --strip-components=1; \
    rm -rf "$SP/flashinfer/experimental/b12x"; \
    cp -r /tmp/fi/flashinfer/. "$SP/flashinfer/"; \
    cp -r /tmp/fi/csrc/. "$SP/flashinfer/data/csrc/"; \
    cp -r /tmp/fi/include/. "$SP/flashinfer/data/include/"; \
    rm -rf /tmp/fi /tmp/flashinfer.tar.gz

COPY src/ /opt/venv/lib/python3.12/site-packages/
COPY ds41-launch.yaml /etc/ds41/launch.yaml
COPY expert-order/ /opt/ds41/expert-order/
COPY ds41-serve /usr/local/bin/ds41-serve
RUN chmod +x /usr/local/bin/ds41-serve

# Defaults of the measured configuration; compose.yaml documents each one.
ENV MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    PORT=15015 \
    KV_CACHE_MEMORY_BYTES=2415919104 \
    CPU_OFFLOAD_GB=8 \
    DS41_OFFLOAD_FROM_ORDINAL=20 \
    DS41_OFFLOAD_TO_ORDINAL=25 \
    DS41_VMM_ENABLE=1 \
    DS41_VMM_COLD_FRACTION=1 \
    DS41_VMM_CHUNK_EXPERTS=16 \
    B12X_EXPERT_ORDER_DIR=/opt/ds41/expert-order \
    B12X_DEFERRED_INPLACE=1 \
    B12X_DEFERRED_WEIGHT_BANKS=0 \
    B12X_DEFERRED_BANK_RESERVE_BYTES=1073741824 \
    B12X_DEFERRED_BANK_ARM_FILE=/cache/deferred-banks.arm \
    B12X_COMPILE_CACHE_DIR=/cache/b12x-compile \
    B12X_SCALE_CODEC=0 \
    DS41_B12X_CSF=1 \
    B12X_CODEC_PIN_TUNING=1 \
    B12X_CODEC_PIN_COMPONENTS=all \
    DS41_ROPE_CACHE_ROWS=1048576 \
    DS41_EMBEDDING_UVA=1 \
    DS41_INDEX_CHUNK=128 \
    DS41_ENGRAM_RAM_LAYERS=1 \
    DS41_ENGRAM_ROW_CACHE_ROWS=8388608 \
    VLLM_SHARE_PYNCCL_COMMS=1 \
    VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=256KB \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_LEVEL=SYS \
    HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1

ENTRYPOINT ["/usr/local/bin/ds41-serve"]
CMD []
