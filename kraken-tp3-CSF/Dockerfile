# Kraken TP3 rebased onto upstream karmic-kraken-beta-20261006-c8c8feeaa8cc4a75
# (vLLM 23f2a1830) + CHECKPOINT=csf-qad (FP4-CSF experts padded 2048 -> 2112
# inside the CSF reader for MOE_TP). Release image 20261007.
ARG BASE_IMAGE=ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261006-c8c8feeaa8cc4a75
FROM ${BASE_IMAGE}
COPY src/ /opt/venv/lib/python3.12/site-packages/
COPY serve-glm53-flash-tp3-kraken.py /usr/local/bin/serve-glm53-flash-tp3-kraken.py
COPY glm53-20260927.json glm53-decode-options.json glm53-optional-eplb.json glm53-tp-experts.json tp3-source-delta.json /opt/lil/
COPY lil/cache.py /opt/lil/runtime/cache.py
COPY lil/image-contract.json /opt/lil/runtime/image-contract.json
RUN chmod +x /usr/local/bin/serve-glm53-flash-tp3-kraken.py && \
    /opt/venv/bin/python -m compileall -q /opt/venv/lib/python3.12/site-packages/vllm /opt/venv/lib/python3.12/site-packages/b12x/comm/pcie >/dev/null
ENV MOE_TP=0 VLLM_B12X_BF16_GEMV=1 VLLM_GLM53_KDA_NO_COPY=0 VLLM_GLM53_TP3_MOE_TP=0 \
    VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS=0 VLLM_TP3_PCIE_DCP=0 VLLM_USE_FASTOKENS=1
