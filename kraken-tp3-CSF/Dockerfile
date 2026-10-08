# Kraken TP3 rebased onto upstream karmic-kraken-beta (vLLM 89f1ceecd, 2026-10-07/08)
# + CSF-QAD (fd660d51 and dec48abd quantized-vision) at TP3. Release image 20261008.
ARG BASE_IMAGE=ghcr.io/local-inference-lab/vllm@sha256:629d097ce4f3ac2ceea86bffb174fec94f5c7fe9bb0a6dbc70e9a05e30b9e54a
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
