ARG BASE_IMAGE=glm53-kraken-tp3:20260925
FROM ${BASE_IMAGE}
COPY src/ /opt/venv/lib/python3.12/site-packages/
COPY serve-glm53-flash-tp3-kraken.py /usr/local/bin/serve-glm53-flash-tp3-kraken.py
COPY support-receipt.json /opt/lil/glm53-20260927.json
ENV VLLM_B12X_BF16_GEMV=1 VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS=0
