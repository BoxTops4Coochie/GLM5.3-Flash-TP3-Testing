# Kraken TP3 release image 20261015 = release 20261008 (sha256:44c2d27ec026, vLLM 89f1ceecd + our TP3
# port + CSF-QAD) + small-row MXFP8 GEMV for decode (<= 8 rows) + launcher with 32-slot KV budgets
# and the finer >32-row graph ladder. Only these files change; all 20261008 layers are reused.
ARG BASE_IMAGE=azallaza/glm53-kraken-tp3:20261008
FROM ${BASE_IMAGE}
COPY src/ /opt/venv/lib/python3.12/site-packages/
COPY serve-glm53-flash-tp3-kraken.py /usr/local/bin/serve-glm53-flash-tp3-kraken.py
RUN chmod +x /usr/local/bin/serve-glm53-flash-tp3-kraken.py && \
    /opt/venv/bin/python -m compileall -q /opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear >/dev/null
ENV VLLM_GLM53_MXFP8_GEMV=1 VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=1MB
