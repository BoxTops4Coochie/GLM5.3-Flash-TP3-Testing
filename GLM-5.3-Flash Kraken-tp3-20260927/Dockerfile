ARG BASE_IMAGE=glm53-kraken-tp3:20260925
FROM ${BASE_IMAGE}
# fastokens 0.3.2 (upstream hash-pinned wheel, docker #109) + vllm #934 fix in src/vllm/tokenizers/hf.py.
COPY fastokens-requirements.txt /tmp/fastokens-requirements.txt
RUN /opt/venv/bin/python -m pip install --no-deps --require-hashes -r /tmp/fastokens-requirements.txt && rm /tmp/fastokens-requirements.txt
COPY src/ /opt/venv/lib/python3.12/site-packages/
COPY serve-glm53-flash-tp3-kraken.py /usr/local/bin/serve-glm53-flash-tp3-kraken.py
COPY support-receipt.json /opt/lil/glm53-20260927.json
# lil external cache: allow GLM TP3 (LMCache L1); contract hash updated for cache.py only.
COPY lil/cache.py /opt/lil/runtime/cache.py
COPY lil/image-contract.json /opt/lil/runtime/image-contract.json
ENV VLLM_B12X_BF16_GEMV=1 VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS=0 VLLM_USE_FASTOKENS=1
# lil-bench: upstream installer and lock (llm-inference-bench v0.7.3), p2pmark built for this image.
COPY install_lil_bench.py lil-bench.lock.json /tmp/lil-bench/
RUN /opt/venv/bin/python /tmp/lil-bench/install_lil_bench.py --lock /tmp/lil-bench/lil-bench.lock.json && rm -rf /tmp/lil-bench
