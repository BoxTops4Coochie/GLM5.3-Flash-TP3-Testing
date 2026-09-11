#!/usr/bin/env bash
set -euo pipefail
# Run the unchanged focused tests with their one required config fixture.
# The image lacks dependencies of vLLM's unrelated global integration harness.
mkdir -p /tmp/tp3-tests
ln -s /opt/venv /tmp/.venv
cp /port/vllm/tests/models/test_glm53_tp3_model.py /tmp/tp3-tests/
cp /port/vllm/tests/models/test_glm5next_vision_tp3.py /tmp/tp3-tests/
cp /port/vllm/tests/config/test_glm53_tp3_geometry.py /tmp/tp3-tests/
cp /port/b12x/tests/policy/test_glm53_tp3_port.py /tmp/tp3-tests/
cp /port/cpu-conftest.py /tmp/tp3-tests/conftest.py
cd /tmp
.venv/bin/python -m pytest -p no:cacheprovider tp3-tests -q
