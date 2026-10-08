#!/bin/bash
# dflash_check.sh LABEL ENV...: boot the release compose with DFlash2, check 8/8, bench 32K C1.
R=/home/aabduh/glm53-tp3-patch-guide/kraken-port/release/kraken-20261008; label=$1; shift
cd $R && /usr/bin/env "$@" HF_CACHE_DIR=/m2-2/huggingface GLM53_IMAGE=glm53-kraken-tp3:20261008 docker compose -p kraken-port -f compose.yaml up -d --force-recreate </dev/null >/tmp/cu.log 2>&1
for i in $(seq 1 90); do sleep 10; [ "$(docker inspect -f '{{.State.Running}}' glm53-kraken-tp3)" != true ] && { echo "$label EXITED"; docker logs glm53-kraken-tp3 2>&1 | grep -iE "Error:" | grep -v triton_kernels | tail -2; exit 1; }; curl -sf localhost:15015/v1/models >/dev/null && break; done
docker logs glm53-kraken-tp3 > $R/dflash-$label-server.log 2>&1
echo "$label: $(grep -oE 'GPU KV cache size: [0-9,]+ tokens' $R/dflash-$label-server.log | tail -1) | $(grep -oE 'Resolved architecture: DFlash2DraftModel' $R/dflash-$label-server.log | head -1)"
python3 - <<'PY'
import json,urllib.request
ok=0
for i in range(8):
    b=dict(model='GLM-5.3-Flash-TP3',max_tokens=512,temperature=0,top_p=0.95,seed=i+17,messages=[dict(role='user',content=f'What is {i+10} plus 2? Reply with only the number.')])
    c=json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:15015/v1/chat/completions',json.dumps(b).encode(),{'Content-Type':'application/json'}),timeout=300))['choices'][0]
    ok+=c['message']['content'].strip()==str(i+12) and c['finish_reason']=='stop'
print('  arith',ok,'/8')
PY
for i in $(seq 1 60); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1); [ "$t" -lt 58 ] && break; sleep 10; done
echo n | /home/aabduh/llm-inference-bench/.venv/bin/python /home/aabduh/llm-inference-bench/llm_decode_bench.py --host localhost --port 15015 --model GLM-5.3-Flash-TP3 --concurrency 1 --contexts 32k --duration 30 --decode-warmup-seconds 15 --max-tokens 24576 --temperature 1 --display-mode plain --dcp-size 1 --skip-prefill --output $R/dflash-$label-c1.json > $R/dflash-$label-c1.log 2>&1
echo "  C1 32K tok/s $(grep -A6 '^Aggregate decode tok/s' $R/dflash-$label-c1.log | grep '│ [0-9]' | head -1) | steps (accept) $(grep -A7 'steps/s' $R/dflash-$label-c1.log | grep '│ [0-9]' | head -1)"
