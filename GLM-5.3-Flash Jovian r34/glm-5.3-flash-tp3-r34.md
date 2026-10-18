# GLM-5.3-Flash — R34 TP3

GLM-5.3-Flash NVFP4 on **three NVIDIA RTX PRO 6000 Blackwell Workstation Edition
GPUs (96 GB each)**, with MTP3, FP8 KV and a 1,048,576-token maximum context.
This is a locally tested TP3 derivative of the
[upstream R34 build](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash.md),
carrying the R30 TP3 compatibility changes forward.

## Status

| Setting | Value |
| --- | --- |
| TP / EP / DCP | 3 / 3 / 3 (Compose `DCP`; 1 supported) |
| Default serving | MTP depth 3 |
| Other smoke-tested modes | No speculation; DFlash2 depth 7 with the retained incoai BF16 draft |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4` |
| Target revision | `46aaae8a82032f77100f2f03e9cc11b391df3b4d` |
| Context / request slots | 1,048,576 / 8; capacity depends on request lengths |
| Sampling | Temperature 1, top_p .95, repetition penalty 1; normal EOS |
| Historical thinking | `clear_thinking=true` by default; request-overridable |
| KV / graphs | FP8 GPU-only cache / FULL_AND_PIECEWISE |
| Batch / maximum graph size | 4096 / 32; configurable in Compose |

## Docker artifact


Published image: `azallaza/glm53-r34-tp3:20260910`

Local build: `glm53-r34-tp3:dcp-20260910` (image ID `sha256:34d5ad37b8bb…`),
31 overlay files, overlay identity `7447b24601a4ad8a866790479b99fb3e95415ef4037111b176fa3ba95e4b155c`.
one source file modified for DCP,
`vllm/v1/attention/ops/dcp.py`: the DCP log-sum-exp combine kernel used the
DCP world size directly as a Triton `arange` extent, which must be a power of
two, so world size 3 could not start. The kernel now masks the extra lanes;
it is inert at DCP=1. The TP3 launcher accepts `DCP=1` or `DCP=3`.

## Runtime backends

| Operation | Backend |
| --- | --- |
| Target attention and dense projections | B12X |
| Recurrent prefill / decode | FlashKDA / B12X |
| Routed experts | EP3-capable automatic selection (FlashInfer CUTLASS NVFP4) |
| TP collectives | B12X PCIe |
| MTP vocabulary head | Private NVFP4 draft copy; BF16 target verifier |

`MOE_BACKEND=auto` is intentional for EP3. Native libraries remain from R34.
The port includes TP3 geometry/loading, three-rank collectives, the qualified
GDN profile component, and the local explicit-no-tool parser fix. Detailed
changes and qualification receipts are kept separately from this post.

## Validation

- 3,191 component tests passed, including TP3 loading, parser replay, allocator
  regressions and three-GPU collective/boundary checks.
- All three serving modes passed exact-answer smokes. MTP3 also passed four
  concurrent requests, exact image OCR and 17,642-token retrieval.
- Retained-history test: **826,266 prompt / 28,883 completion tokens**,
  normal stop, nonempty final answer, no tool calls or sustained degeneration
  in full-channel screens and sampled prose review.
- Exact retrieval: **127,992 prompt tokens**, all three markers correct;
  the answer used a Markdown JSON fence rather than bare JSON.
- Exact retrieval: **899,994 prompt tokens**, all three markers correct;
  the answer again used a Markdown JSON fence.
- DCP3 (the Compose default) on the `dcp-20260910` build: exact-answer smokes,
  four concurrent requests, OCR and 17,642-token retrieval; exact retrieval at
  **127,992** and **899,994** prompt tokens (all markers correct, fenced JSON
  again); retained-history test **826,266 prompt / 27,592 completion tokens**,
  normal stop, clean content channel. Its reasoning channel re-quoted one user
  message repeatedly inside a chronological recap before a normal outline; not
  a token loop, but noted.
- DCP=1 on the same build: exact-answer smokes passed after switching with
  `DCP=1 docker compose up -d` (77–79 verifier steps/s at 0 context in the
  table below).

The 128K/826K/900K tests used top_p .95, min_tokens=0 and EOS enabled. No continuation
was forced. These are bounded checks, not a guarantee against all model errors
or the earlier retained-history/response-channel issues. Prose factual accuracy
was not audited. No R34 throughput benchmark is claimed.

Representative MTP3 startup values (image `glm53-r34-tp3:dcp-20260910`,
Compose default `DCP=3`, graph maximum 32):

```text
TP=3 EP=3 DCP=3
collective_backend=b12x_pcie_oneshot
kda_decode_backend=b12x
kda_prefill_backend=flashkda
mm_encoder_tp_mode=weights
GPU KV cache size: 7,510,219 tokens
Draft-only NVFP4 vocabulary head: 113.48 MiB/rank
```

With `DCP=1` (same image, graph maximum 32):

```text
TP=3 EP=3 DCP=1
collective_backend=b12x_pcie_oneshot
kda_decode_backend=b12x
kda_prefill_backend=flashkda
mm_encoder_tp_mode=weights
GPU KV cache size: 2,091,238 tokens
Draft-only NVFP4 vocabulary head: 113.48 MiB/rank
```


## DCP3 versus DCP1

Same image, MTP3, temperature 1 / top_p .95, C1 sustained decode, 30 s cells,
GPUs at 300 W. Verifier steps/s is execution speed; output tok/s includes
stochastic MTP acceptance. DCP1 and DCP3 sweeps were run back to back (two
adjacent pairs agree within 0.3 points).

| | DCP1 | DCP3 |
| --- | ---: | ---: |
| GPU KV capacity (graph max 32) | 2,091,238 tokens | 7,510,219 tokens |
| Concurrent 1,048,576-token requests | 1.99 | 7.16 |
| Verifier steps/s at 0 context | 77.6 | 71.6 (−7.7%) |
| Verifier steps/s at 128K | 74.8 | 70.4 (−5.9%) |
| Verifier steps/s at 256K | 73.2 | 69.9 (−4.6%) |
| Verifier steps/s at 512K | 70.2 | 69.4 (−1.1%) |
| Output tok/s at 0 / 512K | 198 / 169 | 183 / 180 |

DCP3 costs about 8% of decode speed below 128K context and about 1% at 512K
(crossover near 1M tokens) in exchange for 3.6× the KV capacity. DCP is a
start-time setting: `DCP=1 docker compose up -d` returns the full DCP1 speed.

## Sustained decode benchmark at 350 W

[llm-inference-bench](https://github.com/aabduh/llm-inference-bench) 0.6.2,
MTP3, temperature 1 / top_p .95, concurrency 1 and 8, 30 s measured cells
after a 3 s decode warmup, 8,192-token completion cap, same image and Compose
defaults (half-L2 budgets, graph maximum 32). GPUs were set to **350 W** for
these runs only (peak 90 C; the qualification runs above used 300 W). Cells
give aggregate output tok/s with verifier steps/s in parentheses. "skipped
(KV)" means the eight concurrent requests did not fit the KV cache.

| Context | C1 DCP3 | C1 DCP1 | C8 DCP3 | C8 DCP1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 178.3 (72.9) | 194.1 (79.0) | 603.2 (240.7) | 628.3 (252.4) |
| 8K | 186.0 (72.7) | 207.3 (78.9) | 600.6 (239.9) | 636.8 (253.1) |
| 16K | 178.5 (72.7) | 196.2 (78.7) | 599.0 (237.5) | 638.3 (253.3) |
| 32K | 181.3 (72.4) | 200.7 (78.4) | 587.3 (236.3) | 626.6 (250.6) |
| 64K | 183.1 (72.5) | 199.3 (78.3) | 601.4 (236.7) | 618.1 (247.5) |
| 128K | 181.0 (72.2) | 199.9 (77.5) | 593.4 (235.6) | 621.3 (245.0) |
| 256K | 182.8 (72.0) | 192.6 (76.3) | 577.5 (232.5) | 587.5 (236.5) |
| 512K | 174.5 (71.3) | 192.7 (73.8) | 575.1 (226.6) | skipped (KV) |

At C1, DCP1 is 7–8% faster below 256K and 3% faster at 512K. At C8, DCP1 is
about 4% faster where both fit, but only DCP3 can serve eight concurrent
512K-token requests (4.2M tokens of KV against DCP1's 2.09M). Prefill at C1
was 9.5–10.7K tok/s in both modes. Single sweeps, not confidence intervals.

## Example Compose

Fill in the published image (or set `GLM53_IMAGE`) before starting. Model
weights and runtime caches use named volumes; no source mounts are required.
GPU power limits are managed on the host; these checks used 300 W/GPU. The
file is the same layout as the local operational Compose: `MODE` switches
between MTP3 (default), DFlash2 depth 7 and no speculation, and `DCP`,
`MAX_NUM_BATCHED_TOKENS`, `MAX_CUDAGRAPH_CAPTURE_SIZE` and
`CUDAGRAPH_CAPTURE_SIZES` are plain environment overrides. Changing any of
them recreates the server on `docker compose up -d`.

```yaml
services:
  glm53-tp3:
    image: "${GLM53_IMAGE:-}" # Fill in the published image, or set GLM53_IMAGE.
    container_name: glm53-r34-tp3

    restart: "no"
    init: true

    ipc: host
    shm_size: "64gb"

    ports:
      - "${GLM53_PORT:-15015}:8000"

    volumes:
      # Hugging Face hub cache (checkpoint download / offline reuse).
      - huggingface:/root/.cache/huggingface

      # Persistent JIT / FlashInfer-autotune caches, keyed by the image's
      # source-delta fingerprint. First start of each mode on a new image
      # pays autotune once; afterwards startup is warm.
      - runtime-cache:/cache

    environment:
      # The TP3 launcher pins the model and revision; overrides fail closed.
      MODEL: "local-inference-lab/GLM-5.3-Flash-NVFP4"
      MODEL_REVISION: "46aaae8a82032f77100f2f03e9cc11b391df3b4d"

      # Name clients pass as "model" in API requests. Override per invocation:
      #   SERVED_MODEL_NAME=my-model docker compose up -d
      SERVED_MODEL_NAME: "${SERVED_MODEL_NAME:-GLM-5.3-Flash-TP3}"

      VLLM_NO_USAGE_STATS: "1"

      # Override per invocation without editing this file:
      #   MODE=dflash2 docker compose up -d
      MODE: "${MODE:-mtp}"

      # Scheduling / CUDA graphs; override here, in .env, or before compose.
      MAX_NUM_BATCHED_TOKENS: "${MAX_NUM_BATCHED_TOKENS:-4096}"
      MAX_CUDAGRAPH_CAPTURE_SIZE: "${MAX_CUDAGRAPH_CAPTURE_SIZE:-32}"
      # Empty: adapt the tested capture list to the maximum above.
      # Optional space-separated list, or "none" for vLLM-generated sizes.
      CUDAGRAPH_CAPTURE_SIZES: "${CUDAGRAPH_CAPTURE_SIZES:-}"

      # Decode-context parallelism across the three ranks. 3 shards the
      # attention KV cache across GPUs: ~7.5M-token KV capacity (7 concurrent
      # 1M-token requests) at about -8% decode speed below 128K context and
      # -1% at 512K. 1 restores the DCP1 speed with ~2.1M tokens of KV.
      # Requires a restart:  DCP=1 docker compose up -d
      DCP: "${DCP:-3}"

      # Prefix caching, on by default. Set 0 to force a full prefill on every
      # request:
      #   ENABLE_PREFIX_CACHING=0 docker compose up -d --force-recreate
      ENABLE_PREFIX_CACHING: "${ENABLE_PREFIX_CACHING:-1}"

      # L2 weight-prefetch budgets (decimal MB). These halves of the image
      # defaults measured about +2% decode speed on three RTX PRO 6000 at
      # 300 W and are the configuration all checks in this post ran with.
      VLLM_GLM53_L2_PREFETCH_BUDGET_A_MB: "10"
      VLLM_GLM53_L2_PREFETCH_BUDGET_B_MB: "25"
      VLLM_GLM53_L2_PREFETCH_BUDGET_C_MB: "7.5"
      VLLM_GLM53_L2_PREFETCH_BUDGET_A_MLA_MB: "18"

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0", "1", "2"]
              capabilities: [gpu]

    entrypoint:
      - /bin/bash
      - -lc

    command:
      - |
          set -euo pipefail

          # ============================================================
          # MODE
          #
          # mtp      = MTP depth 3      (default; qualified)
          # dflash2  = DFlash2 depth 7  (retained incoai BF16 draft)
          # mtp0     = no speculation
          # ============================================================

          MODE="$${MODE:-mtp}"

          # ============================================================
          # R34 TP3 SETTINGS
          # ============================================================

          export TP=3
          export DCP="$${DCP:-3}"

          export MAX_MODEL_LEN=$${TP3_MAX_MODEL_LEN:-1048576}
          export MAX_NUM_SEQS=8

          # Tested memory allocation setting.
          export GPU_MEMORY_UTILIZATION=0.95

          export CACHE_MODE=vram

          # ============================================================
          # MODE-SPECIFIC SETTINGS
          # ============================================================

          case "$$MODE" in
            mtp)
              export SPECULATOR=mtp
              export MTP_DEPTH=3
              unset DFLASH_DEPTH || true
              ;;

            mtp0)
              export SPECULATOR=mtp
              export MTP_DEPTH=0
              unset DFLASH_DEPTH || true
              ;;

            dflash2)
              export SPECULATOR=dflash2
              export DFLASH_DEPTH=7

              # Retained BF16 draft preset, smoke-tested on this TP3 port.
              # The published MXFP8 alternative is not requalified here.
              export DFLASH_MODEL=incoai/GLM-5.3-Flash-DFlash2
              export DFLASH_MODEL_REVISION=dc77ff1c99eeb2df044ee3d4f0094eb033fee410

              unset MTP_DEPTH || true
              ;;

            *)
              echo "Invalid MODE=$$MODE"
              exit 1
              ;;
          esac

          # ============================================================
          # SHOW ACTIVE SETTINGS
          # ============================================================

          echo "============================================================"
          echo "GLM-5.3-Flash R34 TP3 (top_p=0.95)"
          echo "============================================================"
          echo "MODEL=$$MODEL"
          echo "MODEL_REVISION=$$MODEL_REVISION"
          echo "MODE=$$MODE"
          echo "SPECULATOR=$$SPECULATOR"
          echo "MTP_DEPTH=$${MTP_DEPTH:-N/A}"
          echo "DFLASH_DEPTH=$${DFLASH_DEPTH:-N/A}"
          echo "TP=$$TP"
          echo "DCP=$$DCP"
          echo "MAX_MODEL_LEN=$$MAX_MODEL_LEN"
          echo "MAX_NUM_SEQS=$$MAX_NUM_SEQS"
          echo "GPU_MEMORY_UTILIZATION=$$GPU_MEMORY_UTILIZATION"
          echo "CACHE_MODE=$$CACHE_MODE"
          echo "============================================================"

          exec /usr/local/bin/serve-glm53-flash-tp3-r34.sh

volumes:
  huggingface:
  runtime-cache:
```

```bash
docker compose -p glm53-r34-tp3 up -d            # MTP3, DCP3 (defaults)
MODE=dflash2 docker compose -p glm53-r34-tp3 up -d
DCP=1 docker compose -p glm53-r34-tp3 up -d      # single-copy KV, full DCP1 decode speed
```

`DCP` selects decode-context parallelism: `3` shards the attention KV cache
across the three GPUs (about 7.5M tokens of KV, seven concurrent 1,048,576-token
requests) at roughly 8% lower decode speed below 128K context and about 1%
lower at 512K; `DCP=1 docker compose up -d` restores the single-copy KV layout
(about 2.1M tokens) and the full DCP1 decode speed. Both were smoke-tested; the
DCP3 checks include exact three-marker retrieval at 128K and 900K prompt tokens.
`MAX_NUM_BATCHED_TOKENS` and `MAX_CUDAGRAPH_CAPTURE_SIZE` can be changed through
the environment or Compose. The default capture list adapts to the selected
maximum; `CUDAGRAPH_CAPTURE_SIZES` can supply an explicit list.
