# GLM-5.3-Flash Uncensored — R34 TP3

`orcarouter/GLM-5.3-Flash-Uncensored-NVFP4` on **three NVIDIA RTX PRO 6000
Blackwell Workstation Edition GPUs (96 GB each)**, with MTP3, FP8 KV and a
1,048,576-token maximum context, served on the same R34 TP3 image as the
released checkpoint.

This is a serving report for a third-party fine-tune: what it takes to run it on
this stack and how it measures. It is not an evaluation of how the fine-tune
differs in behaviour from the base model, and nothing here was checked beyond
the functional and performance tests listed below.

A side-by-side against the released checkpoint — decode, prefill, capacity and
quality in one place — is in
[CHECKPOINT-COMPARISON.md](CHECKPOINT-COMPARISON.md).

## Status

| Setting | Value |
| --- | --- |
| Checkpoint | `orcarouter/GLM-5.3-Flash-Uncensored-NVFP4` (Compose `CHECKPOINT=uncensored`) |
| Revision | `ec0adf4f49c9570807cc11a5f650538c1893ae54` |
| TP / EP / DCP | 3 / 3 / 1; DCP3 supported and qualified |
| Default serving | MTP depth 3 |
| Context / request slots | 1,048,576 / 8; capacity depends on request lengths |
| Sampling | Temperature 1, top_p .95, repetition penalty 1; normal EOS |
| Reasoning effort | `max` by default; `low`/`high`/`max` are the only distinct tiers; request-overridable |
| Historical thinking | `clear_thinking=true` by default; request-overridable |
| KV / graphs | FP8 GPU-only cache / FULL_AND_PIECEWISE |
| Prefix caching | Enabled; required by this split-page configuration and not disableable |
| Batch / maximum graph size | 4096 / 32; configurable in Compose |
| GPU KV capacity | 1,866,544 tokens (DCP1) / 6,900,837 (DCP3) |
| GPU power limit | 350 W per GPU; host setting |

The checkpoint is 191.0 GiB of safetensors and is **not bundled** with this
configuration; download it into the Hugging Face cache yourself.

## Docker artifact

Published image: `azallaza/glm53-r34-tp3:native-dcp-20260913`

```bash
docker pull azallaza/glm53-r34-tp3:native-dcp-20260913
```

No image changes are needed. This checkpoint runs on the same artifact as the
released NVFP4 build; only launcher parameters differ, and the launcher selects
them from `CHECKPOINT`.

**The image alone is not enough.** The TP3 launcher baked into the image is
older than the one this configuration needs (`sha256:17ea8f8a…` in the image
versus `sha256:4b250132…` mounted). The image's copy has no `CHECKPOINT`
handling at all and hard-pins the released checkpoint, so it cannot serve this
model. Download [serve-glm53-flash-tp3-r34.sh](serve-glm53-flash-tp3-r34.sh)
and keep it next to [compose.yaml](compose.yaml), which bind-mounts it over the
image copy.

## Why this checkpoint needs different kernels

The architecture is identical to the released build — `glm5_next` /
`Glm5NextForConditionalGeneration`, 45 layers, 4096 hidden, 64 attention heads,
288 routed experts, vocabulary 154880, MTP layer 45, vision tower — and
`tokenizer.json`, `tokenizer_config.json` and `processor_config.json` are
byte-identical. The chat template differs only cosmetically and exposes the
same three effort tiers.

The difference is the quantization container:

| | Released NVFP4 | This checkpoint |
| --- | --- | --- |
| Scheme | modelopt `MIXED_PRECISION` | compressed-tensors `nvfp4-pack-quantized` |
| Routed experts | NVFP4, static W4A4 | NVFP4, **W4A16** |
| Activation scales | 36,288 calibrated `input_scale` tensors | none |
| MTP layer 45 experts | MXFP8, group 32 | **unquantized BF16** |

An audit of all 111,346 tensors against the released checkpoint found zero shape
mismatches and zero unexpected tensors once `weight_packed` → `weight` and
`weight_global_scale` → `weight_scale_2` are renamed; the packed weights and
E4M3 block scales are the same format and the global scale is the exact
reciprocal. The only gaps are the missing activation scales and the BF16 MTP
layer — which is why the runtime needs weights-only expert kernels rather than
the NVFP4 W4A4 path.

Serving it with the released checkpoint's backends produces **fluent-looking
garbage** (repeated single tokens) while loading cleanly and passing health
checks. The kernel selection below is required, not optional.

## Runtime backends

| Operation | Backend |
| --- | --- |
| Target attention | B12X |
| Dense projections | B12X / cuBLAS, with the CuTe skinny GEMM at measured small-batch shapes |
| Recurrent prefill / decode | FlashKDA / B12X |
| Routed target experts | **Marlin** (weights-only NVFP4) |
| TP collectives | B12X PCIe |
| MTP experts / attention | **Triton** (unquantized layer) / B12X |
| Quantization | `compressed-tensors` |
| Load format | `auto` |

Two runtime properties are not configurable on this stack: prefix caching cannot
be disabled (the GLM-5.3 split target/recurrent-state page layout requires
`mamba_cache_mode=align`, which requires prefix caching), and the KV cache
cannot be unquantized (the B12X backend rewrites `auto`/`fp8`/`fp8_e4m3` to
`fp8_ds_mla` unconditionally). Both are therefore present in every result below.

## Validation

All checks run on this checkpoint at both DCP settings, 350 W, MTP3.

| Check | DCP1 | DCP3 |
| --- | --- | --- |
| C1 exact-answer (3 cases) | pass | pass |
| C8 — eight simultaneous requests | pass | pass |
| Vision / OCR (exact transcription) | pass | pass |
| Retrieval 17.6K | pass | pass |
| Retrieval 128K, three markers | pass | pass |
| Retrieval 900K, three markers | pass | pass |
| Retained history 768K, recall | pass | pass |
| Retained history 768K, long generation | pass, no degeneration | not repeated |
| Tool call | pass | pass |
| Explicit no-tool | pass | pass |

Long-context runs used top_p .95 with EOS enabled and no forced continuation.
The 900K retrieval returned all three markers with a normal stop in 121–135 s.
These are bounded checks, not a guarantee against all model errors; prose
factual accuracy was not audited.

### Answer quality

`llm-inference-bench` `lavd` profile — a 48K-character ledger the model must
keep consistent, scored EXACT / NEAR / FAIL against `72, 46` where NEAR means
both totals fall within ±4. Thirty runs at reasoning effort `max`, concurrency
10, DCP1:

| Metric | Result |
| --- | ---: |
| Requests delivered | 30 / 30 — no errors, timeouts, stalls or truncations |
| Score | EXACT 26 / NEAR 3 / FAIL 1 |
| Credited (EXACT + NEAR) | 96.7% (29 / 30) |
| Median completion tokens | 12,831 |

"Requests delivered" counts responses that returned in full; it is a transport
measure, not a correctness one. The single FAIL is a completed response that
carried a wrong answer, which is why 30 / 30 and FAIL 1 appear together.

For reference, the released checkpoint measured EXACT 29 / NEAR 1 / FAIL 0 on
the same profile and settings. Two-sided Fisher p = 0.35, so at n=30 the two are
**not statistically distinguishable**. Two observations are recorded anyway: this
checkpoint produced one hard FAIL (`60, 35.75`, far outside the ±4 band) where
the released build produced none across 120+ runs, and it emits about 37% fewer
completion tokens at the same `max` setting. Since effort is purely a token
budget on this model, a checkpoint that reasons less at `max` may give up some
accuracy for the same nominal setting.

## DCP3 versus DCP1

| | DCP1 | DCP3 |
| --- | ---: | ---: |
| GPU KV capacity | 1,866,544 tokens | 6,900,837 tokens |
| C1 output tok/s, median | 201.1 | 196.5 |
| C1 verifier steps/s, median | 80.7 | 76.2 |
| C8 aggregate tok/s, median | 661.9 | 623.1 |
| C8 verifier steps/s, median | 253.8 | 239.4 |

DCP3 shards the attention KV cache across the three ranks for 3.7× the capacity,
at about 5–6% of verifier rate. DCP is a start-time setting:
`DCP=3 docker compose up -d` for capacity, `DCP=1` to return to the faster
configuration.

## Sustained decode benchmark at 350 W

`llm-inference-bench` standard sweep, MTP3, temperature 1 / top_p .95, GPUs at
350 W. C8 values are aggregate output throughput across eight requests, not
per-user speed. Verifier steps/s is `output tok/s ÷ acceptance length`;
acceptance length is the average tokens emitted per engine step.

### DCP1

| Context | C1 tok/s | C1 steps/s (accept) | C8 aggregate tok/s | C8 steps/s (accept) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 198.4 | 81.3 (2.44) | 665.4 | 254.8 (2.61) |
| 8K | 203.7 | 80.9 (2.52) | 657.0 | 255.1 (2.58) |
| 32K | 203.9 | 80.5 (2.53) | 658.4 | 252.9 (2.60) |
| 128K | 194.2 | 79.4 (2.45) | 676.6 | 250.2 (2.70) |

### DCP3

| Context | C1 tok/s | C1 steps/s (accept) | C8 aggregate tok/s | C8 steps/s (accept) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 202.5 | 76.8 (2.64) | 618.3 | 240.0 (2.58) |
| 8K | 195.7 | 76.2 (2.57) | 612.2 | 238.2 (2.57) |
| 32K | 197.3 | 76.2 (2.59) | 627.8 | 238.9 (2.63) |
| 128K | 190.7 | 75.7 (2.52) | 658.3 | 241.1 (2.73) |

### Prefill

| Context | DCP1 tok/s | DCP3 tok/s |
| ---: | ---: | ---: |
| 8K | 8,885 | 8,737 |
| 32K | 8,868 | 8,512 |
| 64K | 9,155 | 8,918 |
| 128K | 8,296 | 8,046 |

### Against the released checkpoint

Measured on the same image at the same 350 W, different sessions:

| | Delta versus released NVFP4 |
| --- | --- |
| Decode output tok/s (C1 and C8, both DCP) | **parity** (medians within ±2%) |
| Verifier steps/s | −0.6% to −3.5% |
| **Prefill** | **−15% to −23%** at every context, both DCP |
| KV capacity | −10.7% (DCP1), −7.9% (DCP3) |

Marlin dequantizes weights where the NVFP4 W4A4 kernel does not. Prefill is
compute-bound and pays for that; decode at these batch sizes is memory-bound and
both read the same 4-bit payload, so the extra work is nearly free. The small
verifier-rate deficit is offset by marginally higher MTP acceptance, which is
why raw output throughput lands level.

For chat-style use the cost is negligible. For long-prompt or agentic workloads
that re-prefill often, budget roughly 20% more prefill time and about 11% less
context capacity.

This host drifts 1–4% within an hour and the two sides were measured in
different sessions, so differences under about 4% should not be read as real.
The prefill gap and KV difference are well outside that band.

## Example Compose

Save [compose.yaml](compose.yaml) and
[serve-glm53-flash-tp3-r34.sh](serve-glm53-flash-tp3-r34.sh) together; the
launcher must sit next to the Compose file, which bind-mounts it. This file
defaults to `CHECKPOINT=uncensored`; the released checkpoint remains selectable
on the same image for comparison:

```bash
docker compose config --quiet
docker compose up -d
docker compose logs -f

CHECKPOINT=default docker compose up -d   # released NVFP4, same image
DCP=3 docker compose up -d                # 6,900,837 tokens of KV
```

Both checkpoints must already be present in the Hugging Face cache; neither is
downloaded by this configuration, which runs offline. Other modes use
`MODE=mtp0` or `MODE=dflash2`; MTP3 is the default. GPU power limits are host
settings and are not changed by Compose. A `MODEL` that disagrees with the
selected `CHECKPOINT` fails closed, as does an unknown `CHECKPOINT` value.

Because `max` can produce very long completions, send an explicit `max_tokens`
on requests that must terminate:

```bash
curl -s http://127.0.0.1:15015/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-TP3","max_tokens":4096,
       "messages":[{"role":"user","content":"Explain how prefix caching works."}]}'
```

The full companion Compose is reproduced below.

```yaml
services:
  glm53-tp3:
    image: "${GLM53_IMAGE:-azallaza/glm53-r34-tp3:native-dcp-20260913}"
    container_name: glm53-r34-tp3

    restart: "no"
    init: true

    ipc: host
    shm_size: "64gb"
    # Match this host; choose available CPU IDs on another machine.
    cpuset: "${GLM53_CPUSET:-8-47}"

    ports:
      - "${GLM53_PORT:-15015}:8000"

    volumes:
      # Hugging Face hub cache (checkpoint download / offline reuse).
      - huggingface:/root/.cache/huggingface

      # Persistent JIT / FlashInfer-autotune caches, keyed by the image's
      # source-delta fingerprint. First start of each mode on a new image
      # pays autotune once; afterwards startup is warm.
      - runtime-cache:/cache

      # Required: current launcher has settings absent from the image copy.
      - ./serve-glm53-flash-tp3-r34.sh:/usr/local/bin/serve-glm53-flash-tp3-r34.sh:ro

    environment:
      # Which qualified checkpoint to serve. Switching requires a restart:
      #   CHECKPOINT=uncensored docker compose up -d
      #   CHECKPOINT=default    docker compose up -d
      #
      #   uncensored  orcarouter/GLM-5.3-Flash-Uncensored-NVFP4 (this file's
      #               default). Selecting it also switches the expert and MTP
      #               kernels, because its weights are compressed-tensors W4A16
      #               NVFP4 with an unquantized BF16 MTP layer.
      #   default     the released local-inference-lab GLM-5.3-Flash-NVFP4
      #               build, for comparison against the same image.
      #
      # Both checkpoints must be present in the Hugging Face cache; neither is
      # bundled with this configuration.
      CHECKPOINT: "${CHECKPOINT:-uncensored}"

      # The launcher pins model and revision per CHECKPOINT; a MODEL that
      # disagrees with the selected checkpoint fails closed. Leave unset unless
      # you are deliberately asserting the expected values.
      MODEL: "${MODEL:-}"
      MODEL_REVISION: "${MODEL_REVISION:-}"

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

      # Current baseline: DCP1. DCP3 provides more KV capacity.
      DCP: "${DCP:-1}"
      # Native DCP adapter is guarded off at DCP1; qualified for MTP3/DCP3.
      VLLM_TP3_PCIE_DCP: "${VLLM_TP3_PCIE_DCP:-1}"

      # Server defaults; requests can override these template settings.
      REASONING_EFFORT: "${REASONING_EFFORT:-max}"
      CLEAR_THINKING: "${CLEAR_THINKING:-true}"

      # Prefix caching. It cannot be turned off for this model, and the launcher
      # rejects 0 rather than accepting it silently: vLLM V1 defaults prefix
      # caching ON so omitting the flag is a no-op, and forcing
      # --no-enable-prefix-caching drops mamba_cache_mode from 'align', which
      # the GLM-5.3 split target/recurrent-state page layout requires, aborting
      # engine startup.
      ENABLE_PREFIX_CACHING: "${ENABLE_PREFIX_CACHING:-1}"

      # L2 weight-prefetch budgets (decimal MB). These halves of the image
      # defaults measured about +2% decode speed on three RTX PRO 6000 at
      # 300 W and remain enabled in the current baseline.
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
          export DCP="$${DCP:-1}"

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
              export VLLM_TP3_PCIE_DCP=0
              export SPECULATOR=mtp
              export MTP_DEPTH=0
              unset DFLASH_DEPTH || true
              ;;

            dflash2)
              export VLLM_TP3_PCIE_DCP=0
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
          echo "CHECKPOINT=$${CHECKPOINT:-default}"
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
