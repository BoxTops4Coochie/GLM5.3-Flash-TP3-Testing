# GLM-5.3-Flash — R34 TP3

GLM-5.3-Flash NVFP4 on three NVIDIA RTX PRO 6000 Blackwell Workstation Edition
GPUs (96 GB each), with MTP3, FP8 KV and a 1,048,576-token maximum context.
This updates the previously published R34 TP3 guide for a newer image, a DCP1
default and a `max` reasoning default, and is taken from the running service's
image and Compose settings on September 16, 2026. It supersedes the
`native-dcp-20260913` guide; the change is that DFlash2 now runs under DCP3.

## Status

| Setting | Value |
| --- | --- |
| Checkpoint | `default` (released NVFP4); Compose `CHECKPOINT` also offers `uncensored` |
| TP / EP / DCP | 3 / 3 / 1; DCP3 remains available |
| Default serving | MTP depth 3 |
| Other smoke-tested modes | No speculation; DFlash2 depth 7 with the retained incoai BF16 draft, at DCP1 or DCP3 |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4` |
| Target revision | `46aaae8a82032f77100f2f03e9cc11b391df3b4d` |
| Context / request slots | 1,048,576 / 8; capacity depends on request lengths |
| Sampling | Temperature 1, top_p .95, repetition penalty 1; normal EOS |
| Reasoning effort | `max` by default; `low`/`high`/`max` are the only distinct tiers; request-overridable |
| Historical thinking | `clear_thinking=true` by default; request-overridable |
| KV / graphs | FP8 GPU-only cache / FULL_AND_PIECEWISE |
| Prefix caching | Enabled; required by this split-page configuration and not disableable |
| Batch / maximum graph size | 4096 / 32; configurable in Compose |
| GPU KV capacity at this startup | 2,091,238 tokens |
| GPU power limit | 350 W per GPU; host setting |
| CPU affinity on this host | CPUs 8–47 |

Eight request slots do not mean eight simultaneous full-context requests fit.
At DCP1, startup reports 1.99× capacity for 1,048,576 tokens per request.

## Docker artifact

Published image: `azallaza/glm53-r34-tp3:native-dcp-20260916`

Image ID `sha256:8a8415a74bb1…`, source delta
`f51df58b4846f75fcf496cc266eb04033feaa6aa98a72f4d460d6ceb178b0b64`, 34 overlay
files over the pinned R34 parent.

```bash
docker pull azallaza/glm53-r34-tp3:native-dcp-20260916
```

This supersedes `azallaza/glm53-r34-tp3:native-dcp-20260913`, which in turn
superseded `azallaza/glm53-r34-tp3:20260910`. The single functional change from
`native-dcp-20260913` is the DFlash2 decode-context-parallelism fix below.

It differs from the published September 10 artifact by these source changes:

- `vllm/v1/attention/ops/dcp.py`: the DCP log-sum-exp combine kernel used the
  DCP world size directly as a Triton `arange` extent, which must be a power of
  two, so world size 3 could not start. The kernel now masks the extra lanes;
  inert at DCP=1.
- `vllm/models/glm5next/nvidia/glm53_low_latency_gemm.py` (new, plus two-line
  hooks in the GLM model and MTP constructors): for the small-batch BF16 decode
  projections that cuBLAS runs as a split-K GEMM + reduce pair or an SM80 WMMA
  fallback, use the fork's CuTe skinny GEMM at the exact (shape, batch≤8) pairs
  where it measured faster on RTX PRO 6000. `VLLM_GLM53_LOW_LATENCY_GEMM=0`
  disables it.
- `vllm/v1/sample/ops/topk_topp_sampler.py`: batch<8 top-p uses FlashInfer's
  deterministic nucleus renorm as a mask instead of sort/softmax/cumsum over
  the 155k vocabulary (identical probabilities to 1e-6; at `top_p=1.0` the mask
  keeps every nonzero-probability token, matching the sort path exactly).
- The September 13 image adds fused DCP empty-shard metadata masking and a
  guarded three-rank native PCIe DCP adapter. Larger batches retain NCCL. The
  adapter is inactive at DCP1; Compose disables it for MTP0 and DFlash2.
- `vllm/config/speculative.py`: the TP3 draft-placement step copied the target's
  decode-context-parallel settings onto the draft for both MTP and DFlash. The
  MTP draft is MLA and genuinely participates in DCP, but the DFlash2 draft is a
  dense GQA model, and vLLM requires `--tensor-parallel-size` to exceed a GQA
  model's total KV head count before it will shard it. At TP3 that is
  impossible, so `MODE=dflash2` with `DCP=3` aborted during engine config
  validation. The four `dcp_*` fields are now skipped for DFlash only, matching
  upstream — which never copies DCP to a draft and qualifies TP4/DCP4 with
  DFlash2 depth 7. MTP behaviour is unchanged.

This image bakes in the current TP3 launcher
(`sha256:4b250132…`), so unlike `native-dcp-20260913` it does **not** require the
launcher bind-mount to behave correctly. `CHECKPOINT`, `REASONING_EFFORT` and
`CLEAR_THINKING` all work from the image alone.

The Compose file still mounts [serve-glm53-flash-tp3-r34.sh](serve-glm53-flash-tp3-r34.sh)
over the image copy, which is harmless and lets you edit launcher behaviour
without rebuilding. Verify the file against `launcher_sha256` in
[baseline-receipt.json](baseline-receipt.json); if it matches, mounting it
changes nothing.

## Runtime backends

| Operation | Backend |
| --- | --- |
| Target attention | B12X |
| Dense projections | B12X / cuBLAS, with the CuTe skinny GEMM at measured small-batch shapes |
| Recurrent prefill / decode | FlashKDA / B12X |
| Routed target experts | EP3-capable automatic selection (FlashInfer CUTLASS NVFP4) |
| TP collectives | B12X PCIe |
| MTP experts / attention | Marlin / B12X |
| MTP vocabulary head | Private NVFP4 draft copy; BF16 target verifier |

`MOE_BACKEND=auto` is intentional for EP3. Native libraries remain from R34.
The port includes TP3 geometry/loading, three-rank collectives, the qualified
GDN profile component, and the local explicit-no-tool parser fix.

Two runtime properties are not configurable on this stack, and both were
verified directly on September 15, 2026:

- **Prefix caching cannot be disabled.** `ENABLE_PREFIX_CACHING=0` only omitted
  `--enable-prefix-caching`, and vLLM V1 defaults prefix caching on, so the
  setting silently had no effect. Passing `--no-enable-prefix-caching`
  explicitly drops `mamba_cache_mode` from `align`, which this model's split
  target/recurrent-state page layout requires, and the engine core aborts at
  startup. The mounted launcher now rejects 0 instead of accepting it silently.
- **The KV cache cannot be unquantized.** With the B12X backend,
  `_canonicalize_sparse_mla_kv_cache_dtype` rewrites `auto`, `fp8` and
  `fp8_e4m3` to `fp8_ds_mla` unconditionally, so `--kv-cache-dtype auto` still
  yields FP8. There is no BF16 KV path without changing attention backends.

Both are therefore present in every result below.

## Validation

| Build / setting | Recorded validation |
| --- | --- |
| Original R34 TP3 port | 3,191 component tests; MTP0, MTP3 and DFlash7 exact-answer smokes; MTP3 concurrency, OCR and 17.6K retrieval |
| Original R34, DCP1 | 826,266-token retained history → 28,883 completion tokens, normal stop; exact markers at 127,992 and 899,994 prompt tokens |
| `dcp-20260910`, DCP3 | Concurrency, OCR, retrieval; 826,266-token retained history → 27,592 completion tokens, normal stop |
| `llgemm-20260911`, DCP3 | Smokes, concurrency, OCR, 128K/900K retrieval; retained history → 29,777 completion tokens, normal stop |
| `llgemm-20260911b`, DCP3 | Geometry/collective CPU and GPU checks, smokes, concurrency, OCR and 17.6K retrieval after the two bug fixes |
| Native DCP follow-up, DCP3, 300 W | 448 exact mutable metadata graph checks; 576 rank-local collective checks; 32/32 mixed exact tasks; 128K/900K retrieval; natural 115,331-token prompt → 9,221-token output, normal EOS; packaged-image checks passed |

Earlier retrieval answers sometimes used Markdown JSON fences despite a bare
JSON instruction. The older DCP3 retained-history response repeated a user
message within a reasoning recap. These are bounded checks, not a guarantee
against all model errors or universal factual accuracy; prose accuracy was not
audited. The original 826K history test was not rerun as part of the native DCP
follow-up.

### Answer quality versus reasoning effort

The checkpoint's chat template exposes **three** effort tiers, not the usual
OpenAI ladder. It matches only `low` and `high` literally and renders every
other value — `minimal`, `medium`, `xhigh`, `max`, or any typo — as
`Reasoning Effort: Max`:

```jinja
{%- set effective_reasoning_effort = reasoning_effort
      if reasoning_effort is defined and reasoning_effort in ['low', 'high']
      else 'max' -%}
```

`llm-inference-bench`'s `lavd` profile is a 48K-character ledger the model must
keep consistent, scored EXACT / NEAR / FAIL against the pair `72, 46`, where
NEAR means both totals fall within ±4. Thirty runs per arm at concurrency 10:

| Arm | Exact | Near | Fail | Median completion tokens |
| --- | ---: | ---: | ---: | ---: |
| `high` | 18 | 12 | 0 | 6,288 |
| `max` | 29 | 1 | 0 | 20,287 |

60.0% versus 96.7% exact (two-sided Fisher p = 0.001). This is the reason the
server default is `max`. No run in either arm produced a wrong answer; every
miss is NEAR, arithmetic drift of a point or two in a long ledger rather than a
lost fact. Temperature 0 did not help. The full `max` run is transcribed from
the [lavd-test screenshot](lavdtest.jpeg):

| Metric | Result |
| --- | ---: |
| Completed | 30 / 30 |
| Score | EXACT 29 / NEAR 1 / FAIL 0 |
| Reported pass rate (EXACT + NEAR) | 100.0% (30 / 30) |
| Hit maximum completion-token limit | 0 |
| Completion tokens, average | 21,746 |
| Completion tokens, p50 / p90 / p99 | 20,287 / 30,142 / 36,264 |
| Average request elapsed time | 243.8 s |
| Average TTFT | 43.58 s |
| Reported aggregate generation rate | 108.6 tok/s |
| Mean per-request generation rate | 109.5 tok/s |
| Whole-run GPU power, average / maximum | 1,047 / 1,057 W |
| Combined GPU power limit | 1,050 W |
| Power observation window / samples | 13 min 33 s / 352 |

In this completion-statistics report, "aggregate generation rate" is total
completion tokens divided by the **sum of request generation durations**. It is
not concurrent server output per wall-clock second and should not be compared
directly with the C8 aggregate decode throughput below. GPU power figures are
summed GPU telemetry, not outlet or whole-system power.

`max` can produce very long completions, so send an explicit `max_tokens` on
requests that must terminate.

These are single-host measurements on one checkpoint, not a general accuracy
claim for the model or the quantization.

## DCP3 versus DCP1

DCP1 is the current default and the faster choice for decode. DCP3 shards the
attention KV cache across the three ranks and provided **7,493,749 tokens** in
the native image's qualified configuration, versus **2,091,238** at this DCP1
startup. Actual capacity depends on graph sizes, serving mode and memory
allocation.

The earlier September 10 comparison measured a DCP3 verifier-rate cost of about
8% below 128K context and about 1% at 512K. Those percentages predate the
native DCP optimization, which measured **2.0–2.7% faster C1 verifier rate** at
0/128K context in repeated matched pairs at DCP3, MTP3 and 300 W; C8 showed no
meaningful gain. That is not a speedup claim for the current DCP1 default.

Switch with `DCP=3 docker compose up -d` for capacity, or `DCP=1 docker compose
up -d` to return to this baseline.

DFlash2 can now use either setting. Measured on this image with the uncensored
checkpoint: **1,972,587** tokens of KV at DCP1 against **6,343,179** at DCP3.
On `native-dcp-20260913` and earlier, `MODE=dflash2 DCP=3` aborted at startup.

## Sustained decode benchmark at 350 W

Transcribed from the [DCP1 screenshot](DCP1bench.jpeg) and
[DCP3 screenshot](DCP3bench.jpeg). The baseline receipt records 350 W per GPU;
the decode screenshots themselves do not show power, warmup or measurement
duration. C1 is one concurrent request; C8 is eight, and C8 values are
aggregate output throughput across requests, not per-user speed.

| Context | DCP1 C1 tok/s | DCP1 C8 aggregate tok/s | DCP3 C1 tok/s | DCP3 C8 aggregate tok/s |
| --- | ---: | ---: | ---: | ---: |
| 0 | 199.9 | 636.3 | 191.3 | 619.3 |
| 8K | 184.9 | 656.6 | 182.0 | 583.4 |
| 32K | 206.6 | 687.7 | 200.6 | 620.8 |
| 128K | 195.5 | 643.3 | 198.4 | 688.7 |

MTP acceptance changes raw output throughput. The next table reports the
benchmark's MTP-normalized rate, `output tok/s ÷ acceptance length`, with
acceptance length in parentheses — the average number of tokens emitted per
engine step. The C8 rate retains the benchmark's aggregate accounting.

| Context | DCP1 C1 steps/s (accept len) | DCP1 C8 steps/s (accept len) | DCP3 C1 steps/s (accept len) | DCP3 C8 steps/s (accept len) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 81.9 (2.44) | 259.1 (2.46) | 76.8 (2.49) | 249.3 (2.48) |
| 8K | 81.1 (2.28) | 263.3 (2.49) | 77.0 (2.36) | 243.7 (2.39) |
| 32K | 81.4 (2.54) | 263.4 (2.61) | 77.2 (2.60) | 245.9 (2.52) |
| 128K | 80.4 (2.43) | 255.0 (2.52) | 76.7 (2.59) | 251.7 (2.74) |

DCP1 has the higher normalized rate in all displayed cells. At 128K/C8, DCP3's
higher raw throughput comes with higher acceptance length (2.74 versus 2.52),
while its normalized rate remains slightly lower. These screenshots alone do
not establish repeatability or a controlled DCP-only speed difference.

Prefill, one sample per point (`N=1`). TTFT is time to first token; actual
prompt-token counts differ from the nominal context labels.

| Context label | Prompt tokens | DCP1 TTFT (s) | DCP1 prefill tok/s | DCP3 TTFT (s) | DCP3 prefill tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K | 8,197 | 0.74 | 11,094 | 0.79 | 10,330 |
| 32K | 32,313 | 2.86 | 11,300 | 2.97 | 10,865 |
| 64K | 64,491 | 5.70 | 11,323 | 5.85 | 11,026 |
| 128K | 128,851 | 12.10 | 10,649 | 12.40 | 10,394 |

For reference, the packaged native image's historical DCP3 results at 300 W,
using 180 s of load warmup, 30 s of cell warmup and 45 s measured:

| Context | C1 output tok/s | C1 verifier steps/s | C8 aggregate output tok/s |
| --- | ---: | ---: | ---: |
| 0 | 181.83 | 73.81 | 562.78 |
| 131,072 | 189.52 | 73.18 | 556.44 |

Output rate varies with MTP acceptance. The previously published September 10
throughput tables remain historical measurements of the older image.

### Single-request decode during real coding work

The figures above are benchmark cells. The table below is observational: it is
taken from the engine's own logs during ordinary coding use on September 15–16,
2026, at `num-seq 1` (`Running: 1 reqs, Waiting: 0 reqs`), and is transcribed
from [codingDecode.PNG](codingDecode.PNG).

These are the **five highest-throughput samples** selected from those logs, so
they are best-case peaks rather than an average or a sustained rate. Prompt
length, GPU power and cache state are not recorded in these log lines.

| Sample | Avg generation tok/s | Mean acceptance length | Accepted tok/s | Per-position acceptance | Draft acceptance |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 | 311.0 | 3.20 | 213.90 | 0.880 / 0.718 / 0.606 | 73.4% |
| 2 | 296.4 | 2.82 | 191.38 | 0.801 / 0.592 / 0.430 | 60.8% |
| 3 | 294.5 | 3.88 | 218.48 | 0.988 / 0.968 / 0.922 | 96.0% |
| 4 | 286.3 | 2.82 | 184.59 | 0.812 / 0.596 / 0.414 | 60.7% |
| 5 | 284.3 | 2.88 | 185.58 | 0.825 / 0.597 / 0.460 | 62.7% |

**Accepted tok/s is the user-visible rate**; it peaks at 218.48. "Avg
generation throughput" is vLLM's engine-side figure from a separate log line
(`loggers.py`) than the speculative-decoding metrics (`metrics.py`), so the two
columns are not necessarily averaged over the same window and the higher number
should not be quoted as delivered output speed.

Mean acceptance length is the average tokens emitted per engine step, so
implied verifier rate ranges from about 56 steps/s (sample 3) to 68 steps/s
(sample 2). Acceptance varies widely with content: sample 3 accepted 96.0% of
drafts — 0.922 even at the third speculative position — while samples 2 and 4
accepted about 60%. Highly predictable code yields long accepted runs; prose
and novel identifiers do not. This spread, not raw decode speed, is what moves
end-user throughput between 185 and 218 accepted tok/s here.

## Example Compose

Save the companion Compose and launcher together. The example preserves the
current serving settings while using named volumes for portable checkpoint and
runtime caches. On this host the operational service instead mounts the
existing Hugging Face cache read-only and its existing writable runtime cache,
with offline loading enabled; named volumes start empty and require a
checkpoint download on first use. CPU affinity defaults to this host's CPUs
8–47; set `GLM53_CPUSET` for another machine.

Compose defaults to the published image and pulls it on first start;
`GLM53_IMAGE` selects a different tag or a local build. GPU power limits are
host settings and are not
changed by Compose. This example uses the existing container name and port; it
is a replacement configuration, not a second simultaneous server. Starting or
switching modes recreates the service.

```bash
docker compose config --quiet
docker compose up -d
docker compose logs -f
```

`CHECKPOINT` selects which checkpoint to serve and restarts into it:

```bash
CHECKPOINT=default    docker compose up -d   # released NVFP4 (all figures above)
CHECKPOINT=uncensored docker compose up -d   # orcarouter fine-tune, see below
```

`default` is the configuration every measurement in this document was taken on;
on that setting the launcher emits byte-identical arguments to the
single-checkpoint version, so the second variant has no effect unless selected.
`uncensored` serves `orcarouter/GLM-5.3-Flash-Uncensored-NVFP4`, which you must
download separately -- it is not part of this release. It ships the same
architecture and a byte-identical tokenizer, but a different quantization
container (compressed-tensors W4A16 NVFP4 with an unquantized BF16 MTP layer),
so the launcher also switches to the Marlin expert kernel and a Triton MTP
kernel. Measured against `default` on this image at 350 W: decode throughput at
parity, prefill 15-23% slower, KV capacity about 11% smaller, and all
functional checks passing at DCP1 and DCP3. A `MODEL` that disagrees with the
selected `CHECKPOINT` fails closed, as does an unknown `CHECKPOINT` value.

Other modes use `MODE=mtp0` or `MODE=dflash2`. MTP3 is the default. The DFlash
preset pins the retained BF16 draft revision; its newer MXFP8 alternative is
not qualified here.

The server default reasoning effort is `max`. Use `REASONING_EFFORT=high docker
compose up -d` to change it, or pass `"reasoning_effort":"high"` in a request.
Only `low`, `high` and `max` are distinct; the launcher rejects other values
rather than letting them render as Max. `CLEAR_THINKING=false` preserves
earlier reasoning in conversation history. Both remain request-overridable.
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
    image: "${GLM53_IMAGE:-azallaza/glm53-r34-tp3:native-dcp-20260916}"
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
      #   default     the released GLM-5.3-Flash NVFP4 build. This is the
      #               configuration every measurement in the guide was taken on.
      #   uncensored  orcarouter/GLM-5.3-Flash-Uncensored-NVFP4, a third-party
      #               fine-tune in a different quantization container. Selecting
      #               it also switches the expert and MTP kernels, because its
      #               weights are W4A16 with an unquantized BF16 MTP layer.
      #               Decode is at parity; prefill is 15-23% slower and KV
      #               capacity about 11% smaller. You must download that
      #               checkpoint yourself; it is not part of this release.
      #
      # On CHECKPOINT=default the uncensored branch is inert: the launcher
      # produces byte-identical arguments to the single-checkpoint version.
      CHECKPOINT: "${CHECKPOINT:-default}"

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
