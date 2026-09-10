# GLM-5.3-Flash — R30 TP3

This deployment serves the released NVFP4 checkpoint on **three NVIDIA RTX PRO
6000 Blackwell Workstation Edition GPUs (96 GB each)**, using MTP depth 3,
FP8 KV cache and a 1,048,576-token maximum context.

The layout follows the [upstream R30 deployment page](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash.md).
This is a separately patched and locally tested TP3 derivative. The tables
below describe this three-GPU configuration and its own measurements.

## Status

| Capability | TP3 status |
| --- | --- |
| TP / EP / DCP | 3 / 3 / 1 |
| Normal serving | MTP3; temperature 1, top_p 0.95 |
| Other tested modes | No speculation; DFlash2 K7 with the pinned incoai BF16 draft |
| Checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4` |
| Checkpoint revision | `46aaae8a82032f77100f2f03e9cc11b391df3b4d` |
| Context / request slots | 1,048,576 tokens / 8 slots; slots are not eight simultaneous full-context capacity guarantees |
| Target KV cache | FP8; GPU-only cache |
| CUDA graphs | FULL_AND_PIECEWISE |
| Prefix reuse | Request-boundary recurrent checkpoints |
| Vision / concurrency | Exact OCR, concurrent exact-answer smokes and 17,653-token retrieval passed |
| Long-history stress | 840,340 prompt tokens with 41,611 cold / 27,569 warm completion tokens |
| TP3 external cache / DCP > 1 | Outside this launcher's supported envelope |
| Qualification date | 2026-09-09 |

## Docker artifact

Published image (publication reported by the maintainer):
[azallaza/glm53-r30-tp3 on Docker Hub](https://hub.docker.com/r/azallaza/glm53-r30-tp3).
The example Compose defaults to this tag:

```text
azallaza/glm53-r30-tp3:ported-20260909
```

Local image ID (configuration digest, not a registry pull digest):

```text
sha256:5488ea029556f3eee280f11b05ae3ebab178b6ba281e602c7299b89ae6b0490d
```

Immutable upstream parent:

```text
localinferencelab/vllm@sha256:5f6fcbc681f20b7c052815ca17511d9fe789aea314a17723c202789dd7adc131
```

The derivative adds 29 overlay files: TP3 geometry and padded loading,
three-rank collectives, runtime checks, its launcher and a regenerated GDN
profile. All 11 checked native libraries match the R30 parent. Model weights
are downloaded separately; they are not bundled in the image. Local image
size is about 43.3 GB before registry compression.

The image retains its historical `local-inference.status=development-unqualified`
build label. Local qualification occurred after that build; this document
records those results without changing the tested image bytes. The recent
experimental sampling fusion and MHC prototypes are not included.

## Runtime backends

| Operation | Selected implementation |
| --- | --- |
| Target sparse attention / dense projections | B12X |
| Recurrent prefill | FlashKDA, R30 stable FP32 inverse |
| Recurrent decode | B12X when eligible, supported fallback otherwise |
| Target routed experts | FlashInfer CUTLASS NVFP4, EP3, autotuning enabled |
| TP collectives | B12X PCIe, with supported fallback for other sizes |
| MTP proposal vocabulary | Private NVFP4 draft head; target verifier stays BF16 |
| Sampling | Probabilistic drafting and standard rejection; FlashInfer top-p/top-k |

The upstream B12X target MoE wrapper rejects EP3. `MOE_BACKEND=auto` is
intentional here; forcing the TP4 page's B12X MoE setting prevents startup.
The checkpoint has no RoPE dimensions, so generic MLA RoPE/cache fusion does
not apply.

## Measured performance

### MTP3 sustained decode across context lengths

Measured with `llm-inference-bench` v0.6.2, recorded timestamp
`2026-09-09T19:38:15.171127`, against `GLM-5.3-Flash-TP3` at port 15015.
The benchmark identifies the server as `vLLM 0.26.1rc0+glm53.r30.vllm60e72555`.
Concurrency is **one**, with a **30-second sustained measurement per context**,
a 3-second initial decode warmup at 128K, and per-cell readiness requiring
3 seconds of stable occupancy. Requests use `max_tokens=8192` and
`ignore_eos=true`. The run records three GPUs with 300 W power limits each.

| Nominal context | Actual prompt tokens | Decode output tok/s | Verifier steps/s | Effective output tokens/step |
| --- | ---: | ---: | ---: | ---: |
| 0 | 78 | 193.33 | 76.98 | 2.511 |
| 8K | 8,198 | 198.57 | 76.30 | 2.602 |
| 16K | 16,228 | 196.04 | 75.61 | 2.593 |
| 32K | 32,319 | 187.06 | 75.02 | 2.494 |
| 64K | 64,509 | 191.45 | 74.70 | 2.563 |
| 128K | 128,878 | 191.95 | 73.72 | 2.604 |

Decode throughput uses the report's `aggregate_tps` field, sourced from
continuous OpenAI usage during the measured window. Effective tokens per step
include the verifier token, not just accepted draft tokens. Context labels
are target buckets; actual prompt lengths are shown separately and grow
during generation. These are one run per context, not repeated-run medians
or completed-request end-to-end throughput.

All six cells report zero request errors, no capacity limitation and no
detected exact-text loops. Loop detection is a repetition heuristic, not a
semantic accuracy evaluation. The JSON records temperature as `null` and does
not record an explicit top_p value; the deployment defaults are documented
above, but the benchmark does not independently capture their effective values.
Its engine identity is recorded, but an immutable image digest is not.

Source: `/home/aabduh/llm-inference-bench/benchmark_results.json`.
A [preserved measurement extract](mtp3-decode-benchmark.json) records the
source SHA-256, metadata and exact table inputs. The short qualification
benchmarks below use a different protocol and remain separately labeled.

### Short qualification benchmarks

| Test | Result |
| --- | ---: |
| MTP3 short decode, before long-context pair | 200.9 output tok/s |
| MTP3 short decode, after long-context pair | 197.3 output tok/s |
| Historical R27 comparison using the same short harness | 203.7 output tok/s |
| No-spec short smoke benchmark | 113.9 output tok/s, before GDN profile regeneration |
| DFlash2 K7, pinned incoai BF16 draft | 190.1 output tok/s |

Short MTP3 results are concurrency one, temperature zero, one excluded warmup
and three 512-token samples. Rate uses the interval between first and last
nonempty streamed chunks. These results do not establish a speedup over R27.
Speculative acceptance, startup/autotuning and workload change the result.

| Retained-history MTP3 | Cold | Warm |
| --- | ---: | ---: |
| Prompt tokens | 840,340 | 840,340 |
| Prefix hits | 0 | 840,340 |
| Completion tokens | 41,611 | 27,569 |
| First byte | 121.46 s | 1.51 s |
| Approximate output rate | 177.46 tok/s | 171.34 tok/s |
| Preemptions | 0 | 0 |

The long pair used temperature 1, default top_p 0.95 and explicit
`clear_thinking=false`. Both stopped normally. Sampled review found no
sustained corruption; the warm reasoning briefly repeated a confused hash
comparison before recovering. Requested word counts were not met. These
finite tests do not establish universal accuracy or eliminate corruption.

## Start the server

Requires Docker Compose with NVIDIA GPU support and three compatible GPUs.
The example uses named volumes for weights and compilation/autotuning caches;
no source-code mounts are needed. Allow space for the approximately 184 GB
checkpoint plus runtime image and caches. A fresh weights volume downloads the
pinned checkpoint, and a fresh runtime volume needs compilation/autotuning.

Save this as `compose.yaml` (also supplied alongside this document):

```yaml
services:
  glm53-tp3:
    image: ${GLM53_IMAGE:-azallaza/glm53-r30-tp3:ported-20260909}
    container_name: glm53-r30-tp3
    restart: "no"
    init: true
    ipc: host
    ports:
      - "${GLM53_PORT:-15015}:8000"
    volumes:
      - huggingface:/root/.cache/huggingface
      - runtime-cache:/cache
    environment:
      MODEL: local-inference-lab/GLM-5.3-Flash-NVFP4
      MODEL_REVISION: 46aaae8a82032f77100f2f03e9cc11b391df3b4d
      SERVED_MODEL_NAME: GLM-5.3-Flash-TP3
      TP: "3"
      DCP: "1"
      SPECULATOR: mtp
      MTP_DEPTH: "3"
      MAX_MODEL_LEN: "1048576"
      MAX_NUM_SEQS: "8"
      MAX_NUM_BATCHED_TOKENS: "4096"
      GPU_MEMORY_UTILIZATION: "0.95"
      CACHE_MODE: vram
      ENABLE_PREFIX_CACHING: "1"
      CUDAGRAPH_MODE: FULL_AND_PIECEWISE
      PREFILL_COMPUTE_SHARE: "0.4"
      PREFILL_SCHEDULE_INTERVAL: "1"
      VLLM_NO_USAGE_STATS: "1"
    entrypoint: ["/usr/local/bin/serve-glm53-flash-tp3-r30.sh"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0", "1", "2"]
              capabilities: [gpu]
volumes:
  huggingface:
  runtime-cache:
```

Pull the published image and start the server:

```bash
docker compose -f compose.yaml config --quiet
docker compose -f compose.yaml pull
docker compose -f compose.yaml up -d
docker compose -f compose.yaml logs -f glm53-tp3
```

To override the published tag, set `GLM53_IMAGE` to another full image reference
or a verified registry digest. The registry manifest digest has not been
recorded here; do not use the local image ID as a registry pull digest.

The example uses the same container name and port as the qualified local
service: it is an alternative deployment, not a second concurrent instance.
On the existing host, the original `r30-port/compose.yaml` remains the active
configuration. It also applies host CPU affinity `8-47` and reuses the existing
read-only `/m2-2/huggingface` cache. The portable example omits that
host-specific CPU affinity and uses a writable named weights volume.

With `ipc: host`, the host shared-memory filesystem applies. Adding a
Compose `shm_size` does not enlarge host `/dev/shm`.

### Modes

For no speculation, change `MTP_DEPTH` to `"0"`. For the locally tested DFlash2
mode, replace `SPECULATOR`/`MTP_DEPTH` with:

```yaml
SPECULATOR: dflash2
DFLASH_DEPTH: "7"
DFLASH_MODEL: incoai/GLM-5.3-Flash-DFlash2
DFLASH_MODEL_REVISION: dc77ff1c99eeb2df044ee3d4f0094eb033fee410
```

The upstream published MXFP8 draft has not been requalified in this TP3 port.

### Sampling and reasoning

Server defaults are temperature 1.0, top_p 0.95, reasoning effort high and
`clear_thinking=true`. Explicit request values override sampling defaults.
For retained-history work, send
`"chat_template_kwargs":{"clear_thinking":false}` and preserve prior assistant
reasoning in the conversation. Removing old reasoning changes the prompt;
it is not a numerical correction.

### Check readiness

```bash
curl --fail http://localhost:15015/health
curl --fail http://localhost:15015/v1/models
curl --fail http://localhost:15015/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-TP3","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":256,"temperature":1,"top_p":0.95}'
```

`/health` returns HTTP 200; `/v1/models` should list `GLM-5.3-Flash-TP3` with
`max_model_len` 1048576. A small output cap can be consumed by reasoning.

## Important startup log values

These observations are from the restored, qualified image. Memory totals and
startup duration depend on available GPU memory, cache state and GPU clocks.

```text
R30 TP3 candidate: TP=3 EP=3 DCP=1 spec=mtp:3 autotune=1 cache=r30-tp3-vllm60e72555-b12x3edbcbce-a2da5a0ffea0d20b
GLM53_TP3_RUNTIME_PROOF {"collective_backend":"b12x_pcie_oneshot","expert_parallel_size":3,"kda_decode_backend":"b12x","kda_prefill_backend":"flashkda","mm_encoder_tp_mode":"weights"}
```

| Log item | Observed / expected value |
| --- | --- |
| Expert placement | 96 local / 288 global experts per rank; linear EP placement |
| Target MoE | `FLASHINFER_CUTLASS` |
| Draft vocabulary copy | 113.48 MiB per rank; target vocabulary remains BF16 |
| Model loading memory | 63.04 GiB per rank |
| Attention / recurrent pages | 2048 / 2048 tokens |
| Cache layer groups | Rebalanced to `[9, 9, 8, 8, 12]` |
| Physical page sizes | `[1148928, 1543168]` bytes |
| Shared-pool max-request cost | 9,758,994,432 → 7,388,688,384 bytes, a 24.29% allocation reduction |
| Available KV memory, rank 0 | 10.51 GiB |
| GPU KV token capacity | 1,596,516 tokens |
| Maximum full-context concurrency | 1.52× at 1,048,576 tokens per request |
| Graph memory | 3.75 GiB actual per rank; 7.58 GiB estimate |
| Memory utilization target | 0.95 |
| Sampling defaults | `{'temperature': 1.0, 'top_p': 0.95}` |
| Recurrent prefix policy | Request-boundary caching enabled |
| API ready | `Application startup complete`, HTTP 200 health |

The graph-memory estimator's message suggesting utilization 1.0 is diagnostic,
not this guide's recommended setting; keep the tested 0.95. Likewise, the
scheduler warning about 4096 tokens does not mean the qualified run failed.
Eight request slots share the available cache; capacity is workload dependent.

## Qualification and known limits

Local checks include 42 focused port tests, 120 collective dispatch tests,
seven GPU collective tests, seven Mamba retirement tests and six GPU
checkpoint-restore tests. The regenerated GDN profile contains 1,468
correctness-gated cases. Exact-answer smokes, OCR, concurrency and the long
pair supplement these kernel checks.

TP4/TP8 were not hardware-tested on this three-GPU host. TP3 external cache,
packed NVFP4 KV and alternate DCP configurations are not qualified here.
The four subsequent decode investigations did not yield a production change:
MHC hybrid was slower; MLA fusion was inapplicable; probabilistic sampling
fusion saved only about 2 microseconds per call; B12X EP integration remains
experimental.

Local evidence: `r30-port/SUMMARY.md`, `STATUS.md`, `PATCH-LEDGER.md`,
`results/long-context/RESULTS.md`, and
`optimization/four-leads-20260909/RESULTS.md`. This guide describes the existing
qualified image; publishing it does not restart or modify the running server.
