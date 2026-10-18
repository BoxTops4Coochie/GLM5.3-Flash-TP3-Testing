# GLM-5.3-Flash — Kraken TP3, 20260923

GLM-5.3-Flash NVFP4 on three NVIDIA RTX PRO 6000 Blackwell Workstation Edition
GPUs (96 GB each), MTP3, FP8 KV, 1,048,576-token context. This image builds on
[Kraken TP3 20260921](../kraken-20260921/glm-5.3-flash-tp3-kraken.md) and adds
an FP8 weight-only decode path, an FP8 prefill path, programmatic dependent
launch (PDL) between the PCIe all-reduce and its consumer, and the 256 KiB
one-shot all-reduce cutoff as a default.

**Against the 20260921 baseline (default/MTP3/DCP1): C1 decode +11.4% (88.2
verifier steps/s), C8 decode +8.1% (283), prefill +5.5-6.5%, with
answer-level quality level with the previous release on every test run.**

## What changed since 20260921

| Change | Effect (bracketed, 350 W/GPU) | Numerics |
| --- | --- | --- |
| FP8 weight-only decode (Marlin W8A16) for the large BF16 projections, the LM head, the dense FFN layers and the MTP draft | **+8.6%** C1, +4.5% C8 (decoder projections); **+2.6%** C1 more (LM head, dense FFN, MTP draft) | FP8 E4M3 weights, one scale per output channel, decode batches (<=32 rows) |
| FP8 prefill (CUTLASS W8A8), BF16 copies dropped | **+5.5-6.5%** prefill, **+6%** KV capacity, decode unchanged | FP8 weights and per-token FP8 activations, prefill batches (>32 rows) |
| PDL: one-shot all-reduce releases its consumer early | **+0.56%** C1, neutral C8 | none (scheduling only) |
| `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=256KB` default | **+3.0%** C8, neutral C1 | none (algorithm choice for the same sum) |

Each is switchable from compose; with all off, behaviour matches 20260921
(see [Reverting individual changes](#reverting-individual-changes)).

## Status

| Setting | Value |
| --- | --- |
| Qualified on this image | **default checkpoint, MTP3, DCP1** (all defaults) |
| Carried from 20260921, not re-qualified here | uncensored checkpoint, DFlash2, MTP0, DCP3; see [Scope](#scope-of-qualification) |
| TP / EP / DCP | 3 / 3 / 1 |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4` @ `175ae8ce3b5af842b0d0140dbeb43e9cfc557c49` |
| Context / request slots | 1,048,576 / 8 |
| Sampling defaults | temperature 1, top_p .95; reasoning `max`, `clear_thinking=true` (request-overridable) |
| KV / graphs | FP8 GPU-only cache / FULL_AND_PIECEWISE, graph max 32, finer MTP captures |
| Startup-reported KV capacity | **3,169,983 tokens** (3.02x at maximum context); 20260921: 3,410,670 |
| GPU power limits | 350 W per GPU (host setting) |

The KV figure is 7% below 20260921. Each converted layer keeps two FP8 copies
(Marlin-packed for decode, row-major for prefill) that together equal the BF16
weight they replace, so the gap comes from the extra FP8 LM-head copy (~0.2 GiB)
and ~1 GiB of freed BF16 memory that does not reach the KV pool (allocator /
conversion peak; see [Not done](#not-done)). It is 6% above the same stack with
BF16 prefill (2,989,001).

## Docker artifact

Image: `azallaza/glm53-kraken-tp3:20260923`

```bash
docker pull azallaza/glm53-kraken-tp3:20260923
```

Local build `glm53-kraken-tp3:20260923`, image ID
`sha256:9b3b342159382972e94b07f56ccce8679bc7d43c58cb2aaa30dd84f1788719c2`, built by `build.py` from `src/` on top of
`ghcr.io/local-inference-lab/vllm@sha256:5927520c447fdcbc0990567f9237756ff66a54e360438915a5f70cd5cf9d530d`
(the same parent as 20260921). The registry name is the same artifact.

Built in: everything in 20260921 (TP3 port, DCP rank-padding correction,
recurrent null-state correction, widened runtime proof) plus the changes above.
Compose mounts only the companion [launcher](serve-glm53-flash-tp3-kraken.py);
no source mounts are needed. For review:
[source-changes-20260923.patch](source-changes-20260923.patch) is the diff
against the 20260921 image (model.py +5, mtp.py +2, b12x one-shot +11/-1, b12x
mHC +8/-3) plus the new module [glm53_fp8_dense.py](glm53_fp8_dense.py).
Hashes, observed server state and result summaries:
[release-receipt-20260923.json](release-receipt-20260923.json).
The complete port as patches against the Kraken base (vLLM `67bb922f6`, b12x
1.3.0), with a per-file change list: [upstream/](upstream/README.md).

As in 20260921, the image's lil profile is untouched (its hashes are pinned by
`/opt/lil/image-contract.json`), so the qualified prefill backend and the new
switches are selected through compose environment variables.

## How it got faster

A torch profile of the 20260921 C1 decode step ([decode-profile.md](decode-profile.md))
put it at 13.09 ms with the GPU 94% occupied. Two findings shaped this release:

- **BF16 dense projections were ~3.7 ms of the step, all weight streaming.**
  `modelopt_mixed` keeps attention/KDA projections BF16 while experts are NVFP4.
  Halving their bytes is the FP8 decode path.
- **The PCIe all-reduce was the largest exposed non-compute cost** (72% of
  its 1.6 ms/step on the critical path). World size 3 gets none of b12x's
  TP2/TP4/TP8 remote-push fast paths, so that remains open (see
  [Not done](#not-done)); PDL and the cutoff take the parts reachable without new
  collective kernels.

### FP8 weight-only decode

`vllm/models/glm5next/nvidia/glm53_fp8_dense.py`. After loading, each selected
BF16 weight is quantized to FP8 E4M3 with one scale per output channel and
packed for Marlin W8A16 (activations stay BF16). Batches of at most 32 rows --
every decode and verify step up to 8 sequences x 4 positions -- use it.

Converted per rank: KDA `in_proj_qkvgfab` (8598x4096) and `o_proj`
(4096x2816) x34, DSA `o_proj` (4096x6144) and `q_b_proj` (6144x1536) x11,
dense FFN `gate_up`/`down` x3, the LM head (51648x4096), and the MTP draft's
DSA `o_proj`/`q_b_proj`. Chosen by measurement ([decode-sweep.md](decode-sweep.md),
[fp8-decode.md](fp8-decode.md)): e.g. `in_proj` 44.4 -> 26.9 us and the LM
head 261 -> 137 us from HBM. Small projections and the shared experts are
slower under Marlin and stay BF16.

The packed FP8 tensor becomes the layer's public `weight`, so the existing L2
prefetcher streams the bytes decode reads; a fixed budget now covers about
twice the fraction of each weight. The R34 half-L2 budgets were re-tested for
FP8 weights and kept: sizing them to the FP8 weights measured +0.4% C1 / -0.6%
C8 (noise), the upstream full budgets -1.9% C1 ([l2-budgets.md](l2-budgets.md)).

Unit test: cosine >= 0.99964, relative error <= 2.7% per projection against
BF16, all-zero TP3 padding rows stay exactly zero, CUDA-graph replay identical
to eager. The MTP draft conversion cannot change output quality: the draft only
proposes tokens and rejection sampling preserves the target distribution.

### FP8 prefill

With `VLLM_GLM53_FP8_PREFILL=w8a8` (default), batches above 32 rows of the
converted layers run CUTLASS FP8 on a row-major copy of the same FP8 weights,
with per-token dynamic activation scales, and the BF16 weights are released.
Decode and prefill therefore read identical quantized weights. The LM head
keeps its BF16 weight for large batches, so prompt log-probability requests
stay exact. Unit test: relative error 3.75%, cosine 0.9993 at M=64/512/4096.
Details: [fp8-prefill.md](fp8-prefill.md).

### PDL between the one-shot all-reduce and mHC

The lil profile already sets `B12X_PCIE_ONESHOT_PDL=1` and `B12X_MHC_PDL=1`,
but in this b12x build nothing read the first, and the kernel that consumes
each all-reduce (`MHCPostPrePartialKernel`) had no PDL wait: 0.0% of 41,860
all-reduce -> mHC pairs overlapped. Two small b12x changes (the one-shot's pull
kernel calls `griddepcontrol.launch_dependents` after its cross-GPU barrier;
the partial kernel launches with `use_pdl` and waits at entry) make 98.8%
overlap. Correctness is by construction: the wait returns only after the
all-reduce has completed and its writes are visible. Compile-cache keys are
bumped so a persistent JIT cache cannot serve pre-patch kernels. Details:
[pdl.md](pdl.md).

### 256 KiB one-shot cutoff

At C8 each all-reduce is 32 rows x 4096 x BF16 = 256 KiB, above the upstream
84 KiB one-shot limit, so it fell back to the NCCL ring at 44.9 us versus the
one-shot's 16.1 us. 256 KiB keeps them on the one-shot. C1 messages are 32 KiB
and were already on it.

## Runtime backends

| Operation | Backend |
| --- | --- |
| Target attention | B12X |
| Large BF16 projections, decode (<=32 rows) | **Marlin FP8 W8A16** |
| Large BF16 projections, prefill (>32 rows) | **CUTLASS FP8 W8A8** |
| Other dense projections | B12X / cuBLAS, skinny GEMM at the port's measured shapes |
| Recurrent prefill / decode | FlashKDA / B12X |
| Routed target experts | FlashInfer CUTLASS NVFP4 (EP3) |
| TP collectives | B12X PCIe one-shot up to **256 KiB**, with PDL to the mHC consumer |
| MTP experts / draft head | Marlin MXFP8 / private NVFP4 copy |
| LM head (verifier) | **Marlin FP8 W8A16** for decode; BF16 for large batches |

B12X 1.3.0 and FlashInfer 0.6.18 as in 20260921.

## Validation

### Speed

Measured as bracketed A/B runs on freshly recreated containers, default/MTP3/DCP1, 350 W/GPU,
30 s warmup and 45 s cells, with the changes mounted onto the 20260921 image (the
built image contains byte-identical files, verified by sha256). Verifier
steps/s is the metric: tok/s moves with MTP acceptance, which varies ~5% between
identical runs.

| Configuration | C1 steps/s | C8 steps/s |
| --- | ---: | ---: |
| 20260921 baseline | 79.2 | 262.1 |
| + FP8 decode (decoder projections) | 86.0 (+8.6%) | 274.0 (+4.5%) |
| + PDL | 86.4 (+9.1%) | -- |
| + LM head, dense FFN, MTP draft | 88.2 (+11.4%) | 275.2 |
| + 256 KiB cutoff (**this release**) | 87.9-88.2 | **283.4 (+8.1%)** |

C1 at 32K context: +8.4% from FP8 decode alone. Control pairs agreed within
0.1-0.6% (C1) and up to 1.4% (C8). Full tables: [fp8-decode.md](fp8-decode.md).

Prefill, FP8 prefill versus BF16 prefill on the same stack, cold, tok/s:

| | 8K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: |
| BF16 prefill (better of two controls) | 11,294 | 11,307 | 10,756 | 10,105 |
| **FP8 prefill** | **11,912** | **11,941** | **11,432** | **10,760** |

### Release image benchmark

The built image with the release compose defaults and no mounts other than the launcher;
single pass on a fresh container, llm-inference-bench, 30 s warmup, 45 s cells.
Not a controlled comparison -- the claims above come from the brackets.

| Context | C1 tok/s | C1 steps/s (acceptance) | C8 tok/s | C8 steps/s (acceptance) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 227.8 | 92.63 (2.459) | 725.5 | 290.35 (2.499) |
| 32,768 | 222.5 | 92.47 (2.406) | 758.2 | 289.89 (2.615) |
| 131,072 | 231.0 | 90.45 (2.554) | 736.0 | 278.96 (2.638) |

Prefill tok/s: 8K 11,218, 32K 11,431, 64K 11,157,
128K 10,655. Startup 8/8 exact arithmetic. Raw JSON: [bench/](bench/).

### Quality

Protocols identical to the 20260921 qualification: `lavd` and `hotel-lights`,
30 requests each at concurrency 8, maximum reasoning, temperature 1, top_p .95,
seed 103, 100K-token cap; exact 3-marker retrieval at 128K and 900K; long
generation from an 826K-token retained history with natural EOS, judged EMPTY /
REPEAT (`repeat_8gram` > 0.05) / THIN.

| Test | 20260921 | This release, BF16 prefill | This release, FP8 prefill |
| --- | --- | --- | --- |
| lavd | 27 exact / 3 near | **30 exact** | **29 exact / 1 near** |
| hotel-lights | 27 exact / 3 truncated | **30 exact, 0 truncated** | (not run: decode-dominated) |
| Retrieval 128K / 900K | pass / pass | **pass / pass** | **pass / pass** |
| Long generation @826K | 1 failure in 12, worst clean repeat 0.0151 | **4/4 clean**, worst 0.0052 | **6/6 clean**, worst 0.0092 |

The 30/30 scores are partly sampling luck at n=30; the supported claim is no
detectable harm. The FP8-prefill near answer (`73, 46.5`) is the same near
answer 20260921 produces. Prefill numerics were also measured directly with
teacher-forced prompt log-likelihood: two identical BF16-prefill runs already
differ by ~0.10 nats per token (per-restart kernel autotuning); FP8 prefill adds
3-5% to that, moves code NLL within the control spread and prose NLL +0.36%.
The earlier qualification of the decoder-projection conversion alone scored
lavd 28/2, hotel-lights 28 exact / 1 wrong / 1 truncated, long generation 4/4
([qualification-fp8-v1.md](qualification-fp8-v1.md)). Full details:
[qualification-fp8-v2.md](qualification-fp8-v2.md), [fp8-prefill.md](fp8-prefill.md).

The 20260921 notes on long-context reasoning still apply: at `reasoning_effort:
max` on very long prompts the model can occasionally spend its whole budget
thinking, so allow a generous `max_tokens`.

## Scope of qualification

Only **default checkpoint / MTP3 / DCP1** -- the release defaults -- was
qualified on this image. The other seven 20260921 variations run the new code
paths untested here:

| Variation | 20260921 | New paths it would use |
| --- | --- | --- |
| uncensored checkpoint | qualified | FP8 applies only to layers still unquantized BF16 with the exact shapes above; untested |
| DFlash2 (depth 7, graph 64) | qualified | FP8 target layers, PDL, cutoff (graph-64 verify batches exceed 32 rows at C8, so those run the prefill path); untested |
| MTP0 | qualified (graph fix) | FP8 target layers, PDL, cutoff; untested |
| DCP3 | qualified | FP8, PDL, cutoff with sharded KV; untested |

To run a non-default variation with 20260921 numerics, set
`VLLM_GLM53_FP8_DENSE=0` and an empty `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=`
(PDL stays on; it does not change numerics).

### Reverting individual changes

| Variable | Default | Off |
| --- | --- | --- |
| `VLLM_GLM53_FP8_DENSE` | `1` | `0`: all FP8 paths off (BF16 as 20260921) |
| `VLLM_GLM53_FP8_LM_HEAD` / `_FFN` / `_MTP` | `1` | `0`: that part stays BF16 |
| `VLLM_GLM53_FP8_PREFILL` | `w8a8` | `bf16`: prefill keeps BF16 weights (KV 2,989,001) |
| `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE` | `256KB` | empty: upstream 84 KiB |
| PDL | built in | `GLM53_IMAGE=azallaza/glm53-kraken-tp3:20260921` (not separately switchable; neutral numerically) |

## Investigated and not adopted

Recorded so they are not repeated ([decode-sweep.md](decode-sweep.md),
[decode-profile.md](decode-profile.md) and the linked documents):

- Reducing attention-head padding 72 -> 66: the B12X extend kernel requires
  heads divisible by 8, and MLA attention is only ~2.5% of a decode step.
- 400 W per GPU: tested by the user earlier; negligible gain, much higher temperatures.
- Inductor compilation (`mode: 3`) to unlock b12x's fused all-reduce + RMSNorm:
  -4.8% C1, and the fusion still did not register.
- MoE backends: FlashInfer CuTe DSL is SM100-only; vLLM CUTLASS NVFP4 supports
  SM120 but refuses EP3, which TP3 requires.
- The fork's `VLLM_MXFP8_LM_HEAD` switch: -0.8% C1 (b12x MXFP8 a16 kernel);
  the Marlin FP8 head in this release replaces it.
- L2 prefetch budgets resized for FP8 weights: no gain.
- PCIe two-shot all-reduce: world size 4 only. Async scheduling: already on.

## Not done

- **A TP3 push transport for the one-shot all-reduce.** TP2's LL-style push
  measures 8.96 us at 32 KiB against 16.1 us for TP3's pull; a three-rank
  version is estimated at ~3% C1 but needs new b12x IPC layout, kernel ABI and
  a stress harness.
- **Recovering the KV gap.** Running prefill on the Marlin copy alone would free
  another ~2 GiB/GPU (above 20260921's capacity) at ~3.5% slower prefill; about
  1 GiB of the memory freed by FP8 prefill also does not reach the KV pool.

## Example Compose

Download [compose.yaml](compose.yaml) and
[serve-glm53-flash-tp3-kraken.py](serve-glm53-flash-tp3-kraken.py) into one
directory. Set `HF_CACHE_DIR` to a populated Hugging Face cache (offline loading
is enabled) and adjust `CPUSET` for your host.

```bash
export HF_CACHE_DIR=/path/to/huggingface
docker compose up -d
# Previous numerics on this image (all new paths off except PDL):
VLLM_GLM53_FP8_DENSE=0 VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE= docker compose up -d
```

Stop active inference before switching configurations.

```yaml
# Kraken TP3 release example, image azallaza/glm53-kraken-tp3:20260923.
# Defaults: the qualified default/MTP3/DCP1 configuration with FP8 decode,
# FP8 prefill and the 256 KiB one-shot cutoff. PDL and all source corrections
# are built into the image; only the launcher is mounted.
# GPU power is a host setting: tested350W/GPU; never exceed400W.
services:
  glm53-kraken-tp3:
    image: "${GLM53_IMAGE:-azallaza/glm53-kraken-tp3:20260923}"
    container_name: glm53-kraken-tp3
    restart: "no"
    ipc: host
    shm_size: "64gb"
    cpuset: "${CPUSET:-8-47}"
    gpus: all
    ports:
      - "15015:8000"
    volumes:
      - ${HF_CACHE_DIR:?Set HF_CACHE_DIR to your populated Hugging Face cache}:/root/.cache/huggingface:ro
      - runtime-cache:/cache
      - ./serve-glm53-flash-tp3-kraken.py:/usr/local/bin/serve-glm53-flash-tp3-kraken.py:ro
    environment:
      # Set here, in .env, or before `docker compose up -d`.
      # default = released NVFP4; uncensored = orcarouter NVFP4 checkpoint.
      # Both checkpoints are qualified in all four mode/DCP combinations;
      # see release/kraken-20260921/VARIATIONS.md.
      CHECKPOINT: "${CHECKPOINT:-default}"
      MODEL: "${MODEL:-}"
      MODEL_REVISION: "${MODEL_REVISION:-}"
      SERVED_MODEL_NAME: "${SERVED_MODEL_NAME:-GLM-5.3-Flash-TP3}"

      # mtp = depth3; dflash2 (or dflash) = depth7; mtp0 = speculation off.
      # Draft arguments are added only in DFlash mode.
      MODE: "${MODE:-mtp}"
      DFLASH_MODEL: "${DFLASH_MODEL:-local-inference-lab/GLM-5.3-Flash-DFlash2}"
      DFLASH_MODEL_REVISION: "${DFLASH_MODEL_REVISION:-}"

      # DCP1 = measured speed baseline; DCP3 = shard KV across all3 GPUs.
      # DCP3 is qualified; its rank-padding correction is baked into the image.
      DCP: "${DCP:-1}"
      MAX_NUM_BATCHED_TOKENS: "${MAX_NUM_BATCHED_TOKENS:-4096}"
      # Empty chooses32 for MTP/off or64 for DFlash (covers C8 verification).
      MAX_CUDAGRAPH_CAPTURE_SIZE: "${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
      # Empty: finer MTP/off captures for default/DCP1, graph32, 8 slots.
      # Other settings retain their ladder; explicit space-separated lists override.
      CUDAGRAPH_CAPTURE_SIZES: "${CUDAGRAPH_CAPTURE_SIZES:-}"
      MAX_NUM_SEQS: "${MAX_NUM_SEQS:-8}"
      MAX_MODEL_LEN: "${MAX_MODEL_LEN:-1048576}"
      GPU_MEMORY_UTILIZATION: "${GPU_MEMORY_UTILIZATION:-0.95}"
      REASONING_EFFORT: "${REASONING_EFFORT:-max}"
      CLEAR_THINKING: "${CLEAR_THINKING:-true}"

      # 256KB keeps C8's 256 KiB all-reduces on the b12x one-shot instead of
      # the NCCL ring: +3.0% C8 steps/s, neutral at C1 (measured 2026-09-23 on
      # the FP8 stack). Empty restores the upstream 84KiB cutoff.
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: "${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE-256KB}"

      # KDA backends. The Kraken profile ships kda_prefill_backend=b12x, which
      # `resolve_kda_prefill_backend` refuses to select under `auto` because it
      # is not serving-qualified. It degrades the recurrent state built across
      # very long prompts: on an 826K history it cost uncensored/DFlash2 one
      # failure in four, and held uncensored/MTP3 to 38% of the default
      # checkpoint's output length. flashkda -- what R34 served in all 429
      # recorded receipts -- takes both arms to 4/4 at full length.
      # Decode stays on b12x, so this remains a b12x build.
      # See qualification-20260920/kda-prefill-backend/RESULTS.md.
      # The image's profile still ships kda_prefill_backend=b12x and is not
      # patched: /opt/lil/image-contract.json pins the profile hashes and the
      # entrypoint refuses to start if they change. So the qualified backend is
      # selected here. Decode stays b12x; only recurrent prefill moves.
      ADDITIONAL_CONFIG: "${ADDITIONAL_CONFIG:-{\"glm53_kda_decode_backend\":\"b12x\",\"kda_prefill_backend\":\"flashkda\"}}"
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
      VLLM_NO_USAGE_STATS: "1"
      # 1 asserts the resolved backend set matches the qualified one. The
      # baked vllm/v1/worker/utils.py accepts either b12x or flashkda for
      # prefill and still requires b12x collectives, EP3 and b12x KDA decode,
      # so this stays on for every supported configuration.
      GLM53_TP3_REQUIRE_RUNTIME_PROOF: "${GLM53_TP3_REQUIRE_RUNTIME_PROOF:-1}"

      # MTP acceptance-length adaptation. Empty = disabled, the qualified
      # configuration. A positive integer averages accepted draft lengths over
      # that many verification steps and trims the speculative-token count,
      # with depth 3 as the upper bound. Measured 0.0% at C1, where the
      # controller never trims; under evaluation at C8.
      MTP_ADAPTIVE_WINDOW: "${MTP_ADAPTIVE_WINDOW:-}"
      # FP8 weight-only decode (Marlin W8A16) for the large BF16 projections:
      # KDA in_proj/o_proj, DSA o_proj/q_b_proj, dense FFN, LM head, MTP draft.
      # +11.4% C1 / +8.1% C8 steps/s with PDL and the cutoff. 0 = BF16 as in
      # 20260921. Sub-flags switch parts off individually.
      VLLM_GLM53_FP8_DENSE: "${VLLM_GLM53_FP8_DENSE:-1}"
      VLLM_GLM53_FP8_LM_HEAD: "${VLLM_GLM53_FP8_LM_HEAD:-1}"
      VLLM_GLM53_FP8_FFN: "${VLLM_GLM53_FP8_FFN:-1}"
      VLLM_GLM53_FP8_MTP: "${VLLM_GLM53_FP8_MTP:-1}"
      # w8a8: prefill on CUTLASS FP8, BF16 copies dropped (+5.5-6.5% prefill,
      # +6% KV). bf16: prefill keeps the BF16 weights (more memory, unchanged
      # prefill numerics). Only read when VLLM_GLM53_FP8_DENSE=1.
      VLLM_GLM53_FP8_PREFILL: "${VLLM_GLM53_FP8_PREFILL:-w8a8}"
      # Half-L2 budgets retained from R34; re-tested with FP8 weights and kept.
      VLLM_GLM53_L2_PREFETCH_BUDGET_A_MB: "10"
      VLLM_GLM53_L2_PREFETCH_BUDGET_B_MB: "25"
      VLLM_GLM53_L2_PREFETCH_BUDGET_C_MB: "7.5"
      VLLM_GLM53_L2_PREFETCH_BUDGET_A_MLA_MB: "18"
      VLLM_GLM53_L2_PREFETCH_PERSIST_MB: "${VLLM_GLM53_L2_PREFETCH_PERSIST_MB:-0}"
      VLLM_GLM53_L2_PREFETCH_A_NEXT_MB: "${VLLM_GLM53_L2_PREFETCH_A_NEXT_MB:-0}"
    entrypoint: ["/opt/venv/bin/python", "/usr/local/bin/serve-glm53-flash-tp3-kraken.py"]
    command: []

volumes:
  runtime-cache:
```
