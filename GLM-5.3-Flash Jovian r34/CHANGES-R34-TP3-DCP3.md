# GLM-5.3-Flash R34 → TP3/EP3/DCP3: complete change list

Everything applied on top of the published R34 image to serve GLM-5.3-Flash
NVFP4 on three RTX PRO 6000 Blackwell GPUs (TP3, EP3) with decode-context
parallelism across the three ranks (DCP3). One document, self-contained; the
detailed receipts live in `PORT-CHANGES.md`, `PATCH-LEDGER.md`, and `STATUS.md`.

## 1. Base and build method

| Item | Value |
| --- | --- |
| Parent image | `localinferencelab/vllm@sha256:d2d13141fc158f3e5f989930c4be4637eaf28322307288724c2faa7cd4e9bcc7` (`jovian-judgement-community-20260910-r34`) |
| vLLM source lock | `c496604123b1f4441007b952a7ee37ab12c8f6ad` |
| B12X source lock | `59d51a36a942d56a9c36265855cdc7856fa7712e` |
| Source lock SHA-256 | `e7b5712d12676c8daf0a000398cfa2d57eedb3e290cedf611fcee28fe2413dd0` |
| Build | `prepare-build.py` diffs the working trees `r34-port/vllm` and `r34-port/b12x` against the pristine image sources and copies only changed `*.py` (and changed `policy/_profiles/data/*.json.gz`) into `build/overlay/`; `Dockerfile` is `FROM <parent digest>` + `COPY overlay/ /` + `COPY source-delta.json /opt/glm53-flash/tp3-source-delta.json`. No native library is replaced; ten active vLLM native libraries hash-match the parent. |
| Cache namespace | Launcher derives `LOCAL_INFERENCE_CACHE_FINGERPRINT=r34-tp3-vllm26948c19-b12xf5366673-<delta16>` from the shipped source-delta hash, so JIT/autotune caches under `/cache/jit/` are per build. |

## 2. Why TP3 needs source changes at all

The released checkpoint has 64 attention heads, 64 KV heads, 64 KDA heads,
288 routed experts and a 2,048-wide expert intermediate. None of the head
counts divide by 3, so upstream (TP4/TP8 only) cannot shard it. The port keeps
the checkpoint's logical sizes and introduces a *physical* TP3 geometry:

| Logical | Physical TP3 | Per rank |
| --- | ---: | ---: |
| 64 attention / KV heads | 72 | 24 |
| 64 KDA heads | 66 | 22 |
| shared-expert intermediate 2048 | 2112 | 704 |
| MTP projection 4096 | 4098 | 1366 |
| vocabulary 154,880 | 154,944 (padding 192) | 51,648 |
| routed experts 288 | 288 (expert parallel) | 96 |
| vision heads 16 / intermediate 4096 / projection 10240 (weights mode) | 18 / 4098 / 10242 | — |

Padded rows/columns are zero-filled and semantically inert. Routed experts
stay whole and are split by **expert parallelism (EP3)** because 2048 is not
divisible by 3; the TP3 config guard enforces `--enable-expert-parallel`.

## 3. Source files in the overlay

### vLLM (`/opt/glm53-flash/vllm/vllm/…`)

| File | Change |
| --- | --- |
| `transformers_utils/configs/glm53_tp3.py` | **New.** Detects GLM-5.3 at TP3, validates the released shapes, and rewrites the HF config to the physical geometry above (`num_attention_heads=72`, `num_key_value_heads=72`, `linear_num_heads=66`, `glm53_tp3_shared_expert_intermediate_size=2112`, `glm53_tp3_mtp_projection_size=4098`, vocab storage/padding, vision sizes). Requires EP for TP3. Also the draft (DFlash) geometry variant. |
| `model_executor/layers/linear.py` | `loaded_output_size(s)` / `loaded_input_size` on Column/Row/MergedColumnParallelLinear: load the checkpoint's logical shard into a larger physical shard and zero the tail. |
| `model_executor/parameter.py` | Parameter shard helpers aware of logical vs physical sizes and quantized storage units (packed FP4 bytes, block scales). |
| `model_executor/layers/quantization/modelopt.py` | ModelOpt NVFP4/MXFP8 linear and MoE methods: logical-to-storage factors for padded output widths, padded shared-expert intermediate, block-aligned output padding restored to logical width. |
| `model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` | Same storage-factor handling for the compressed-tensors NVFP4 scheme. |
| `model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` | KDA layer at 22 physical heads/rank: padded `in_proj_qkvgfab` split sizes, `f_b_proj`/`g_b_proj`/`o_proj` logical sizes, replicated per-head scalar shard, B12X KDA decode/FlashKDA prefill plans for the TP3 geometry; align-mode Mamba block retirement fix. |
| `models/glm5next/nvidia/attention.py` | Sparse MLA (DSA) attention at 24 heads/rank: padded `q_b_proj`/`kv_b_proj`/`o_proj` with logical loaded sizes; TP3-safe q/kv head bookkeeping. |
| `models/glm5next/nvidia/model.py` | Model wiring for the physical geometry: shared-expert `Glm5NextMLP` with `loaded_intermediate_size`, EP topology, TP3 config application, runtime-proof hook, padded vocabulary. |
| `models/glm5next/nvidia/mtp.py` | MTP module: padded `eh_proj` (4098), participates in the target EP topology, draft-only NVFP4 vocabulary head over the padded vocabulary. |
| `models/glm5next/nvidia/multimodal.py` | Vision encoder in weights mode at 18 heads / 4098 / 10242 with logical loading. |
| `model_executor/models/qwen3_dflash.py`, `qwen3_dflash2.py` | DFlash draft (36 q heads / 9 KV heads / padded vocab) loading; draft stays dense and outside the target EP group. |
| `config/speculative.py` | Propagates TP3/DCP settings to the draft parallel config; TP3 gating for MTP/DFlash; draft geometry hook. |
| `config/vllm.py` | Early TP3 geometry setup before model construction; TP3 gating. |
| `distributed/device_communicators/cuda_communicator.py` | Allows world size 3 for the B12X PCIe one-shot all-reduce dispatch (TP group); disabled-B12X RoCE fallback. |
| `v1/core/kv_cache_utils.py` | Split GLM-5.3 KV cache groups (target pages / recurrent state) with TP3 shapes; group rebalancing. |
| `v1/engine/core.py`, `v1/worker/gpu/model_runner.py`, `v1/worker/gpu_model_runner.py` | Engine/runner integration: geometry setup, boundary-state restore fix for large offsets, runtime proof emission. |
| `v1/worker/utils.py` | `GLM53_TP3_RUNTIME_PROOF`: with `GLM53_TP3_REQUIRE_RUNTIME_PROOF=1` the worker asserts the resolved backends (`b12x_pcie_oneshot`, EP size 3, KDA decode `b12x`, KDA prefill `flashkda`, vision `weights`) and logs them. |
| `v1/spec_decode/dflash.py` | DFlash proposer with the padded draft geometry. |
| `parser/engine/parser_engine.py` | Explicit `tool_choice="none"` honored without tool schemas (narrow local fix; 3,009 replay tests). |
| `v1/attention/ops/dcp.py` | **DCP3 fix.** `_correct_attn_cp_out_kernel` used the DCP world size as a Triton `tl.arange` extent, which must be a power of two, so world size 3 crashed in the DCP profiling pass. The kernel now takes `N` plus `N_ROUNDED = next_power_of_2(N)` and masks lanes ≥ N with `-inf` in the log-sum-exp combine. Verified against a torch reference for world sizes 2/3/4/6 including empty shards. Inert at DCP=1. |

### B12X (`/opt/glm53-flash/b12x/b12x/…`)

| File | Change |
| --- | --- |
| `comm/pcie/pcie_dma.py`, `comm/pcie/pcie_oneshot.py` | World size 3 added to the supported sets (kernel math and 64-bit addressing unchanged); `B12X_PCIE_ONESHOT_THREADS=512` measured best for three-rank decode messages. |
| `policy/generation/attention_corpus.py`, `moe_corpus.py`, `providers/blockscaled.py` | TP3 corpus geometry (22 KDA heads, 704-wide shared/routed intermediate alias) so GPU profiles can be generated for the TP3 shapes. |
| `policy/_profiles/data/nvidia.rtx.pro.6000.blackwell.json.gz` | Shipped RTX PRO 6000 profile with the **`attention.gdn` component regenerated** for the TP3 KDA geometry (1,468 cases, six TP3 serving cases; all 20 other components byte-identical to the parent's). |

### Launchers (`/usr/local/bin/`)

| File | Change |
| --- | --- |
| `serve-glm53-flash.sh` | Parent dispatcher with one added branch: `TP=3` → `exec serve-glm53-flash-tp3-r34.sh`. TP4/TP8 body unchanged. |
| `serve-glm53-flash-tp3-r34.sh` | **New TP3 launcher.** Fails closed unless `TP=3`, `DCP∈{1,3}`, GPU-only cache, weights-mode vision, the pinned checkpoint `local-inference-lab/GLM-5.3-Flash-NVFP4` @ `46aaae8a82032f77100f2f03e9cc11b391df3b4d`, MTP depth 0/3 or DFlash2 depth 7 with an immutable draft revision. Sets `GLM53_KDA_DECODE_BACKEND=b12x`, `GLM53_KDA_PREFILL_BACKEND=flashkda`, B12X PCIe all-reduce, `B12X_PCIE_ONESHOT_THREADS=512`, `MOE_BACKEND=auto` (R34's default `b12x` MoE rejects EP3; auto selects FlashInfer CUTLASS NVFP4), `GLM53_TP3_REQUIRE_RUNTIME_PROOF=1`, per-build cache namespace, configurable `MAX_NUM_BATCHED_TOKENS` / `MAX_CUDAGRAPH_CAPTURE_SIZE` / `CUDAGRAPH_CAPTURE_SIZES` (capture list adapts to the maximum). Re-enables FlashInfer autotune (the parent launcher passes `--no-enable-flashinfer-autotune`; appended `--enable-flashinfer-autotune` wins). Then execs the parent `serve-glm53-flash-nvfp4-dflash2.sh` with `--enable-expert-parallel --mm-encoder-tp-mode weights`, `clear_thinking=true` chat default and `temperature 1 / top_p 0.95`. |

## 4. Compose (`compose.yaml`, `release/compose.yaml`)

Single service, inline command block. `MODE` selects `mtp` (MTP3, default),
`mtp0`, or `dflash2` (pinned incoai BF16 DFlash7 draft `dc77ff1c…`). Exports
`TP=3`, `DCP="${DCP:-3}"`, `MAX_MODEL_LEN=1048576`, `MAX_NUM_SEQS=8`,
`GPU_MEMORY_UTILIZATION=0.95`, `CACHE_MODE=vram`; env knobs
`MAX_NUM_BATCHED_TOKENS` (4096), `MAX_CUDAGRAPH_CAPTURE_SIZE` (32),
`CUDAGRAPH_CAPTURE_SIZES`, `ENABLE_PREFIX_CACHING`, and the half L2-prefetch
budgets `VLLM_GLM53_L2_PREFETCH_BUDGET_{A,B,C,A_MLA}_MB = 10/25/7.5/18`
(measured +2% decode). The local file bind-mounts the Hugging Face cache, the
persistent `/cache` volume and the local launcher; the release file uses named
volumes and `GLM53_IMAGE`. `DCP=1 docker compose up -d` switches back to the
single-copy KV layout; DCP is a start-time setting.

## 5. DCP3 specifically

- Attention KV (11 sparse-attention layers) is sharded across the three ranks
  with `cp_kv_cache_interleave_size=4`; the 34 KDA layers keep replicated
  recurrent state, as upstream. Logical KV block becomes 6,144 tokens
  (2,048 × 3). DCP-group collectives (query all-gather, LSE all-gather,
  output reduce-scatter per DSA layer) run over PyNCCL; the symmetric-memory
  "direct" paths are unavailable on SM120 and B12X's PCIe DCP kernels support
  world sizes 2/4/8/16 only.
- Capacity: **7,510,219** KV tokens at DCP3 vs **2,091,238** at DCP1 (graph
  max 32; the qualified DCP1 figure at graph max 256 was 1,596,516).
- Cost: about −8% verifier steps/s at 0–128K context, −5% at 256K, −1 to −3%
  at 512K; crossover near 1M tokens. `--dcp-comm-backend a2a`, NCCL channel
  and protocol knobs were measured and are not better.

## 6. Qualification on the TP3/DCP3 build

- 3,191 component tests (TP3 geometry/loader/vision/corpus, sparse Mamba
  allocator, collective dispatch, parser replay) plus three-GPU boundary-state
  and collective eager/graph/torture tests.
- DCP1: MTP0, DFlash7, MTP3 exact-answer smokes; MTP3 concurrency, OCR, 17.6K
  retrieval; exact retrieval at 127,992 and 899,994 prompt tokens; retained
  history 826,266 prompt → 28,883 completion tokens, normal stop.
- DCP3: smokes, concurrency, OCR, 17.6K/128K/900K exact retrieval; retained
  history 826,266 → 27,592 tokens, normal stop, clean content channel.

## 7. Image identities

| Build | Tag | Image ID | Overlay | Delta SHA-256 |
| --- | --- | --- | ---: | --- |
| TP3 port (DCP1 only) | `glm53-r34-tp3:ported-20260910` | `d391f61f66e7` | 30 files | `1b1d1c379ef7cb3ff8f47e245633c188a9ab1c7fc98b86bc7d38faa59905b693` |
| **TP3 + DCP3** | `glm53-r34-tp3:dcp-20260910` | `34d5ad37b8bb` | 31 files (+ `v1/attention/ops/dcp.py`, launcher `DCP=1|3`) | `7447b24601a4ad8a866790479b99fb3e95415ef4037111b176fa3ba95e4b155c` |

No registry publication has been performed.
