# GLM-5.3 Flash Kraken TP3 — 20261015

This model is based on [GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) by Local Inference Lab, Inc., a non-profit organization, available at <https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD>. GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0.

Image: **`azallaza/glm53-kraken-tp3:20261015`** (local `glm53-kraken-tp3:20261015`, `sha256:c62012f9a7f1ea5dbef978bfe3d6234a7d76f22e8532000936f08eda3e6ff78a`).
Built on top of `azallaza/glm53-kraken-tp3:20261008` (`sha256:44c2d27ec026…`): every 20261008 layer is
reused, so the pull is only the changed files. Same weights, checkpoints and modes as 20261008; the default is now **32 slots** (was 8)
(see `../kraken-20261008/README.md` for the CSF-QAD details, DCP3, DFlash2 and the other checkpoints).

```bash
docker pull azallaza/glm53-kraken-tp3:20261015
HF_CACHE_DIR=/path/to/huggingface docker compose up -d                  # csf-qad-2, MTP3/DCP1, 32 slots, 3.96x 1M
HF_CACHE_DIR=/path/to/huggingface DCP=3 docker compose up -d            # DCP3, 32 slots, 10.0x 1M
HF_CACHE_DIR=/path/to/huggingface MAX_NUM_SEQS=8 docker compose up -d   # 8 slots (20261008 default), 4.01x 1M
```

Host settings: 350 W/GPU (never exceed 400 W), +6000 memory offset.

## What changed since 20261008

| Change | Effect (csf-qad-2, 350 W, 32K context) |
| --- | --- |
| **Small-row MXFP8 GEMV** for decode steps of <= 8 rows (C1/C2 with MTP3, and the MTP draft up to 8 streams). Reads the packed MXFP8 weights in place (exact E4M3 -> BF16 conversion, scale folded in), compiled at load. `VLLM_GLM53_MXFP8_GEMV=0` turns it off | **+5.5% C1** decode steps/s (113.6 vs 107.6); C8 neutral |
| MTP CUDA graph maximum follows `MAX_NUM_SEQS` (slots x (draft + 1)); was fixed at 32 rows (the 2026-10-13 launcher fix, now built in). Found by [shihanqu/RTX-Pro-6000-Recipes](https://github.com/shihanqu/RTX-Pro-6000-Recipes/tree/main/glm53-kraken-tp3-dcp3-zen) | above 8 slots steps keep CUDA graphs (C12 160 -> 999 tok/s) |
| **Finer graph ladder above 32 rows**: every 8 rows (2 MTP3 streams) up to 128, instead of 48/64/96/128 | **+10% C20, +11% C28** tok/s (no padding to the next graph); C16/C32 unchanged |
| KV budgets for csf-qad* at `MAX_NUM_SEQS=32` | DCP1 **4,152,888 (3.96x 1M)**, DCP3 **10,485,760 (10.0x 1M)** |
| One-shot all-reduce cutoff 256 KB -> **1 MB** (compose default and image ENV) | decode all-reduces up to 32 slots stay on the b12x one-shot: C16 78.9 -> 40.7 us/call, C32 110 -> 79.4 us (~+1%) |
| `DRAFT_TOKENS` other than 3: graph ladder at whole stream counts | e.g. `DRAFT_TOKENS=2` at 32 slots: +3.4% C8 / +2.5% C32 tok/s, slower at C1. Not qualified; adjust to your load |

`source.patch`: the 4 files that differ from the 20261008 image (`glm53_mxfp8_gemv.py` new, `mxfp8/b12x.py`,
`b12x_blockscaled.py`, the launcher). With `MAX_NUM_SEQS=8` the command line is the 20261008 one plus the GEMV.

**Default slots 8 -> 32**: up to 32 concurrent streams with CUDA graphs. Costs ~1% C1 and 47K KV tokens at DCP1
(4,199,951 -> 4,152,888); DCP3 stays 10,485,760 (10.0x 1M). `MAX_NUM_SEQS=8` restores the 20261008 sizing.

## Measurements on this image (32 slots, 350 W, 32K, llm_decode_bench 30 s cells)

| | C1 | C8 | C16 | C20 | C28 | C32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode steps/s | 113.8 | 313.1 | — | — | — | 615.7 |
| aggregate tok/s | 282 | 822 | 1,176-1,225 | 1,287-1,300 | 1,551-1,588 | 1,640-1,677 |
| 20261008 + 10-13 launcher, tok/s | 263-273 | 802 | 1,155-1,223 | 1,170 | 1,410-1,433 | 1,633-1,689 |

C16-C28 from the ladder A/B on the same configuration (GEMV overlay = this image's code).

## Quality and stress (this image's code: GEMV + 32 slots + fine ladder + 1 MB)

| Test | 20261008 | **20261015** |
| --- | --- | --- |
| lavd, 30 runs C8 | 30 exact | **30 exact** |
| lavd, 30 runs C2 (target model on the GEMV) | — | **30 exact** |
| hotel-lights, 30 runs C8 (seed 103) | 28 exact / 1 wrong / 1 truncated | **27 exact / 3 wrong** |
| hotel-lights, 30 runs C2 (target model on the GEMV) | — | **28 exact / 2 wrong** |
| Retrieval 128K / 900K | pass / pass (20261007) | **pass / pass** |
| Long generation at 826K, seeds 201-204 | 4/4 clean, worst 8-gram repeat 0.0044 (20261007) | **4/4 clean, worst 0.0017** |
| Arithmetic | 8/8 | 8/8 |

hotel-lights has ranged 27-30 exact across earlier qualified configurations; at C8 the GEMV only serves the
MTP draft, which cannot change the output distribution under standard rejection sampling.

| Stress, 32 slots, fine ladder (DCP1 2190 blocks unless noted) | Result | Min free |
| --- | --- | ---: |
| 32 concurrent x 10 photos 6000x4000 | 32/32 OK, 0 preemptions | 1.69 GB |
| 3 x ~890K + 29 x ~14K prompts, 24K/16K decode | 32/32 OK, 0 preemptions, peak KV 74% | 1.69 GB |
| DCP3 2145 blocks (10,485,760 KV): 32 concurrent x 10 photos 6000x4000 | 32/32 OK, 0 preemptions | 2.24 GB |
| DCP3: 9 x ~890K + 23 x ~14K prompts (9.18M prompt tokens) | 32/32 OK, 0 preemptions | 2.24 GB |

The 8-slot budgets (DCP1 2215 blocks / DCP3 2145) and their stress results are unchanged from 20261008.

## Known issues

- `MTP_ADAPTIVE_WINDOW` does not work with csf-qad-2 on this image (crashes on the first request, then fails
  b12x preparation on restart). Leave it empty (default).

## Not re-run on this image

DCP3 quality, DFlash2, the default/uncensored checkpoints, vision tests, LMCache L1 (unchanged code paths;
results in the 20261008 README). The GEMV also serves those configurations at <= 8 rows.

## Files

- `Dockerfile` (FROM 20261008), `src/` (the 3 changed vLLM files), `serve-glm53-flash-tp3-kraken.py`
  (built in and mounted by the compose), `compose.yaml` (below), `source.patch`, `image-inspect.json`,
  `build.log`, `bench/` (C1/C8/C32 on the built image).

## Variables

Same as 20261008, plus:

| Variable | Default | Values |
| --- | --- | --- |
| `VLLM_GLM53_MXFP8_GEMV` | `1` | `1`, `0` (b12x dense GEMM for every row count) |
| `MAX_NUM_SEQS` | `32` (was `8`) | `32` (graph 128), `8` (graph 32, ~1% faster C1, +47K KV): csf-qad* KV budgets for both; other values use automatic KV sizing |
| `DRAFT_TOKENS` | empty (= 3 MTP / 7 DFlash2) | `1`-`7`; only the defaults are qualified |
| `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE` | `1MB` | size; empty = upstream 84 KiB |

```yaml
# Kraken TP3 release example, image azallaza/glm53-kraken-tp3:20261015
# (20261008 + small-row MXFP8 GEMV + 32-slot budgets and finer graph ladder;
# our TP3 port rebased onto upstream karmic-kraken-beta, vLLM 89f1ceecd).
# Defaults: CSF-QAD (dec48abd, quantized vision) / MTP3 / DCP1 / MOE_TP=1, W4A4
# experts with resident scales, 32 slots, KV 4,152,888 tokens (3.96x 1M), top-p 0.95.
# CHECKPOINT=csf-qad serves fd660d51 (BF16 vision); default = original NVFP4.
# All source changes are built into the image; the launcher is also mounted.
# GPU power is a host setting: tested 350W/GPU; never exceed 400W.
services:
  glm53-kraken-tp3:
    image: "${GLM53_IMAGE:-azallaza/glm53-kraken-tp3:20261015}"
    container_name: glm53-kraken-tp3
    restart: "no"
    # Room for LMCache/vLLM to shut down cleanly. A stale LMCache L1 arena left in
    # /dev/shm is removed by the launcher at the next start (lock-protected).
    stop_grace_period: 90s
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
      # csf-qad-2 = local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd:
      #   QAD experts with compressed scales, MXFP8 attention/shared experts,
      #   quantized vision tower (MXFP8 attention, NVFP4 MLP; replicated per GPU).
      # csf-qad = the same checkpoint at fd660d51 (identical decoder, BF16 vision
      #   sharded across the GPUs). Both: LIL License 1.0; MODE=mtp and MOE_TP=1.
      # default = local-inference-lab/GLM-5.3-Flash-NVFP4@175ae8ce (original).
      # uncensored = orcarouter NVFP4 checkpoint (experimental on this image).
      CHECKPOINT: "${CHECKPOINT:-csf-qad-2}"
      # csf-qad/csf-qad-2 only. a4 = W4A4 experts (default, ~10% faster decode than a16,
      # quality-qualified); a16 = upstream's W4A16.
      CSF_ACTIVATIONS: "${CSF_ACTIVATIONS:-a4}"
      # csf-qad/csf-qad-2 only. 1 = expand expert scales once per layer (+3.7% C1, +2% C8,
      # ~2.4 GiB/GPU); 0 = keep them compressed (per-call decode, more free memory).
      CSF_RESIDENT_SCALES: "${CSF_RESIDENT_SCALES:-1}"
      MODEL: "${MODEL:-}"
      MODEL_REVISION: "${MODEL_REVISION:-}"
      SERVED_MODEL_NAME: "${SERVED_MODEL_NAME:-GLM-5.3-Flash-TP3}"

      # mtp = depth3; dflash2 (or dflash) = depth7; mtp0 = speculation off.
      # Draft arguments are added only in DFlash mode.
      MODE: "${MODE:-mtp}"
      # 0 = baseline EP3; 1 = TP-sharded routed experts (704 channels/rank),
      # +2.8-4% C1. MTP only: default checkpoint qualified at DCP1; DCP3 and the
      # uncensored checkpoint (Marlin TP experts) experimental; DFlash/MTP0 need 0.
      MOE_TP: "${MOE_TP:-1}"
      # auto = LM-head BF16 release + shared PyNCCL + explicit KV budget, applied
      # only for MTP3/MOE_TP=1/DCP1 with default slots/batch/graph/length budgets
      # measured on this image: csf-qad* DCP1 2215 blocks (4,199,951 KV; 4x1M text
      # + 8x10 large-image stress), csf-qad* DCP3 2145 blocks (10,485,760 KV, 10.0x),
      # default DCP1 1700 blocks (3,222,912 KV). Other combinations (default DCP3,
      # uncensored, LMCACHE=l1) use automatic sizing. 0 = off everywhere.
      # TP-expert MoE backend (MOE_TP=1): empty = b12x (csf-qad*, required) /
      # flashinfer_cutlass (default) / marlin (uncensored).
      MOE_TP_BACKEND: "${MOE_TP_BACKEND:-}"
      KV_RECLAIM: "${KV_RECLAIM:-auto}"
      # off = GPU prefix cache only; l1 = LMCache host-RAM tier (LMCACHE_L1_GB GiB,
      # must fit the host's /dev/shm; ipc: host). Reloads evicted prefixes from RAM
      # (100K prefix: 8.5 s -> 0.24 s); KV budget not re-measured on this image.
      # The LMCache disk tier (L2) is not tested at TP3 and is rejected.
      LMCACHE: "${LMCACHE:-off}"
      LMCACHE_L1_GB: "${LMCACHE_L1_GB:-96}"
      # Optional explicit KV bytes/GPU for the budgeted configurations (empty = tested budget).
      KV_CACHE_MEMORY_BYTES: "${KV_CACHE_MEMORY_BYTES:-}"
      # b12x tensor-core BF16 GEMV for decode BF16 linears <=8 rows (C1): ~+2.6-3%
      # C1 steps/s, C8 unchanged, quality-qualified 2026-09-27. 0 = cuBLAS as before.
      VLLM_B12X_BF16_GEMV: "${BF16_GEMV:-1}"
      VLLM_USE_FASTOKENS: "${FASTOKENS:-1}"
      DFLASH_MODEL: "${DFLASH_MODEL:-local-inference-lab/GLM-5.3-Flash-DFlash2}"
      DFLASH_MODEL_REVISION: "${DFLASH_MODEL_REVISION:-}"

      # DCP1 = measured speed baseline; DCP3 = shard KV across all3 GPUs.
      # DCP3 is qualified; its rank-padding correction is baked into the image.
      DCP: "${DCP:-1}"
      # Native PCIe DCP gather/reduce + fused empty-shard mask; 1 enables, 0 disables.
      # Screened default/MTP3/DCP3, C1: +2.6-3.0% steps/s, no meaningful KV loss.
      # Other modes and long natural-output correctness not yet qualified for this option.
      # Included in this image; larger batches retain existing collectives.
      VLLM_TP3_PCIE_DCP: "${NATIVE_DCP:-1}"
      MAX_NUM_BATCHED_TOKENS: "${MAX_NUM_BATCHED_TOKENS:-4096}"
      # Empty: 32 for MTP/off (MTP above 8 slots: MAX_NUM_SEQS x (DRAFT_TOKENS + 1)) or 64 for DFlash.
      MAX_CUDAGRAPH_CAPTURE_SIZE: "${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
      # Empty: finer MTP/off captures for default and csf-qad* at DCP1, graph32, 8 slots;
      # above 8 MTP3 slots every 8 rows (2 streams) to the maximum (+10% C20, +11% C28 vs
      # the 20261008 48/64/96/128 ladder). Other settings retain their ladder; explicit
      # space-separated lists override (and turn off the KV budget).
      CUDAGRAPH_CAPTURE_SIZES: "${CUDAGRAPH_CAPTURE_SIZES:-}"
      # 32 (default) or 8: csf-qad* MTP3 has measured KV budgets for both. 32 slots:
      # graph 128, DCP1 4,152,888 (3.96x 1M) / DCP3 10.0x; 8 slots: graph 32, DCP1
      # 4,199,951 (4.01x) / DCP3 10.0x, ~1% faster C1, at most 8 concurrent streams.
      # Other values use automatic KV sizing.
      MAX_NUM_SEQS: "${MAX_NUM_SEQS:-32}"
      MAX_MODEL_LEN: "${MAX_MODEL_LEN:-1048576}"
      GPU_MEMORY_UTILIZATION: "${GPU_MEMORY_UTILIZATION:-0.95}"
      REASONING_EFFORT: "${REASONING_EFFORT:-max}"
      CLEAR_THINKING: "${CLEAR_THINKING:-true}"

      # 1MB keeps decode all-reduces up to 32 slots (128 rows x 4096 BF16 = 1 MiB)
      # on the b12x one-shot instead of the NCCL ring: C16 78.9 -> 40.7 us/call,
      # C32 110 -> 79.4 us (~+1% steps/s); 256KB (20261008) covered C8 only.
      # Prefill-size all-reduces stay on NCCL. Empty = upstream 84KiB.
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: "${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE-1MB}"
      # Small-row MXFP8 GEMV for decode steps of <= 8 rows (C1/C2 with MTP3, and
      # the MTP draft up to 8 streams): reads the packed MXFP8 weights in place,
      # +5.5% C1 steps/s, C8 neutral. 0 = the b12x dense GEMM for every row count.
      VLLM_GLM53_MXFP8_GEMV: "${VLLM_GLM53_MXFP8_GEMV:-1}"

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
      # prefill and requires b12x collectives and b12x KDA decode.
      # Expert proof checks EP3 normally, or actual TP3 partitioning with MOE_TP=1.
      GLM53_TP3_REQUIRE_RUNTIME_PROOF: "${GLM53_TP3_REQUIRE_RUNTIME_PROOF:-1}"

      # MTP acceptance-length adaptation. Empty = disabled, the qualified
      # configuration. A positive integer averages accepted draft lengths over
      # that many verification steps and trims the speculative-token count,
      # with DRAFT_TOKENS as the upper bound. Measured 0.0% at C1. Not working on
      # this image with csf-qad-2: window 32 at 32 slots crashed on the first request
      # and then failed b12x preparation on restart (LM-head bf16_gemv m3). Leave empty.
      MTP_ADAPTIVE_WINDOW: "${MTP_ADAPTIVE_WINDOW:-}"
      # Speculative tokens per step. Empty = mode default (MODE=mtp 3, dflash2 7);
      # 1-7 accepted. Qualified: MTP 3 and DFlash2 7. Other MTP values get a graph
      # ladder at whole stream counts (rows = streams x (tokens + 1)) and automatic
      # KV sizing (no reclaim budget). Measured at MAX_NUM_SEQS=32: DRAFT_TOKENS=2
      # tok/s +3.4% C8 / +2.5% C32 vs 3, but slower at C1. Adjust to your load.
      DRAFT_TOKENS: "${DRAFT_TOKENS:-}"
      # FP8 weight-only decode (Marlin W8A16) for the large BF16 projections:
      # KDA in_proj/o_proj, DSA o_proj/q_b_proj, dense FFN, LM head, MTP draft.
      # With csf-qad* most of these are MXFP8 in the checkpoint already (b12x);
      # only the dense FFN and LM head remain BF16 and use this path.
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
