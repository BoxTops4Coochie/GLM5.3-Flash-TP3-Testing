# GLM-5.3 Flash Kraken TP3 — 20261007

This model is based on [GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) by Local Inference Lab, Inc., a non-profit organization, available at <https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD>. GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0.

Image: **`azallaza/glm53-kraken-tp3:20261007`** (local `glm53-kraken-tp3:20261007`,
`sha256:b024e106a5448970fda30331cf64a2a1c209225a14e534bbf54c3030cfeaf744`).
Our GLM-5.3-Flash TP3 port (the 20260927 feature set) rebased onto upstream
`ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20261006-c8c8feeaa8cc4a75`
(vLLM `23f2a1830`), plus TP3 support for the **CSF-QAD** checkpoint, which
upstream cannot serve at TP3. Includes fastokens and `lil-bench` (upstream).

```bash
docker pull azallaza/glm53-kraken-tp3:20261007
```

20260927 remains the qualified image for the original checkpoint (about 1.5%
faster at C1 and +96K KV for it); this image is the one for CSF-QAD.

Host settings unchanged: 350 W/GPU (never exceed 400 W), +6000 memory offset.

## What changed since 20260927

| Change | Switch | Effect |
| --- | --- | --- |
| Rebased onto upstream 20261006 (3-way merge of every 20260927 change; one conflict, DFlash head padding, now upstream's) | always | newer vLLM/b12x/FlashInfer/LMCache |
| **CSF-QAD at TP3**: MOE_TP padding (2048 -> 2112, 704/rank) inside the FP4-CSF reader, compressed scale planes padded with zero rows/neutral columns (verified against upstream slicing on real shards, all ranks) | `CHECKPOINT=csf-qad` | upstream refuses TP3 (no EP: unpadded width; EP: CSF needs TP-only) |
| KDA fused input projection padded 8598 -> 8608 rows/rank for b12x MXFP8 (N8), in both KDA paths | csf-qad | required |
| `f_a` copied before the MXFP8 `f_b_proj` (b12x needs 16-byte-aligned input) | csf-qad | required |
| W4A4 routed experts (NVFP4 activations) instead of upstream's W4A16 | `CSF_ACTIVATIONS=a4` (default) | **+10-12% decode**, quality unchanged (see Quality) |
| Resident expanded expert scales (no per-call CSF decode) | `CSF_RESIDENT_SCALES=1` (default) | **+3.7% C1, +2% C8**, ~2.4 GiB/GPU |
| L2 prefetch also covers b12x packed MXFP8 weights (windows were empty before) | always | +1.5-2% C8 (csf-qad) |
| Exact CUDA-graph ladder (1 2 4 8 12 16 20 24 28 32) for csf-qad | always | +2% at C3 |
| KV budgets re-measured on this base: csf-qad DCP1 2215 blocks; default DCP1 1700 blocks (20260927's 1779 runs out of memory in startup autotuning here); others automatic | `KV_RECLAIM=auto` | see KV |
| Marlin MXFP8: FP32-input fix (`VLLM_GLM53_MXFP8_LINEAR=marlin` experiment switch, off) | off | measured slower, not used |
| `MTP_MOE_BACKEND` launcher switch (MTP draft expert backend) | empty = marlin | b12x measured neutral |

## Modes

| Setting (350 W, cooled runs) | KV tokens | C1 steps/s 0/32K/128K | C8 steps/s 0/32K/128K | Status |
| --- | ---: | --- | --- | --- |
| **csf-qad MTP3 DCP1** (default) | **4,199,951 (4.01x 1M)** | 106.6 / 105-108 / 100.4 | 295.5 / 307-315 / 279.0 | qualified (below) |
| csf-qad MTP3 DCP3 | 10,325,560 (9.85x 1M) | 101.6 / 99.7 / 97.6 | 275.0 / 274.8 / 263.5 | 8x1M stress, 8/8 |
| csf-qad, `CSF_ACTIVATIONS=a16` | 4,199,951 | 92.5 / 91.1 / 88.5 | 228.7 / 226.4 / 221.2 | hotel-lights 28/30 |
| default (original NVFP4) MTP3 DCP1 | 3,222,912 | 109.4 / 107.7-110.5 / 104.2 | 304.4 / 300.8 / 284.6 | 8/8, speed |
| default on 20260927, for reference | 3,319,246 | 111.3 / 109.3 / 105.6 | 305.2 / 302.1 / 287.7 | qualified |

Cold prefill (32K / 128K): csf-qad DCP1 10,825 / 10,173 tok/s; DCP3 10,920 / 10,577;
original NVFP4 (lil-bench, 20260927) 12,175 / 11,327. Model memory: csf-qad
58.6 GiB/GPU (60.96 with resident scales) vs 64.46 for the original checkpoint.

```bash
docker compose up -d                                   # csf-qad, MTP3/DCP1, W4A4 + resident scales, 4.01x 1M
DCP=3 docker compose up -d                             # csf-qad DCP3, 9.85x 1M
CSF_ACTIVATIONS=a16 docker compose up -d               # upstream's W4A16 experts (slower)
CSF_RESIDENT_SCALES=0 docker compose up -d             # compressed scales (more free memory, ~-3.7% C1)
CHECKPOINT=default docker compose up -d                # original NVFP4 checkpoint on this image
CHECKPOINT=uncensored docker compose up -d             # uncensored, automatic KV (experimental)
```

## CSF-QAD checkpoint

`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@fd660d51` holds the same
QAD-distilled routed experts as the original checkpoint (`175ae8ce`, via its
`mtp-bf16` branch) with their FP8 block scales losslessly compressed (CSF,
19.0 -> 9.8 GB), attention and shared experts stored as MXFP8, and NVFP4 MTP
experts. Compressed experts run only on the b12x MoE backend, so `csf-qad`
requires `MOE_TP=1` (b12x, no expert parallelism) and `MODE=mtp`. The launcher
prepares the serving directory from the snapshot's `metadata/` (lil runtime).

License: LIL License 1.0 — no reuploads; any README or landing page of a project
that runs it must begin with the attribution notice above.

## Quality (csf-qad, W4A4 + resident scales, MTP3/DCP1)

Protocol identical to the 20260927 qualification (30 runs per test at C8, max
reasoning, top-p 0.95, T=1, 100K cap).

| Test | Original NVFP4 (20260927) | **csf-qad W4A4** |
| --- | --- | --- |
| lavd | 30 exact | **30 exact** |
| hotel-lights | 29 exact / 1 truncated / 0 wrong | **58/60 exact** (28/2 seed 103, 30/0 seed 104) |
| hotel-lights, W4A16 (same seed 103) | — | 28 exact / 2 wrong |
| Retrieval 128K / 900K | pass / pass | **pass / pass** |
| Long generation @826K, seeds 201-204 | 4/4 clean, worst 0.0020 | **4/4 clean, worst 0.0044** |

The hotel-lights misses are short-reasoning errors; W4A16 misses as often on the
same seed, so W4A4 is kept. Evidence: `../../qualification-20261007-csf/`.

## KV budgets and stress

| Configuration | Budget | KV tokens | Stress | Min free |
| --- | --- | ---: | --- | ---: |
| csf-qad DCP1 (default) | 2215 blocks | 4,199,951 (4.01x) | 4 x ~891K + 120K decode, 0 preemptions, peak KV 89% | 1.80 GB |
| csf-qad DCP1 | 2170 blocks | 4,113,354 | same, peak KV 90% | 2.42 GB |
| csf-qad DCP1, compressed scales | 2350 blocks | 4,455,977 | same, peak KV 76% | 2.96 GB |
| csf-qad DCP3 | automatic (27.31 GiB) | 10,325,560 (9.85x) | 8 x ~891K + 120K, 0 preemptions | 2.67 GB |
| default DCP1 | 1700 blocks | 3,222,912 | launch + speed only | — |

DCP3 already exceeds the 8 request slots, so no DCP3 reclaim budget is set.
Uncensored and `LMCACHE=l1` budgets were not re-measured on this base and use
automatic sizing.

## Speed-up work on csf-qad (2026-10-06/07)

Kept: resident scales, L2 prefetch of packed MXFP8 weights, exact graph ladder,
W4A4. Measured and rejected: Marlin MXFP8 linears (-3% C1, -5% C8), forced b12x
`dynamic` experts / inline scale decode (-1 to -2.5%), `quantized` MXFP8
activations (neutral), b12x MTP draft experts (neutral), micro/dynamic cutover
256/512 pairs (neutral). Remaining gap to the original checkpoint is b12x's
MXFP8 GEMMs at C1 and its expert GEMM at C8 (profile:
`../../rebase-20261006/profile/`). Back-to-back runs without cool-down drift
-7 to -9% (GPU 2 idles near 71 C); all numbers above are from cooled runs.

## Not covered

DFlash2 and MTP0 on this image, csf-qad with LMCACHE=l1, uncensored budgets,
image inputs, DCP3 quality tests (DCP3 has arithmetic + stress only).

## Files

- `Dockerfile`, `src/` (all files changed vs upstream 20261006), `lil/` (cache.py
  TP3 gate + contract hash), `serve-glm53-flash-tp3-kraken.py` (launcher, also
  mounted by the compose file), marker JSONs, `compose.yaml` (below).
- `source.patch`: every source change versus upstream 20261006 (42 files, +2140/-402).
- `image-inspect.json`, `build.log`.
- Evidence: `../../rebase-20261006/RESULTS.md` (speeds, profile, stress),
  `../../qualification-20261007-csf/` (quality), `../../csf-qad-tp3-20261006/`
  (upstream cannot run CSF-QAD at TP3).

## Variables

| Variable | Default | Values |
| --- | --- | --- |
| `CHECKPOINT` | `csf-qad` | `csf-qad`, `default`, `uncensored` (experimental) |
| `CSF_ACTIVATIONS` | `a4` | `a4`, `a16` (csf-qad only) |
| `CSF_RESIDENT_SCALES` | `1` | `1`, `0` (csf-qad only) |
| `MODE` | `mtp` | `mtp` (depth 3), `dflash2`, `mtp0` (csf-qad: mtp only) |
| `DRAFT_TOKENS` | empty (= 3 MTP / 7 DFlash2) | `1`-`7`; only the defaults are qualified |
| `DCP` | `1` | `1`, `3` |
| `MOE_TP` | `1` | `1` TP experts (required for csf-qad), `0` EP3 |
| `MOE_TP_BACKEND` | empty | `b12x` (csf-qad), `flashinfer_cutlass`, `marlin` |
| `KV_RECLAIM` | `auto` | `auto`, `0` |
| `KV_CACHE_MEMORY_BYTES` | empty | explicit bytes/GPU for the budgeted configurations |
| `LMCACHE` / `LMCACHE_L1_GB` | `off` / `96` | `off`, `l1` / GiB of host RAM |
| `BF16_GEMV` | `1` | `1`, `0` |
| `FASTOKENS` | `1` | `1`, `0` |
| `NATIVE_DCP` | `1` | native PCIe DCP collectives at DCP3 (`0` = off) |

```yaml
# Kraken TP3 release example, image azallaza/glm53-kraken-tp3:20261007
# (our TP3 port rebased onto upstream karmic-kraken-beta-20261006-c8c8fe).
# Defaults: CSF-QAD checkpoint / MTP3 / DCP1 / MOE_TP=1, W4A4 experts with
# resident scales, KV 4,199,951 tokens (4.01x 1M), top-p 0.95.
# CHECKPOINT=default serves the original NVFP4 checkpoint on the same image.
# All source changes are built into the image; the launcher is also mounted.
# GPU power is a host setting: tested 350W/GPU; never exceed 400W.
services:
  glm53-kraken-tp3:
    image: "${GLM53_IMAGE:-azallaza/glm53-kraken-tp3:20261007}"
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
      # csf-qad = local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@fd660d51
      #   (same QAD experts, compressed expert scales, MXFP8 attention/shared
      #   experts; LIL License 1.0). Needs MODE=mtp and MOE_TP=1 (b12x experts).
      # default = local-inference-lab/GLM-5.3-Flash-NVFP4@175ae8ce (original).
      # uncensored = orcarouter NVFP4 checkpoint (experimental on this image).
      CHECKPOINT: "${CHECKPOINT:-csf-qad}"
      # csf-qad only. a4 = W4A4 experts (default, ~10% faster decode than a16,
      # quality-qualified); a16 = upstream's W4A16.
      CSF_ACTIVATIONS: "${CSF_ACTIVATIONS:-a4}"
      # csf-qad only. 1 = expand expert scales once per layer (+3.7% C1, +2% C8,
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
      # measured on this image: csf-qad 2215 blocks (4,199,951 KV, 4x1M stress),
      # default 1700 blocks (3,222,912 KV). DCP3, uncensored and LMCACHE=l1 use
      # automatic sizing here (csf-qad DCP3: 10,325,560 KV). 0 = off everywhere.
      # TP-expert MoE backend (MOE_TP=1): empty = b12x (csf-qad, required) /
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
      # Empty chooses32 for MTP/off or64 for DFlash (covers C8 verification).
      MAX_CUDAGRAPH_CAPTURE_SIZE: "${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
      # Empty: finer MTP/off captures for default and csf-qad at DCP1, graph32, 8 slots.
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
      # prefill and requires b12x collectives and b12x KDA decode.
      # Expert proof checks EP3 normally, or actual TP3 partitioning with MOE_TP=1.
      GLM53_TP3_REQUIRE_RUNTIME_PROOF: "${GLM53_TP3_REQUIRE_RUNTIME_PROOF:-1}"

      # MTP acceptance-length adaptation. Empty = disabled, the qualified
      # configuration. A positive integer averages accepted draft lengths over
      # that many verification steps and trims the speculative-token count,
      # with DRAFT_TOKENS as the upper bound. Measured 0.0% at C1, where the
      # controller never trims; under evaluation at C8.
      MTP_ADAPTIVE_WINDOW: "${MTP_ADAPTIVE_WINDOW:-}"
      # Speculative tokens per step. Empty = mode default (MODE=mtp 3, dflash2 7);
      # 1-7 accepted. Qualified: MTP 3 and DFlash2 7. Any other MTP value turns
      # off the KV reclaim budget (automatic KV sizing) and the measured graph
      # ladder; above 3 the graph maximum grows to 8 x (tokens + 1).
      DRAFT_TOKENS: "${DRAFT_TOKENS:-}"
      # FP8 weight-only decode (Marlin W8A16) for the large BF16 projections:
      # KDA in_proj/o_proj, DSA o_proj/q_b_proj, dense FFN, LM head, MTP draft.
      # With csf-qad most of these are MXFP8 in the checkpoint already (b12x);
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
