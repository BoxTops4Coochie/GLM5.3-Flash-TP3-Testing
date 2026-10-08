# GLM-5.3 Flash Kraken TP3 — 20261008

This model is based on [GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) by Local Inference Lab, Inc., a non-profit organization, available at <https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD>. GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0.

Image: **`azallaza/glm53-kraken-tp3:20261008`** (local `glm53-kraken-tp3:20261008`, `sha256:44c2d27ec0269e4620895a3ce21f639ee93f9e4d9b935e896ed52a3a6900c486`).
Our GLM-5.3-Flash TP3 port rebased onto upstream `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta`
(`sha256:629d097c…`, vLLM `89f1ceecd`), serving the newest CSF-QAD revision (`dec48abd`, quantized
vision tower) as well as the previous one (`fd660d51`). Upstream cannot serve either at TP3.

```bash
docker pull azallaza/glm53-kraken-tp3:20261008
```

Previous images: `azallaza/glm53-kraken-tp3:20261007` (same feature set on upstream 20261006, CSF-QAD `fd660d51` only);
20260927 (original NVFP4 checkpoint, ~1.5% faster C1 for that checkpoint).
Host settings: 350 W/GPU (never exceed 400 W), +6000 memory offset.

## What changed since 20261007

| Change | Effect |
| --- | --- |
| Rebased onto upstream vLLM `89f1ceecd` (3-way merge; conflicts only in the CSF expert path) | newer vLLM/b12x/LMCache |
| Upstream's CSF loader now pads TP experts generically (added for GLM-5.3 744B TP6); it serves our MOE_TP 2048 -> 2112 (704/rank) unchanged | our own CSF padding patch dropped |
| **`CHECKPOINT=csf-qad-2`** = CSF-QAD `dec48abd`: identical QAD decoder, vision tower stored quantized (MXFP8 attention, NVFP4 MLP/merger) in a standard ModelOpt layout | new default |
| Vision tower in data mode for csf-qad-2 (replicated per GPU; its quantized widths do not pad at TP3); runtime proof accepts data mode | same memory as the sharded BF16 tower |
| DFlash2 at TP3 fixed (GLM-5.3 TP3 drafts use our loaded-size padding instead of upstream's BF16-only path); csf-qad* also allow MODE=dflash2 | DFlash2 works again (MTP3 remains faster) |

## Modes and measurements (350 W)

| Setting | KV tokens | Decode steps/s @32K C1 / C8 | Prefill 32K | Status |
| --- | ---: | --- | --- | --- |
| **csf-qad-2 MTP3 DCP1** (default) | **4,199,951 (4.01x 1M)** | 107-108 / — (user run) | ~10.5K tok/s | quality + stress below |
| **csf-qad-2 MTP3 DCP3** | **10,485,760 (10.00x 1M)** | 98.4 / 271.1 | 10,185 | image + 8x1M stress, 8/8 |
| csf-qad MTP3 DCP1 (`fd660d51`) | 4,199,951 (4.01x) | 105.4 / 306.0 | — | regression-checked on this image |
| csf-qad on 20261007, for reference | 4,199,951 | 105-108 / 307-315 | 10,825 | fully qualified |
| csf-qad-2 **DFlash2** (`MODE=dflash2`) | 3,344,754 (auto) | 81.8 (accept 2.60) = 213 tok/s | — | 8/8 |
| original NVFP4 **DFlash2** (`CHECKPOINT=default MODE=dflash2 MOE_TP=0`) | 3,278,412 (auto) | 84.1 (accept 2.45) = 206 tok/s | — | 8/8 (20260927: 205.6 tok/s) |

MTP3 (~270 tok/s C1 for csf-qad-2) remains faster than DFlash2 on every checkpoint.

```bash
docker compose up -d                                   # csf-qad-2 (dec48abd), MTP3/DCP1, 4.01x 1M
CHECKPOINT=csf-qad docker compose up -d                # fd660d51 (BF16 vision)
CSF_ACTIVATIONS=a16 docker compose up -d               # upstream's W4A16 experts (slower)
DCP=3 docker compose up -d                             # csf-qad-2 DCP3, 10.0x 1M (2145 blocks)
CHECKPOINT=default docker compose up -d                # original NVFP4 (1700-block budget)
```

## Quality and stress on this image (csf-qad-2)

| Test | csf-qad (`fd660d51`) on 20261007 | **csf-qad-2 on 20261008** |
| --- | --- | --- |
| lavd, 30 runs C8 | 30 exact | **30 exact / 0 wrong / 0 truncated** |
| hotel-lights, 30 runs C8 (seed 103) | 28 exact / 2 wrong (58/60 over two batches) | **28 exact / 1 wrong / 1 truncated** |
| Vision basic (OCR, text, counting, colors, chart), T=0/T=1 | 120/120 (BF16 vision) | **120/120** |
| Vision hard (small OCR, dense count, chart values, shades) | 117/120 | **117/120** |
| Arithmetic | 8/8 | 8/8 |

| Stress (2215 blocks) | Result | Min free |
| --- | --- | ---: |
| 8 concurrent x 10 photos 6000x4000 (79.6K prompt tokens each) | all OK | 2.0 GB |
| 4 x ~891K prompts + 120K decode each (right after the image stress) | all OK, 0 preemptions, peak KV 90% | 1.44 GB (GPU 0) |
| DCP3 (2145 blocks): 8 concurrent x 10 photos 6000x4000 | all OK | 2.44 GB |
| DCP3 (2145 blocks): 8 x ~891K prompts + 120K decode each | 8/8 OK, 0 preemptions, peak KV 56% | 2.92 GB |

DCP3 decode steps/s (cooled, 0/32K/128K): C1 99.8 / 98.4 / 96.4, C8 280.0 / 271.1 / 266.5;
prefill 32K / 128K 10,185 / 9,786 tok/s (csf-qad DCP3 on 20261007: C1 101.6 / 99.7 / 97.6,
C8 275.0 / 274.8 / 263.5).

Vision test kit and results: `../../vision-test/`. Evidence: `../../rebase-89f1cee/RESULTS.md`,
`../../qualification-20261008-csf2/`.

## CSF-QAD checkpoint

Both revisions hold the same QAD-distilled routed experts as the original NVFP4 checkpoint, with
their FP8 block scales losslessly compressed (CSF) and MXFP8 attention/shared experts. `dec48abd`
also stores the vision tower quantized and uses the standard ModelOpt layout. Compressed experts
run only on the b12x MoE backend, so `csf-qad*` require `MOE_TP=1`; W4A4 experts
(`CSF_ACTIVATIONS=a4`) and resident expanded scales (`CSF_RESIDENT_SCALES=1`) are the defaults.
License: LIL License 1.0 — no reuploads; any README of a project that runs it must begin with the
attribution notice above.

## Not re-run on this image

Long-context retrieval/generation, DCP3 quality tests, DFlash2 quality, uncensored checkpoint, LMCache L1
(qualified on 20261007 / 20260927 with the identical text weights).

## Files

- `Dockerfile`, `src/` (files changed vs upstream `89f1ceecd`), `lil/` (cache.py TP3 gate + contract
  hash), `serve-glm53-flash-tp3-kraken.py`, marker JSONs, `compose.yaml` (below).
- `source.patch`: every source change versus upstream `89f1ceecd` (41 files, +2064/-386).
- `image-inspect.json`, `build.log`.

## Variables

| Variable | Default | Values |
| --- | --- | --- |
| `CHECKPOINT` | `csf-qad-2` | `csf-qad-2`, `csf-qad`, `default`, `uncensored` (experimental) |
| `CSF_ACTIVATIONS` | `a4` | `a4`, `a16` (csf-qad* only) |
| `CSF_RESIDENT_SCALES` | `1` | `1`, `0` (csf-qad* only) |
| `MODE` | `mtp` | `mtp`, `dflash2`, `mtp0` |
| `DRAFT_TOKENS` | empty (= 3 MTP / 7 DFlash2) | `1`-`7`; only the defaults are qualified |
| `DCP` | `1` | `1`, `3` |
| `MOE_TP` | `1` | `1` (required for csf-qad*), `0` EP3 |
| `MOE_TP_BACKEND` | empty | `b12x` (csf-qad*), `flashinfer_cutlass`, `marlin` |
| `KV_RECLAIM` | `auto` | `auto`, `0` |
| `KV_CACHE_MEMORY_BYTES` | empty | explicit bytes/GPU for the budgeted configurations |
| `LMCACHE` / `LMCACHE_L1_GB` | `off` / `96` | `off`, `l1` / GiB of host RAM |
| `BF16_GEMV` | `1` | `1`, `0` |
| `FASTOKENS` | `1` | `1`, `0` |
| `NATIVE_DCP` | `1` | native PCIe DCP collectives at DCP3 (`0` = off) |

```yaml
# Kraken TP3 release example, image azallaza/glm53-kraken-tp3:20261008
# (our TP3 port rebased onto upstream karmic-kraken-beta, vLLM 89f1ceecd).
# Defaults: CSF-QAD (dec48abd, quantized vision) / MTP3 / DCP1 / MOE_TP=1, W4A4
# experts with resident scales, KV 4,199,951 tokens (4.01x 1M), top-p 0.95.
# CHECKPOINT=csf-qad serves fd660d51 (BF16 vision); default = original NVFP4.
# All source changes are built into the image; the launcher is also mounted.
# GPU power is a host setting: tested 350W/GPU; never exceed 400W.
services:
  glm53-kraken-tp3:
    image: "${GLM53_IMAGE:-azallaza/glm53-kraken-tp3:20261008}"
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
      # Empty chooses32 for MTP/off or64 for DFlash (covers C8 verification).
      MAX_CUDAGRAPH_CAPTURE_SIZE: "${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
      # Empty: finer MTP/off captures for default and csf-qad* at DCP1, graph32, 8 slots.
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
