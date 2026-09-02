# GLM-5.3 Flash Native TP3 / EP3 — v23

v23 adds working native MTP speculative decoding to the deterministic v22 TP3/EP3 target path.

The target model remains `local-inference-lab/GLM-5.3-Flash-NVFP4` on TP3 + EP3 with B12X sparse MLA, Triton KDA decode, FP8 KV cache, vision enabled through replicated encoder mode, and FlashInfer autotuning disabled.

The final v23 image is:

```text
voipmonitor/vllm:glm53-native-tp3-bf16-v23d-mtp-tp-loader
```

## Status

v23d is the current qualified base for MTP work.

```text
checkpoint:              local-inference-lab/GLM-5.3-Flash-NVFP4
snapshot:                378ca54585c46542bad1f3cb3ed0d73ae51cdb62
TP / EP:                 3 / 3
context:                 1,048,576
vision:                  enabled
vision TP mode:          data / replicated encoder
KV dtype:                FP8
target attention:        B12X
target MoE:              FLASHINFER_CUTLASS
KDA decode:              Triton
KDA hot projection:      FP8 E4M3 W8A16 / Humming
FlashInfer autotune:     disabled
graphs:                  FULL -> FULL_DECODE_ONLY
TP/EP communication:     PyNCCL / NCCL 2.31.2
```

The target path remains deterministic. With MTP disabled, the v23d control reproduces the expected arithmetic result `38000` in 20/20 runs.

With MTP1 enabled, v23d is the best qualified speculative configuration from the MTP1–MTP5 sweep: it preserves correctness while improving warm 512-token decode from roughly 110 tok/s to roughly 123 tok/s.

For the complete MTP1–MTP5 benchmark matrix, correctness results, vision results, acceptance rates, and per-depth analysis, see:

**[`GLM53-TP3-MTP1-MTP5-qualification.md`](GLM53-TP3-MTP1-MTP5-qualification.md)**

---

# What changed from v22

v22 established the deterministic target-model baseline. v23 does not replace that target architecture; it makes the checkpoint's native MTP layer usable under TP3.

```text
v23a: initial native MTP bring-up
v23b: TP3-safe shared output head / vocabulary handling
v23c: TP3-safe MTP VocabParallelEmbedding handling
v23d: TP-aware MTP checkpoint loader, including shared-expert padding
```

The final correctness fix is v23d.

## MTP checkpoint loading

The main target model already had TP3 padding and loading logic for dimensions that are not naturally divisible by three. The MTP layer initially bypassed part of that path.

v23d reuses the target model's TP padding helper while loading MTP weights. In particular, the MTP shared expert must respect the same logical-to-physical TP3 geometry as the target model instead of attempting to load the raw logical checkpoint shape directly.

This removed the remaining MTP TP3 load mismatch and allowed the draft layer to initialize from the same LIL checkpoint as the target.

---

# Draft attention backend selection

After the MTP weights were made TP3-safe, startup still failed while selecting an attention backend for the draft model.

vLLM creates a separate draft configuration for MTP. The target's automatically selected B12X backend was not automatically inherited by that draft configuration.

The working runtime configuration explicitly selects B12X for MTP:

```json
{
  "method": "mtp",
  "num_speculative_tokens": 1,
  "attention_backend": "B12X",
  "moe_backend": "marlin"
}
```

The target remains:

```text
attention: B12X
MoE:       FLASHINFER_CUTLASS
```

while the MTP layer uses:

```text
attention: B12X
MoE:       MARLIN MXFP8
```

This is intentional.

---

# Recommended v23 configuration

MTP1 is the recommended qualified speculative depth.

```yaml
services:
  glm53-native-tp3:
    image: voipmonitor/vllm:glm53-native-tp3-bf16-v23d-mtp-tp-loader
    container_name: glm53-native-tp3
    restart: "no"
    ipc: host
    shm_size: "64gb"

    ports:
      - "15015:8000"

    volumes:
      - /m2-2/huggingface:/root/.cache/huggingface

    environment:
      HF_HUB_OFFLINE: "1"
      GLM53_DEBUG_KDA_STATE: "0"
      GLM53_DEBUG_MLA_HEADS: "0"
      GLM53_DEBUG_INDEXER: "0"
      GLM53_TORCH_PROFILE: "0"

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0", "1", "2"]
              capabilities: [gpu]

    command:
      - local-inference-lab/GLM-5.3-Flash-NVFP4
      - --tensor-parallel-size
      - "3"
      - --enable-expert-parallel
      - --mm-encoder-tp-mode
      - data
      - --max-model-len
      - "1048576"
      - --max-num-seqs
      - "8"
      - --max-num-batched-tokens
      - "4096"
      - --gpu-memory-utilization
      - "0.97"
      - --kv-cache-dtype
      - fp8
      - --disable-custom-all-reduce
      - --served-model-name
      - GLM-5.3-Flash-TP3
      - --moe-backend
      - auto
      - --linear-backend
      - auto
      - --speculative-config
      - '{"method":"mtp","num_speculative_tokens":1,"attention_backend":"B12X","moe_backend":"marlin"}'
      - --compilation-config
      - '{"cudagraph_mode":"FULL"}'
      - --additional-config
      - '{"glm53_kda_decode_backend":"triton"}'
```

For a non-speculative v23d control, remove only `--speculative-config`.

---

# Qualification summary

| Depth | Correctness | General decode | Acceptance | Qualification |
|---|---|---|---|---|
| MTP1 | PASS | Best overall | Highest tested | **Recommended** |
| MTP2 | FAIL | Fast short test | Lower | Disqualified |
| MTP3 | PASS | Slower than MTP1 | Lower | Qualified |
| MTP4 | PASS | Slower / budget caveat | Lower | Qualified |
| MTP5 | PASS | Substantial slowdown | Lowest | Qualified |

MTP1 is therefore the v23 default. Deeper speculative depths can increase throughput on very short deterministic workloads, but the lower acceptance rate and extra draft work reduce general long-form decode performance.

Detailed numbers intentionally live in the separate qualification document rather than being duplicated here:

**[`GLM53-TP3-MTP1-MTP5-qualification.md`](GLM53-TP3-MTP1-MTP5-qualification.md)**

---

# v23 result

v23 completes the native MTP bring-up for the TP3/EP3 stack.

The key outcome is not simply that MTP starts: the final v23d loader preserves the target model's TP3 geometry, the draft backend is selected explicitly, deterministic target behavior remains intact, vision remains available, and MTP1 produces a meaningful decode-speed improvement without failing the frozen correctness test.

Further optimization work after v23 should treat v23d + MTP1 as the qualified speculative baseline.

## Future MTP optimization

MTP is working and qualified at depth 1, but there is still room to improve it.

The current MTP1 path is limited by a combination of draft-model cost, acceptance rate, draft MoE/backend efficiency, and the fact that deeper speculative depths do not currently translate into better general decode throughput. MTP2–MTP5 showed that simply increasing speculative depth is not enough; improving the quality/cost balance of the draft path is the more promising direction.

After the current v24 MLA/TP3 work is complete, planned follow-up work includes:

- improving MTP draft efficiency without changing target-model correctness,
- investigating whether draft MoE/backend selection can be made faster than the current MARLIN MXFP8 path,
- testing newer MTP/runtime changes from the qualified Jovian/vLLM work,
- revisiting acceptance-rate behavior and per-position usefulness,
- checking whether target-side improvements from v24 compose cleanly with MTP1,
- and re-qualifying speculative depths only when there is a concrete reason to expect a throughput gain.

The immediate priority is v24. MTP optimization will resume afterward using v23d + MTP1 as the reference point.
