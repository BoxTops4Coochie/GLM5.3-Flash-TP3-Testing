# GLM-5.3-Flash — Kraken TP3

GLM-5.3-Flash NVFP4 on three NVIDIA RTX PRO 6000 Blackwell Workstation Edition
GPUs (96 GB each), with MTP3, FP8 KV and a 1,048,576-token maximum context.
This ports the R34 TP3 changes to the pinned `karmic-kraken-beta` build and
retains the environment-based checkpoint, speculation and DCP switches.
The release example below defaults to the measured **MTP3 / DCP1** configuration.

## Status

| Setting | Value |
| --- | --- |
| Checkpoint | `default` and `uncensored`; both qualified in all four mode/DCP combinations |
| TP / EP / DCP | 3 / 3 / 1; DCP3 qualified with the supplied correction |
| Release example serving mode | MTP depth 3 |
| Other qualified mode | DFlash2 depth 7, MXFP8 draft, DCP1 and DCP3 |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4` |
| Target revision | `175ae8ce3b5af842b0d0140dbeb43e9cfc557c49` |
| Context / request slots | 1,048,576 / 8; actual capacity depends on request lengths |
| Sampling defaults | Temperature 1, top_p .95; no repetition penalty override |
| Reasoning / historical thinking | `max` / `clear_thinking=true`; request-overridable |
| KV / graphs | FP8 GPU-only cache / FULL_AND_PIECEWISE |
| Prefix caching | Enabled; recurrent cache mode `align` |
| Batch / maximum graph size | 4096 / 32 for MTP; automatic graph64 for DFlash |
| Startup-reported KV capacity, MTP3/DCP1 | **3,410,670 tokens** with finer captures (3.25× at maximum context) |
| GPU power limits | 350 W per GPU; host setting |
| CPU affinity on the measured host | CPUs 8–47; configurable in the release compose |

Eight request slots do not imply eight full-context requests fit at DCP1.
The table uses the engine's startup allocation report. The benchmark's
block-derived estimate uses different hybrid-cache accounting; neither is
evidence of tested full-capacity operation.

## Docker artifact

Published image: `azallaza/glm53-kraken-tp3:20260921`

```bash
docker pull azallaza/glm53-kraken-tp3:20260921
```

The release image is `glm53-kraken-tp3:baked-20260921`, built from
`glm53-kraken-tp3:ported-20260919`
(`sha256:c7f5aebe998defd34cfc901f8e462eadac5ee90b4180ea965b9dc9e87d23bc3d`)
with the three source corrections baked in rather than mounted.
Parent: `ghcr.io/local-inference-lab/vllm@sha256:5927520c447fdcbc0990567f9237756ff66a54e360438915a5f70cd5cf9d530d`.
The measurements in this document were taken on the local build; the published
image is the same artifact under a registry name. The example Compose defaults
to `azallaza/glm53-kraken-tp3:20260921`; `GLM53_IMAGE` overrides it.

The release requires the companion [launcher](serve-glm53-flash-tp3-kraken.py),
which compose mounts. The three source corrections — [DCP
correction](dcp.py), [recurrent padding correction](mamba_utils.py) and
[widened runtime proof](utils.py) — are **built into this image**; they are
included here for review, not because compose needs them.

The stock profile's `kda_prefill_backend: b12x` is deliberately left alone.
`/opt/lil/image-contract.json` pins sha256 hashes of the launcher and profiles
and the entrypoint refuses to start when they disagree, so patching the profile
would mean rewriting the integrity receipt it exists to provide. The qualified
prefill backend is selected through `ADDITIONAL_CONFIG` in compose instead,
which is why that variable must be kept when adapting the example.
Hashes and the observed server configuration are recorded in
[baseline-receipt.json](baseline-receipt.json). The completed qualification and
restored baseline are recorded in [qualification-receipt.json](qualification-receipt.json),
including hashes of the companions at that qualification snapshot and result summaries.
The subsequent MTP0 graph fix is documented in [mtp0-graph-fix.md](mtp0-graph-fix.md);
[mtp0-fix-receipt.json](mtp0-fix-receipt.json) records that fix snapshot.
The current capture policy and companion hashes are recorded in
[decode-tuning-receipt.json](decode-tuning-receipt.json).

The port includes:

- TP3 attention, recurrent-head, projection, MTP, DFlash and vocabulary padding,
  with strict checkpoint loading and zeroed padded tails; EP3 retained.
- Three-rank PCIe collectives adapted to Kraken's prepared-kernel interfaces.
- Measured skinny-GEMM projection dispatch, half-L2 budgets, cache/boundary
  corrections and explicit `tool_choice=none` handling.
- A warm-start fix that saves all ranks' FlashInfer tuning entries. Saving only
  rank0 left other ranks tuning while rank0 waited, causing a restart deadlock.
- The supplied DCP correction rounds the Triton rank vector to a power of two
  and masks the extra lanes; the original world-size-three path failed to compile.

Kraken's current sampler is retained: the old R34 small-batch mask was slower.
Its own tuner already selected the R34 GDN block16 choice. The old R34 native
DCP adapter was not transplanted; DCP3 here uses the newer upstream path with
the supplied rank-padding correction.

## Runtime backends

| Operation | Backend |
| --- | --- |
| Target attention | B12X |
| Dense projections | B12X / cuBLAS, with skinny GEMM at the port's measured shapes |
| Recurrent prefill / decode | FlashKDA / B12X |
| Routed target experts | EP3-capable automatic selection (FlashInfer CUTLASS NVFP4) |
| TP collectives | B12X PCIe oneshot; baseline cutoff 84 KiB |
| Default-checkpoint MTP experts / attention | Marlin / B12X |
| MTP vocabulary head | Private NVFP4 draft copy; BF16 target verifier |
| DFlash draft | `local-inference-lab/GLM-5.3-Flash-DFlash2`, MXFP8, dense attention |

DFlash revision: `713226ab03bc38afdf955c7450436c2f7176f6f8`.
The native Kraken stack is retained, including B12X 1.3.0 and FlashInfer 0.6.18.

### Recurrent prefill backend

The Kraken profile ships `additional-config: {kda_prefill_backend: b12x}`.
vLLM's own `resolve_kda_prefill_backend` will not select that backend under
`auto` — "that backend must be requested by name until its serving
qualification lands" — so the profile opts into an implementation upstream does
not consider ready to serve.

On very long prompts it measurably degrades the recurrent state that KDA
prefill builds, and long-context output quality with it.

This release serves `kda_prefill_backend: flashkda`, which is what the R34 TP3
build used in all 429 of its recorded runtime receipts. Measured cost: across
32 prefill cells (8 configurations x 8K/32K/64K/128K) throughput moves by a
median of **+0.5%**, range **-1.2% to +2.7%**, with 25 cells faster and 7
slower. That spread is inside run-to-run noise, so the honest claim is that
flashkda prefill costs nothing rather than that it is faster. Decode is
untouched and reproduces within +/-3.2%, inside the known container-uptime
envelope.

Everything else remains B12X: attention, dense projections, routed experts, TP
collectives and recurrent **decode**.

The runtime proof in `src/vllm/v1/worker/utils.py` is widened to accept either
`b12x` or `flashkda` for prefill. The image's copy pins `b12x` and raises on
anything else, which makes the qualified configuration impossible to select;
the mounted copy keeps the guard active for collectives, EP3, KDA decode and
the encoder mode.

## Validation

| Configuration / component | Recorded validation |
| --- | --- |
| TP3 source port | 36 CPU geometry/loading/compatibility/tuning-cache checks |
| Default DFlash2/DCP1 | Cold/warm startup, concurrent exact arithmetic, short OCR, named tool call and retrieval |
| Default MTP3/DCP1 | Repeated startups and 8/8 exact arithmetic checks on each tested arm; C1/C8 speed cells at 0/32K/128K |
| DCP correction | 16 eager and graph-replay cases across world sizes 2/3/4/6, both LSE bases, strided layouts and empty shards |
| Default MTP3/DCP3 with supplied correction | Startup and 8/8 exact arithmetic responses, natural EOS |
| Environment launcher | All 12 checkpoint/mode/DCP combinations pass configuration rendering; representative actual wrapper invocations checked |

All eight default/uncensored × MTP3/DFlash2 × DCP1/DCP3 configurations are
qualified on this build. Every one passes the arithmetic smoke with natural
EOS, exact 3-key retrieval at both 128K and 900K, decode at 0 / 32K / 128K at
concurrency 1 and 8, and standalone cold prefill to 128K. Per-variation
verifications and statistics are in
[VARIATIONS.md](VARIATIONS.md).

Long-generation integrity was measured separately: 40 requests at an 826K-token
retained history, natural EOS, top_p .95 — four seeds per configuration, and
twelve on default/MTP3/DCP1 to bound the one failure. Thirty-eight produced
full-length answers of 21,350–84,952 characters with worst `repeat_8gram`
0.0151. Two results need naming precisely:

- **One genuine failure** (1 of 38 scored trials, 2.6%), default/MTP3/DCP1
  seed 202: 49,194 characters of coherent reasoning (`repeat_8gram` 0.0042)
  followed by an end of turn with no answer.
- **Two responses answered by acting.** The 826K prompt is an agentic coding
  transcript, and in these the model issued a tool call rather than writing to
  the answer channel, then stopped on `<|observation|>` awaiting the result —
  once writing a complete ~13K-token retrospective into a `write` argument, and
  once calling a tool immediately in 47 tokens. The harness supplies no
  `tools`, so those tokens parse into neither content nor tool calls and the
  responses look empty. This is a property of the test prompt, not of
  generation integrity, and they are counted separately.

Reproducibility was checked on the one failure: three repeats of that seed on
the same configuration produced two full answers, so it is not a stable failing
trajectory. The third repeat exposed a separate mode worth stating — the model
looped inside the *reasoning* channel (191,979 characters at `repeat_8gram`
0.736) and ran to the 65,536-token cap, finishing on `length` rather than
`stop`. Re-scoring all 40 qualified trials on reasoning quality found **none
truncated and none looping**; the highest reasoning repetition among them was
0.1469, in a request that still ended naturally with a full answer.

The practical consequence is not a port defect: at `reasoning_effort: max` on
very long prompts the model can occasionally spend its whole token budget
thinking. Allow a generous `max_tokens` for long-context work. This matches the
hotel-lights behaviour recorded below.

Coverage limits worth stating: text-integrity inspection is not factual
verification of every claim in a generated retrospective, and speed requests
ignore EOS and do not establish correctness. Full-capacity operation at the
KV limit is not qualified.

Uncensored MTP0 originally crashed with CUDA graphs. The supplied recurrent
padding correction replaces an invalid -1 state index with reserved null block 0.
It passes 48 GPU regression cases, 116 arithmetic requests, 128K/900K retrieval
and an 826K natural-output request with the original full-graph ladder.
MTP0/DCP1 C1 decode is 124.3/123.5/122.0 tok/s at 0/32K/128K;
see [the fix results](mtp0-graph-fix.md) for C8 results and test limits. Default MTP0 passed
its original arithmetic smoke; that alone does not prove it cannot hit the shared bug.
See [draft comparison](draft-comparison.json).

See [complete measurements](qualification-measurements.md) and the compact
[results JSON](qualification-results.json) for KV, decode, verifier steps,
acceptance, repeated prefill and per-configuration long-context outcomes.

### Answer quality versus reasoning effort

Kraken's default MTP3/DCP1 `lavd` qualification used 30 requests at concurrency
8, maximum reasoning, temperature 1, top-p .95 and a 100K-token output cap.
It scored **27 EXACT / 3 NEAR / 0 FAIL**, with all 30 responses stopping
naturally and no truncations, stalls, timeouts or request errors. Median
completion length was **18,426 tokens** (range 12,949–37,901). Each near answer
was `73, 46.5`, one record and half an hour above the exact `72, 46`.
Near scores are not exact answers. Evidence: the qualification directory's
`lavd.json` and configuration receipt.

On `hotel-lights`, the same C8/max/100K-cap settings yielded **27 exact answers
and 3 truncated runs with no final answer** (30 total). Median completion
length was47,350.5 tokens. The three truncated runs reached100,000 tokens;
their saved reasoning repeatedly rechecks the puzzle without delivering the
final answer. No request errors, stalls or timeouts occurred. This profile is
**not free of runaway reasoning**. These observations alone do not attribute
the behavior to the port, MTP or an attention kernel.

The focused draft comparison and eager MTP0 controls are complete. These measurements
cover `max` only; no Kraken reasoning-effort comparison is claimed.
Reasoning remains `max` with `clear_thinking=true`;
requests may override those defaults.

## DCP3 versus DCP1

The matched MTP3 sweep allocated **3,446,120 KV tokens at DCP1** and
**8,839,353 at DCP3** (about2.57×). DFlash2 allocated2,998,163 and6,937,346
respectively. These are startup allocations, not tested full-capacity limits.

Default MTP3 output tok/s, contexts0/32K/128K:

| Mode | C1 | C8 aggregate |
| --- | --- | --- |
| DCP1 |192.4 /201.1 /206.2 |686.6 /687.1 /670.9 |
| DCP3 |170.1 /186.0 /191.5 |616.1 /643.3 /644.5 |

DCP3 verifier rates were about7–8% lower atC1 and5–8% lower atC8 in this
single sequential sweep. DCP1 remains the speed baseline. Repeated client
prefill was roughly9.6–10.3K prompt tok/s across8K–128K in both configurations.
Full tables, acceptance and sample counts are in the measurement companion.
Use `DCP=3 docker compose up -d` after active requests finish to trade some
speed for additional KV capacity.


## Finer graph captures — September 20 update

The default checkpoint at DCP1 now uses exact intermediate captures when graph maximum is32 and request slots are8. MTP0 uses `1 2 3 4 5 6 7 8 16 32`; MTP3 uses `1 2 4 8 12 16 20 24 28 32`. Explicit ladders and other configurations keep their previous behavior. No image rebuild is required.

At C5, the matched-order repeat improved MTP0 throughput by **24.8–26.1%** and MTP3 verifier request-step rate by **26.8–27.9%**, versus the stronger of two controls at each context. The earlier screen also beat both controls. No consistent C1/C8 gain is claimed. MTP3 C1/C8 step rates remained within1.5% of the controls; the largest shortfall against the stronger control was1.4% at C8/128K. MTP0 C1/C8 remained near baseline.

Matched-order C5 results, contexts0/32K/128K:

| Measurement | 0 | 32K | 128K |
|---|---:|---:|---:|
| MTP0 aggregate tok/s | 376.3 | 376.1 | 367.5 |
| MTP3 aggregate tok/s | 496.6 | 533.3 | 504.5 |
| MTP3 aggregate verifier request-steps/s | 196.0 | 200.0 | 195.2 |

KV allocation changes from **4,299,726 to 4,284,666** tokens for MTP0 (~0.35% less in the confirmation; ~0.5% in the first screen), and **3,446,120 to 3,410,670** for MTP3 (~1.03% less in the confirmation; ~1.2% in the first screen). To recover the original ladder/capacity, explicitly set `CUDAGRAPH_CAPTURE_SIZES="1 2 4 8 16 32"`.

Tests used DCP1,350W/GPU,top-p.95,batch4096,84KiB cutoff,15-second pre-decode warmup and30-second measured cells. Both graph candidates had a second run in the same C1→C5→C8 order as controls; arithmetic checks at C8 and C5 passed. This update received basic correctness and decode checks only; the previously recorded longer qualification predates these captures. The C5 percentages are not C1/C8 or DCP3 claims.

The256KiB cutoff remains optional: the new original-ladder MTP3 C8 screen beat both controls by2.3–4.3% in verifier rate, consistent with earlier repeats. It did not consistently help MTP0/C1. Next-layer prefetch had no useful gain and stays0. The isolated, state-correct convolution/KDA fusion prototype was neutral for MTP0 and slower for MTP3; it was not integrated into serving.

Raw new measurements: [decode-tuning-results.json](decode-tuning-results.json).

## Earlier sustained decode benchmark at 350 W

The supplied [benchmark_results.json](benchmark_results.json), recorded
September 19, 2026 at 16:06 local time, uses llm-inference-bench 0.6.2: 30 seconds
measured per cell, 3-second decode warmup setting, C1/C8,0/32K/128K and
`max_tokens=8192`. All six cells report no errors, loops, underfill, capacity
limit or warmup timeout.

The JSON stores `dcp_size=0` (the benchmark option), not an explicit DCP receipt.
The running server created before this run and its startup logs identify
**default checkpoint / MTP3 / DCP1 / graph32 / batch4096 /84KiB cutoff**;
see the receipt and [startup values](startup-values.log). The benchmark inherits
temperature/top_p defaults rather than explicitly recording both request values.

Aggregate output tokens/sec; C8 is total across eight concurrent requests:

| Context | DCP1 C1 tok/s | DCP1 C8 aggregate tok/s |
| --- | ---: | ---: |
| 0 | 208.1 | 686.6 |
| 32,768 | 214.0 | 703.6 |
| 131,072 | 225.9 | 685.0 |

Verifier request-steps/sec with mean acceptance length in parentheses. At C8,
the counter sums across requests; it is not physical GPU batch forwards/sec.
Acceptance helps explain raw-throughput variation; normalizing by it does not
remove all routing, scheduling or thermal variation.

| Context | DCP1 C1 steps/s (acceptance) | DCP1 C8 steps/s (acceptance) |
| --- | ---: | ---: |
| 0 | 83.47 (2.494) | 269.66 (2.546) |
| 32,768 | 82.97 (2.579) | 272.60 (2.581) |
| 131,072 | 81.50 (2.771) | 263.02 (2.604) |

Prefill is the benchmark's client prompt-tokens/TTFT measurement, one scout
sample per point. It is not a repeated standalone prefill or pure kernel-rate
measurement. Prompt counts differ from nominal context labels.

| Context label | Prompt tokens | TTFT(s) | Client prefill tok/s |
| --- | ---: | ---: | ---: |
| 8K | 8,196 | 0.735 | 11,150 |
| 32K | 32,312 | 2.850 | 11,338 |
| 64K | 64,490 | 5.615 | 11,485 |
| 128K | 128,850 | 12.333 | 10,448 |

Whole-run summed GPU power averaged **965.7W** and peaked at
**1,085.0W** in telemetry; configured limits were 350 W/GPU
(1,050W combined). These are GPU readings, not outlet/whole-system power.
The maximum sampled GPU temperature was 86°C.

A separate matched cutoff experiment with the original target capture ladder used 30-second decode warmup and 45-second
measurement windows. Raising `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE` to `256KB`
improved MTP3/DCP1 C8 verifier rate by roughly **2–5%** across two candidate
passes against two controls, with no meaningful KV reduction. C1 did not
improve. This is optional and **was not enabled in the table above**; no
DFlash or uncensored cutoff gain is claimed.

Single-request real-coding observations are not included: no Kraken log sample
with sufficient workload context has been supplied.

### DFlash2 DCP1 reference measurements

Earlier measurements on the same local image used the MXFP8 DFlash2 draft,
depth7, graph64, batch4096 and350W/GPU. These used30-second decode warmup and
45-second measurement windows, so they are kept separate from the newer MTP
run. Startup-reported KV allocation was **2,998,163 tokens**. Sources:
[C1 JSON](dflash2-c1.json) and [repeated C8 JSON](dflash2-c8.json).

| Context | C1 output tok/s | C8 aggregate output tok/s |
| --- | ---: | ---: |
| 0 | 165.7 | 535.6 |
| 32,768 | 169.6 | 562.8 |
| 131,072 | 146.8 | 545.3 |

DFlash2/DCP3 is now measured in the qualification companion. Default-checkpoint
C1 output154.7/161.5/159.1 tok/s andC8 aggregate511.6/525.7/525.0 tok/s;
its retrieval and retained-history checks passed within the stated coverage.
Every uncensored configuration is qualified; see [VARIATIONS.md](VARIATIONS.md).

## Example Compose

Download this post's companion `compose.yaml`, `serve-glm53-flash-tp3-kraken.py`,
`dcp.py`, `mamba_utils.py`, and `utils.py` into one directory. Set `GLM53_IMAGE` to the image tag and
`HF_CACHE_DIR` to a populated Hugging Face cache; offline loading is enabled.
Adjust CPU affinity for your host. Defaults below are MTP3/DCP1 with the September20 finer capture policy.
The earlier benchmark retains its original capture ladder and receipt.

```bash
export GLM53_IMAGE=YOUR_PUBLISHED_IMAGE
export HF_CACHE_DIR=/path/to/huggingface
CHECKPOINT=default MODE=mtp DCP=1 docker compose up -d
# Alternative modes; see validation scope above before deployment.
CHECKPOINT=default MODE=dflash2 DCP=1 docker compose up -d
CHECKPOINT=uncensored MODE=mtp DCP=3 docker compose up -d
```

Variables can instead be saved in `.env`. Stop active inference before switching.
Graph size defaults to 32 for MTP/off and 64 for DFlash; batch size remains 4096.

```yaml
# Kraken TP3 release example: measured default/MTP3/DCP1 baseline.
# GPU power is a host setting: tested350W/GPU; never exceed400W.
services:
  glm53-kraken-tp3:
    image: "${GLM53_IMAGE:?Set GLM53_IMAGE to the published image or your local build}"
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
      # Tested rank-three padding correction, required for DCP3 on this image.
      - ./dcp.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/dcp.py:ro
      # Recurrent padding fix for graph-enabled MTP0.
      - ./mamba_utils.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/mamba_utils.py:ro
      # Runtime proof accepts flashkda as well as b12x KDA prefill; the image's
      # copy pins b12x, which forbids the qualified long-context configuration.
      - ./utils.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/utils.py:ro
    environment:
      # Set here, in .env, or before `docker compose up -d`.
      # default = released NVFP4; uncensored = orcarouter NVFP4 checkpoint.
      # Both checkpoints are qualified in all four mode/DCP combinations.
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
      # Default/MTP3/DCP3 passed basic checks with the mounted fix.
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

      # Optional256KB: earlier C8 gains used the original capture ladder.
      # No consistent gain confirmed with the finer default captures.
      # Empty retains upstream84KiB. No DFlash/uncensored gain claimed.
      VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: "${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-}"
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
      VLLM_NO_USAGE_STATS: "1"
      GLM53_TP3_REQUIRE_RUNTIME_PROOF: "1"
      # Half-L2 budgets retained from R34.
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

### Further decode experiments — September 20

The baseline above is retained. Corrected next-layer prefetch did not improve single-request decode. Exact MTP draft graphs and the 256KiB cutoff with finer target captures showed no consistent improvement over paired controls. All 57 speed cells were valid and 117 arithmetic checks passed; no new code or default was adopted. Earlier cutoff gains used the original target ladder. Details: [decode follow-ups](decode-followups.md). These checks do not extend long-output qualification.
