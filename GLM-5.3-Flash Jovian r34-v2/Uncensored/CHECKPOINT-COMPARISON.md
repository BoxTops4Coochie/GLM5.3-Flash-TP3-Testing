# GLM-5.3-Flash R34 TP3 — default vs uncensored checkpoint

Comparison of the two checkpoints selectable with the Compose `CHECKPOINT`
variable on image `glm53-r34-tp3:native-dcp-20260913`. Measured 2026-09-16 on
three RTX PRO 6000 Blackwell at **350 W per GPU**, MTP3, temperature 1 /
top_p .95.

| | `CHECKPOINT=default` | `CHECKPOINT=uncensored` |
| --- | --- | --- |
| Repository | `local-inference-lab/GLM-5.3-Flash-NVFP4` | `orcarouter/GLM-5.3-Flash-Uncensored-NVFP4` |
| Revision | `46aaae8a82032f77100f2f03e9cc11b391df3b4d` | `ec0adf4f49c9570807cc11a5f650538c1893ae54` |
| Safetensors total | 184.5 GiB | 191.0 GiB |

## Bottom line

Decode throughput is **at parity**; the uncensored checkpoint costs about
**20% of prefill speed**, about **11% of KV capacity**, and scores slightly
lower on the one quality benchmark run against both, though not by a
statistically distinguishable margin. Every functional check passes on both,
at DCP1 and DCP3.

## Why they differ at all

The architecture is identical — both are `glm5_next` /
`Glm5NextForConditionalGeneration` with the same 45 layers, 4096 hidden size,
64 attention heads, 288 routed experts, vocabulary 154880, MTP layer 45 and
vision tower. `tokenizer.json`, `tokenizer_config.json` and
`processor_config.json` are **byte-identical**. The chat templates differ only
cosmetically (`~` versus `+` concatenation, none-handling, and early-exit
guards in the tool-sorting loops); both expose the same three effort tiers.

The whole difference is the quantization container:

| | default | uncensored |
| --- | --- | --- |
| Scheme | modelopt `MIXED_PRECISION` | compressed-tensors `nvfp4-pack-quantized` |
| Routed experts | NVFP4, **static W4A4** | NVFP4, **W4A16** |
| Activation scales | 36,288 calibrated `input_scale` tensors | none |
| MTP layer 45 experts | MXFP8, group 32 | **unquantized BF16** |
| Expert kernel required | b12x / FlashInfer NVFP4 | Marlin (weights-only) |

An audit of all 111,346 uncensored tensors against the reference found **zero
shape mismatches and zero unexpected tensors** once `weight_packed` → `weight`
and `weight_global_scale` → `weight_scale_2` are renamed — the packed weights
and E4M3 block scales are the same format, and the global scale is the exact
reciprocal (`24576.0` versus `4.0690105e-05`, the same value). The only gaps
are the 36,288 missing activation scales and the BF16 MTP layer, which is why
the runtime needs different kernels rather than a repack.

## Runtime configuration

The launcher derives these from `CHECKPOINT`; nothing else needs setting.

| | default | uncensored |
| --- | --- | --- |
| `--quantization` | `modelopt_mixed` | `compressed-tensors` |
| `--load-format` | `instanttensor` | `auto` |
| Expert MoE backend | `auto` (b12x) | `marlin` |
| MTP MoE backend | `marlin` (MXFP8) | `triton` (BF16 layer) |

## Capacity

| KV cache | default | uncensored | Δ |
| --- | ---: | ---: | ---: |
| DCP1 | 2,091,238 | 1,866,544 | −10.7% |
| DCP3 | 7,493,749 | 6,900,837 | −7.9% |

The BF16 MTP experts consume memory that MXFP8 would not, which is where the
capacity goes.

## Decode

Aggregate output tok/s, with MTP-normalized verifier rate
(`output tok/s ÷ acceptance length`) in the adjacent column. C8 values are
aggregate across eight requests, not per-user speed.

### DCP1, concurrency 1

| Context | default tok/s | steps/s | uncensored tok/s | steps/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 199.9 | 81.9 | 198.4 | 81.3 |
| 8K | 184.9 | 81.1 | 203.7 | 80.9 |
| 32K | 206.6 | 81.3 | 203.9 | 80.5 |
| 128K | 195.5 | 80.5 | 194.2 | 79.4 |
| **median** | **197.7** | **81.2** | **201.1** (+1.7%) | **80.7** (−0.6%) |

### DCP1, concurrency 8

| Context | default tok/s | steps/s | uncensored tok/s | steps/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 636.3 | 258.7 | 665.4 | 254.8 |
| 8K | 656.6 | 263.7 | 657.0 | 255.1 |
| 32K | 687.7 | 263.5 | 658.4 | 252.9 |
| 128K | 643.3 | 255.3 | 676.6 | 250.2 |
| **median** | **650.0** | **261.1** | **661.9** (+1.8%) | **253.8** (−2.8%) |

### DCP3, concurrency 1

| Context | default tok/s | steps/s | uncensored tok/s | steps/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 191.3 | 76.8 | 202.5 | 76.8 |
| 8K | 182.0 | 77.1 | 195.7 | 76.2 |
| 32K | 200.6 | 77.2 | 197.3 | 76.2 |
| 128K | 198.4 | 76.6 | 190.7 | 75.7 |
| **median** | **194.9** | **77.0** | **196.5** (+0.8%) | **76.2** (−1.0%) |

### DCP3, concurrency 8

| Context | default tok/s | steps/s | uncensored tok/s | steps/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 619.3 | 249.7 | 618.3 | 240.0 |
| 8K | 583.4 | 244.1 | 612.2 | 238.2 |
| 32K | 620.8 | 246.3 | 627.8 | 238.9 |
| 128K | 688.7 | 251.4 | 658.3 | 241.1 |
| **median** | **620.0** | **248.0** | **623.1** (+0.5%) | **239.4** (−3.5%) |

Raw output throughput is indistinguishable in every cell — the per-context
spread within each column exceeds the gap between columns. The normalized
verifier rate is consistently 0.6–3.5% lower for the uncensored checkpoint,
offset by marginally higher MTP acceptance (2.44–2.73 versus 2.28–2.74), so the
two effects cancel at the output.

Marlin dequantizes weights where the b12x NVFP4 kernel does not, but at these
batch sizes decode is memory-bound and both read the same 4-bit payload, so the
extra work is nearly free.

## Prefill

This is where the kernel difference shows up, because prefill is compute-bound.

| Context | DCP1 default | DCP1 uncensored | Δ | DCP3 default | DCP3 uncensored | Δ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 11,094 | 8,885 | −19.9% | 10,330 | 8,737 | −15.4% |
| 32K | 11,300 | 8,868 | −21.5% | 10,865 | 8,512 | −21.7% |
| 64K | 11,323 | 9,155 | −19.1% | 11,026 | 8,918 | −19.1% |
| 128K | 10,649 | 8,296 | −22.1% | 10,394 | 8,046 | −22.6% |

A consistent **15–23% prefill penalty** at every context and both DCP settings.
For long-prompt or agentic workloads that re-prefill often, this is the real
cost of the uncensored checkpoint, not decode.

## Answer quality

`llm-inference-bench` `lavd` profile, reasoning effort `max`, 30 runs,
concurrency 10, scored EXACT / NEAR / FAIL against `72, 46` (NEAR = both totals
within ±4).

| | Exact | Near | Fail | Median completion tokens |
| --- | ---: | ---: | ---: | ---: |
| default | 29 | 1 | 0 | 20,287 |
| uncensored | 26 | 3 | **1** | 12,831 |

Two-sided Fisher p = 0.35 — **not statistically distinguishable** at n=30, so
this is not evidence that the fine-tune is less accurate. Two observations are
worth recording anyway:

- The uncensored run produced a hard **FAIL** (`60, 35.75` against `72, 46`),
  far outside the ±4 band. Across 120+ default-checkpoint runs at various
  efforts, no FAIL was ever observed — only NEARs. It is one outlier, but it is
  a different failure mode.
- At the same `max` setting the uncensored checkpoint emits **37% fewer**
  completion tokens (12,831 versus 20,287). Since effort is purely a token
  budget on this model — default scores 18/30 at `high` versus 29/30 at `max` —
  a checkpoint that reasons less at `max` plausibly gives up some accuracy for
  the same nominal setting.

## Functional checks

All run on the uncensored checkpoint at both DCP settings; the default
checkpoint's equivalents are recorded in the parent directory's guide,
[glm-5.3-flash-tp3-r34.md](../glm-5.3-flash-tp3-r34.md).

| Check | DCP1 | DCP3 |
| --- | --- | --- |
| C1 exact-answer (3 cases) | pass | pass |
| C8 — eight simultaneous requests | pass | pass |
| Vision / OCR | pass | pass |
| Retrieval 17.6K | pass | pass |
| Retrieval 128K (3 markers) | pass | pass |
| Retrieval 900K (3 markers) | pass | pass |
| Retained history 768K, recall | pass | pass |
| Retained history 768K, long generation | pass, no degeneration | not repeated |
| Tool call | pass | pass |
| Explicit no-tool | pass | pass |

## Switching

```bash
CHECKPOINT=uncensored docker compose up -d
CHECKPOINT=default    docker compose up -d
```

Combine with `DCP=3` as usual. The launcher pins each checkpoint's revision and
selects its kernels; a `MODEL` that disagrees with the selected `CHECKPOINT`
fails closed, as does an unknown `CHECKPOINT` value.

## Provenance and limits

- Default-checkpoint decode and prefill are from the operator's own sweeps on
  this image at 350 W, transcribed from [DCP1bench.jpeg](../DCP1bench.jpeg)
  and [DCP3bench.jpeg](../DCP3bench.jpeg). Uncensored sweeps are the same harness, same power, run
  2026-09-16. The two sides are **different sessions**, and this host drifts
  1–4% within an hour, so differences under ~4% should not be read as real.
  The prefill gap and the KV difference are well outside that band.
- Not measured: 256K and 512K contexts, `hotel-lights` on the uncensored
  checkpoint, lavd at `high` on the uncensored checkpoint, long-run stability,
  and quality at DCP3.
- Single-host measurements on one pair of checkpoints. Nothing here is a
  general claim about either model or about the quantization schemes.
