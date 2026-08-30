# VTP1–VTP5: Generic TP3 Experiments

This document covers the earlier **VTP1 through VTP5** image line.

These images used the generic Transformers-style path rather than the native NVIDIA `Glm5Next` implementation.

## Goal

Run GLM-5.3 Flash with:

```text
tensor_parallel_size = 3
```

despite several dimensions not being naturally divisible by 3.

The generic route resolved to:

```text
TransformersMultiModalMoEForCausalLM
```

instead of native:

```text
Glm5NextForConditionalGeneration
```

## Images / progression

The experimental sequence was:

```text
VTP1
VTP2
VTP3
VTP4
VTP5
```

Related base images investigated during this phase included:

```text
cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1
verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2
vllm/vllm-openai:glm53-flash-x86_64-cu130
```

plus local-inference-lab variants.

## Generic TP3 padding strategy

### Attention

```text
64 logical heads
-> 66 physical heads
-> 22 heads/GPU
```

### Routed MoE intermediate size

The generic implementation also required padding:

```text
2048 -> 2112
```

This was later recognized as undesirable for the target architecture because proper EP3 can naturally split the experts without padding each routed expert's internal width.

### Vocabulary

```text
154880 -> 154944
154944 / 3 = 51648 entries/GPU
```

## Expert-parallel limitation

GLM-5.3 Flash has:

```text
288 routed experts
```

which naturally divides across EP3:

```text
288 / 3 = 96 experts/GPU
```

The generic Transformers/VTP route did not provide the native GLM-5.3 EP behavior we wanted, so it could not take full advantage of this topology cleanly.

## VTP5 stopping point

VTP5 progressed deep into checkpoint loading, reaching roughly shard 35/43 before failing on ModelOpt metadata associated with the KDA / forget-gate path, around tensors such as:

```text
forget_gate.f_a_proj.weight_scale
```

The generic loader could not correctly reconcile the checkpoint's ModelOpt metadata with the transformed/fused parameter layout.

## Why development stopped after VTP5

Work on the VTP line was **intentionally stopped after VTP5** so engineering effort could be concentrated on the newer **native ModelOpt NVFP4 route**.

VTP5 was not treated as proof that the generic approach could never be made to work. It was simply the point where continuing it offered less value than moving to native `Glm5Next`, which provided:

- native sparse MLA
- the correct native GLM-5.3 layer implementation
- proper EP3 support for 288 routed experts
- a cleaner ModelOpt NVFP4 target
- less need to distort expert geometry
- better alignment with the SM120 kernels actually used by the model

The continuation of the work is documented in:

[Native GLM5Next TP3](./NATIVE-TP3.md)

## Historical value

VTP1–VTP5 remain useful as experimental/reference builds because they demonstrated that:

- TP3 dimension padding is feasible
- vocab padding works
- 64 attention heads can be mechanically adapted for TP3
- the generic checkpoint could progress far into loading
- the remaining problems were increasingly architecture/loader-specific rather than simply "TP3 is impossible"

They are not the recommended starting point for current development.
