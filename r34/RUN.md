# R34 TP3 operation

This port carries the R30 TP3 source changes onto the pulled R34 base, including
physical TP3 padding/loaders, three-rank collectives, target/MTP/DFlash/vision
geometry, the qualified GDN profile component, and the tested local explicit
no-tool parser fix. Upstream source and native libraries remain R34.
See PATCH-LEDGER.md and STATUS.md for lineage and qualification results.

## Build

```bash
r27-port/.venv/bin/python r34-port/prepare-build.py
docker build -t glm53-r34-tp3:ported-20260910 r34-port/build
```

Parent is pinned by digest in build/Dockerfile. This is a local image; no Docker
Hub publication has been performed. build/source-delta.json records every shipped
patch file. Diagnostic experiments and rejected optimizations are excluded.

## Serve

From /home/aabduh/glm53-tp3-patch-guide:

```bash
docker compose -f r34-port/compose.yaml up -d
```

Defaults: TP3/EP3/DCP3, MTP3, max model length 1,048,576, max sequences 8,
FP8 KV, FlashKDA recurrent prefill, B12X recurrent decode/attention/linear and
three-rank collectives, EP-capable MoE auto. R34's default B12X MoE selection
is overridden explicitly because this TP3 port requires EP3 support.
The inherited R34 split target page default is 2,048 tokens; the legacy vLLM
block-size CLI remains 256. This port does not substitute the old base launcher.

Model revision remains 46aaae8a82032f77100f2f03e9cc11b391df3b4d.
API model name GLM-5.3-Flash-TP3, host port 15015. Sampling defaults:
temperature 1, top_p .95, repetition penalty 1; history cleanup defaults true
and remains request-overridable. Normal EOS is preserved.

MODE=mtp0 selects no speculation; MODE=dflash2 selects the retained BF16 incoai
DFlash7 checkpoint pin. Set these variables before docker compose.
DCP=3 (default, image `dcp-20260910`) shards the attention KV cache across the
three ranks: 7,510,219-token KV capacity at about −8% decode speed below 128K
context and −1% at 512K. `DCP=1 docker compose -f r34-port/compose.yaml up -d`
restores the single-copy layout (2,091,238 tokens) and full DCP1 speed.
Details: optimization/dcp3-20260910/RESULTS.md.
For scheduler/graph overrides:

```bash
MAX_NUM_BATCHED_TOKENS=8192 MAX_CUDAGRAPH_CAPTURE_SIZE=128 \
  docker compose -f r34-port/compose.yaml up -d
```

Defaults are 4096 / 32 (compose.yaml; the launcher alone defaults to 256). Empty CUDAGRAPH_CAPTURE_SIZES adapts the tested list
below the selected maximum plus that maximum; an explicit space-separated list
is also supported. `none` delegates capture-list generation to vLLM.
Changing environment settings recreates the server on compose up.
The local launcher and persistent r34-port/cache are mounted separately from
R30. GPU power is not changed by Compose; respect the user's 400 W ceiling.

## Rollback

Wait for requests/GPU work to finish, then:

```bash
docker stop glm53-r34-tp3
docker start glm53-r30-tp3
```

R30 and R34 share GPUs and port 15015; run only one. R30 files, image and cache
are preserved. R34 TP3 testing is bounded, not a universal model-quality or
concurrency qualification. Prior R30 retained-history degeneration and occasional
reasoning-only replies are not declared fixed by a source port.
