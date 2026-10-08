import json, pathlib, re, subprocess
R = pathlib.Path(__file__).resolve().parent
Q = R.parents[1] / 'qualification-20261008-csf2' / 'RESULTS.md'
q = Q.read_text() if Q.exists() else ''
def res(name):
    m = re.search(rf'DONE {name} rc=\d+ \| exact (\d+) / wrong (\d+) / truncated (\d+)', q)
    return f'{m.group(1)} exact / {m.group(2)} wrong / {m.group(3)} truncated' if m else 'pending'
import os
dcp3 = R.parents[1] / 'rebase-89f1cee' / 'stress-text-dcp3-2145'
def dcp3_text():
    f = dcp3 / 'stress-results.json'
    if not f.exists(): return 'running', '—', 'stress pending'
    res = json.load(open(f)); kv = [json.loads(l) for l in open(dcp3 / 'stress-kv.log') if 'kv_cache' in l]
    peak = {}
    for line in open(dcp3 / 'stress-mem.csv'):
        p = [x.strip() for x in line.split(',')]
        if len(p) >= 3 and p[1].isdigit(): peak[int(p[1])] = max(peak.get(int(p[1]), 0), int(p[2]))
    ok = sum(1 for x in res if x.get('ok')); pre = max(k['num_preemptions_total'] or 0 for k in kv)
    return (f'{ok}/{len(res)} OK, {int(pre)} preemptions, peak KV {max(k["kv_cache_usage_perc"] or 0 for k in kv):.0%}',
            f'{(97887 - max(peak.values())) / 1024:.2f} GB', 'image + 8x1M stress, 8/8' if ok == len(res) and pre == 0 else 'stress issue')
DCP3_TEXT, DCP3_TEXT_FREE, DCP3_STATUS = dcp3_text()
img = subprocess.check_output(['docker', 'inspect', 'glm53-kraken-tp3:20261008', '--format', '{{.Id}}'], text=True).strip()
compose = (R / 'compose.yaml').read_text()
readme = f'''# GLM-5.3 Flash Kraken TP3 — 20261008

This model is based on [GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD) by Local Inference Lab, Inc., a non-profit organization, available at <https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD>. GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0.

Image: **`azallaza/glm53-kraken-tp3:20261008`** (local `glm53-kraken-tp3:20261008`, `{img}`).
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
| **csf-qad-2 MTP3 DCP3** | **10,485,760 (10.00x 1M)** | 98.4 / 271.1 | 10,185 | {DCP3_STATUS} |
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
| lavd, 30 runs C8 | 30 exact | **{res('lavd')}** |
| hotel-lights, 30 runs C8 (seed 103) | 28 exact / 2 wrong (58/60 over two batches) | **{res('hotel-lights')}** |
| Vision basic (OCR, text, counting, colors, chart), T=0/T=1 | 120/120 (BF16 vision) | **120/120** |
| Vision hard (small OCR, dense count, chart values, shades) | 117/120 | **117/120** |
| Arithmetic | 8/8 | 8/8 |

| Stress (2215 blocks) | Result | Min free |
| --- | --- | ---: |
| 8 concurrent x 10 photos 6000x4000 (79.6K prompt tokens each) | all OK | 2.0 GB |
| 4 x ~891K prompts + 120K decode each (right after the image stress) | all OK, 0 preemptions, peak KV 90% | 1.44 GB (GPU 0) |
| DCP3 (2145 blocks): 8 concurrent x 10 photos 6000x4000 | all OK | 2.44 GB |
| DCP3 (2145 blocks): 8 x ~891K prompts + 120K decode each | {DCP3_TEXT} | {DCP3_TEXT_FREE} |

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
{compose}```
'''
(R / 'README.md').write_text(readme)
print('README written; lavd:', res('lavd'), '| hotel:', res('hotel-lights'))
