# Executive Summary

**Test 1 — Distributed exact retrieval:** Ten unique records were spread throughout increasingly large contexts from ~122K to ~953K API prompt tokens. FP8 KV returned every requested code exactly, finishing **50/50 overall** and **10/10 at 952,751 tokens**.

**Test 2 — Distributed multi-hop reasoning:** Ten chains per context required linking customer → account → tier → multiplier plus a separate base allocation, then performing arithmetic. FP8 KV finished **30/30 overall**, including **10/10 at 963,574 API prompt tokens**.

**Test 3 — Adversarial temporal/revision reasoning:** At ~966K tokens, each answer required resolving superseded records, rejecting obsolete/draft decoys, following a longer graph, applying the active calculation-order policy, and doing arithmetic. The run completed naturally and scored **10/10** with **15,760 output tokens**.

**Test 4 — Real Linux kernel documentation exam:** A deterministic **950,000-token** corpus built from 395 upstream Linux documentation files was followed by 23 technical questions. GLM answered **23/23**, cited valid in-corpus source paths for **23/23**, and finished naturally with `finish_reason=stop`; semantic correctness still requires source-by-source grading.

## Combined Results

| Test | Largest API prompt | Result |
|---|---:|---:|
| Distributed exact retrieval | 952,751 | 10/10 at largest context |
| Distributed multi-hop reasoning | 963,574 | 10/10 at largest context |
| Adversarial temporal/revision reasoning | 966,393 | 10/10 |
| Real Linux documentation exam | 950,462 | 23/23 answered, 23/23 valid source paths |

Synthetic exact checks completed successfully:

```text
Simple retrieval:      50/50
Multi-hop reasoning:   30/30
Adversarial temporal:  10/10

Synthetic total:       90/90
```

The Linux documentation exam is reported separately because its current score is structural/source-grounding rather than a completed semantic correctness grade.

---

# GLM-5.3 Flash TP3 — FP8 Long-Context KV Cache Validation

## Overview

This test validated the current **FP8 KV cache** configuration across progressively larger contexts, from roughly 128K to nearly 1M prompt tokens.

- API model: `GLM-5.3-Flash-TP3`
- Endpoint: `http://127.0.0.1:15015/v1/chat/completions`
- KV cache dtype: **FP8**
- Temperature: `0`
- Top-p: `1.0`
- Seed: `0`
- Ten distributed retrieval targets per context
- Exact-match scoring

### Result

| Target | API prompt tokens | Score | Elapsed |
|---:|---:|---:|---:|
| 131,072 | 121,602 | 10/10 | 13.82 s |
| 262,144 | 246,977 | 10/10 | 27.84 s |
| 524,288 | 497,723 | 10/10 | 60.47 s |
| 786,432 | 748,468 | 10/10 | 103.51 s |
| 1,000,000 | 952,751 | 10/10 | 259.08 s |

**Overall: 50/50 exact retrievals (100%).**

## What the script does

For each target context size, the benchmark:

1. Loads the local GLM-5.3 tokenizer.
2. Deterministically generates 10 unique authorization-code records.
3. Generates neutral filler text.
4. Places the 10 records at approximately evenly spaced positions throughout the context.
5. Appends one query asking for all 10 records.
6. Sends the request to the current vLLM server with deterministic generation settings.
7. Parses the returned JSON.
8. Scores every returned code by exact match.
9. Records prompt size, API token count, output token count, elapsed time, prompt SHA-256, and per-record pass/fail.
10. Saves the exact generated prompts so a later KV-cache configuration can be tested against byte-identical inputs.

Saved output locations:

```text
/home/aabduh/glm53-kv-ab/results-fp8.jsonl
/home/aabduh/glm53-kv-ab/context-131072.txt
/home/aabduh/glm53-kv-ab/context-262144.txt
/home/aabduh/glm53-kv-ab/context-524288.txt
/home/aabduh/glm53-kv-ab/context-786432.txt
/home/aabduh/glm53-kv-ab/context-1000000.txt
```

## Exact benchmark script

<details>
<summary><strong>Expand full glm53-longctx-kv-test.py</strong></summary>

```python
#!/usr/bin/env python3

import hashlib
import json
import os
import random
import string
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

URL = "http://127.0.0.1:15015/v1/chat/completions"
MODEL = "GLM-5.3-Flash-TP3"
KV_LABEL = os.environ.get("KV_LABEL", "FP8")
MODEL_REPO = "local-inference-lab/GLM-5.3-Flash-NVFP4"

TARGET_CONTEXTS = [
    131_072,
    262_144,
    524_288,
    786_432,
    1_000_000,
]

NEEDLES_PER_CONTEXT = 10
CONTENT_RESERVE = 4096
MAX_OUTPUT_TOKENS = 2048
SEED = 530053

OUTDIR = Path.home() / "glm53-kv-ab"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESULT_FILE = OUTDIR / f"results-{KV_LABEL.lower()}.jsonl"

random.seed(SEED)

print(f"Loading tokenizer for {MODEL_REPO} ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    local_files_only=True,
    trust_remote_code=True,
)
print("Tokenizer loaded.")

def make_code(rng):
    chars = string.ascii_uppercase + string.digits
    parts = [
        "".join(rng.choice(chars) for _ in range(4)),
        "".join(rng.choice(chars) for _ in range(4)),
        "".join(rng.choice(chars) for _ in range(4)),
    ]
    return "-".join(parts)

def build_context(target_tokens, test_index):
    rng = random.Random(SEED + test_index)
    needles = []

    for i in range(NEEDLES_PER_CONTEXT):
        code = make_code(rng)
        needles.append({
            "id": f"RECORD_{i+1:02d}",
            "code": code,
        })

    needle_texts = [
        (
            f"\n\nIMPORTANT DATABASE RECORD {n['id']}:\n"
            f"The authorization code for {n['id']} is {n['code']}.\n"
            f"End of database record.\n\n"
        )
        for n in needles
    ]

    needle_token_counts = [
        len(tokenizer.encode(x, add_special_tokens=False))
        for x in needle_texts
    ]

    desired_content_tokens = target_tokens - CONTENT_RESERVE
    total_needle_tokens = sum(needle_token_counts)

    if total_needle_tokens >= desired_content_tokens:
        raise RuntimeError("Needles exceed target context budget")

    filler_needed = desired_content_tokens - total_needle_tokens

    filler_phrase = (
        "This archival record contains routine operational information. "
        "The material is intentionally repetitive and carries no special "
        "instruction or authorization value. "
    )

    filler_phrase_ids = tokenizer.encode(
        filler_phrase,
        add_special_tokens=False,
    )

    repeats = (filler_needed // len(filler_phrase_ids)) + 2
    filler_ids = (filler_phrase_ids * repeats)[:filler_needed]

    chunks = []

    for i in range(NEEDLES_PER_CONTEXT + 1):
        start = round(i * len(filler_ids) / (NEEDLES_PER_CONTEXT + 1))
        end = round((i + 1) * len(filler_ids) / (NEEDLES_PER_CONTEXT + 1))
        chunks.append(
            tokenizer.decode(
                filler_ids[start:end],
                skip_special_tokens=True,
            )
        )

    parts = [chunks[0]]

    for i, needle_text in enumerate(needle_texts):
        parts.append(needle_text)
        parts.append(chunks[i + 1])

    context = "".join(parts)

    question_lines = [
        f"{n['id']}: ?"
        for n in needles
    ]

    query = (
        "\n\nEND OF ARCHIVE.\n\n"
        "Retrieve the authorization code for every requested database record "
        "from the archive above.\n\n"
        "Return ONLY one JSON object using exactly this format:\n"
        '{"RECORD_01":"CODE","RECORD_02":"CODE",...}\n\n'
        "Do not explain your answer. Do not omit any record.\n\n"
        "Requested records:\n"
        + "\n".join(question_lines)
    )

    full_user_text = context + query

    raw_tokens = len(
        tokenizer.encode(
            full_user_text,
            add_special_tokens=False,
        )
    )

    return full_user_text, needles, raw_tokens

def parse_answer(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end+1])
    except Exception:
        return None

summary = []

print()
print("=" * 78)
print(f"GLM-5.3 LONG-CONTEXT KV TEST — {KV_LABEL}")
print("=" * 78)

for test_index, target in enumerate(TARGET_CONTEXTS):

    print()
    print("=" * 78)
    print(f"TARGET CONTEXT: {target:,}")
    print("=" * 78)

    user_text, needles, raw_tokens = build_context(
        target,
        test_index,
    )

    sha = hashlib.sha256(
        user_text.encode("utf-8")
    ).hexdigest()

    prompt_file = OUTDIR / f"context-{target}.txt"

    if not prompt_file.exists():
        prompt_file.write_text(
            user_text,
            encoding="utf-8",
        )
    else:
        existing = prompt_file.read_text(encoding="utf-8")
        existing_sha = hashlib.sha256(
            existing.encode("utf-8")
        ).hexdigest()

        if existing_sha != sha:
            raise RuntimeError(
                f"Existing prompt mismatch for {target}"
            )

    print(f"Raw user tokens: {raw_tokens:,}")
    print(f"Prompt SHA256:    {sha}")

    start = time.perf_counter()

    r = requests.post(
        URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": user_text,
                }
            ],
            "temperature": 0,
            "top_p": 1.0,
            "seed": 0,
            "max_tokens": MAX_OUTPUT_TOKENS,
        },
        timeout=1800,
    )

    elapsed = time.perf_counter() - start

    if r.status_code != 200:
        print(f"HTTP ERROR: {r.status_code}")
        print(r.text[:4000])

        record = {
            "kv": KV_LABEL,
            "target_context": target,
            "raw_user_tokens": raw_tokens,
            "prompt_sha256": sha,
            "http_status": r.status_code,
            "elapsed_sec": elapsed,
            "passed": 0,
            "total": NEEDLES_PER_CONTEXT,
            "error": r.text[:4000],
        }

        with RESULT_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")

        continue

    obj = r.json()
    message = obj["choices"][0]["message"]

    answer_text = (
        message.get("content")
        or ""
    )

    reasoning_text = (
        message.get("reasoning")
        or message.get("reasoning_content")
        or ""
    )

    usage = obj.get("usage", {})
    parsed = parse_answer(answer_text)

    passed = 0
    detail = []

    for needle in needles:
        expected = needle["code"]
        actual = None
        ok = False

        if isinstance(parsed, dict):
            actual = parsed.get(needle["id"])
            ok = actual == expected

        if ok:
            passed += 1

        detail.append({
            "id": needle["id"],
            "expected": expected,
            "actual": actual,
            "pass": ok,
        })

    print()
    for x in detail:
        status = "PASS" if x["pass"] else "FAIL"
        print(
            f"{x['id']}: {status} | "
            f"expected={x['expected']} | "
            f"actual={x['actual']}"
        )

    print()
    print(f"RESULT:            {passed}/{NEEDLES_PER_CONTEXT}")
    print(f"Elapsed:           {elapsed:.2f} sec")
    print(f"Prompt tokens API: {usage.get('prompt_tokens')}")
    print(f"Output tokens:     {usage.get('completion_tokens')}")
    print(f"Reasoning chars:   {len(reasoning_text)}")

    record = {
        "kv": KV_LABEL,
        "target_context": target,
        "raw_user_tokens": raw_tokens,
        "prompt_sha256": sha,
        "http_status": 200,
        "elapsed_sec": elapsed,
        "api_prompt_tokens": usage.get("prompt_tokens"),
        "api_completion_tokens": usage.get("completion_tokens"),
        "passed": passed,
        "total": NEEDLES_PER_CONTEXT,
        "details": detail,
        "answer": answer_text,
    }

    with RESULT_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    summary.append(
        (
            target,
            usage.get("prompt_tokens"),
            passed,
            elapsed,
        )
    )

print()
print("=" * 78)
print(f"SUMMARY — {KV_LABEL}")
print("=" * 78)

for target, actual, passed, elapsed in summary:
    print(
        f"{target:>9,} target | "
        f"{actual or 0:>9,} API prompt | "
        f"{passed:>2}/{NEEDLES_PER_CONTEXT} | "
        f"{elapsed:>8.2f}s"
    )

print()
print(f"Results saved to: {RESULT_FILE}")
print(f"Prompts saved to: {OUTDIR}")
```

</details>

## Complete FP8 console output

<details>
<summary><strong>Expand full results</strong></summary>

```text
(glm53-kv-test-venv) aabduh@threadripper-aabduh:~$ KV_LABEL=FP8 python3 ~/glm53-longctx-kv-test.py
[transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Loading tokenizer for local-inference-lab/GLM-5.3-Flash-NVFP4 ...
Ignoring corrupted tree cache file /m2-2/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/trees/46aaae8a82032f77100f2f03e9cc11b391df3b4d.json: [Errno 13] Permission denied: '/m2-2/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/trees/46aaae8a82032f77100f2f03e9cc11b391df3b4d.json'
Tokenizer loaded.

==============================================================================
GLM-5.3 LONG-CONTEXT KV TEST — FP8
==============================================================================

==============================================================================
TARGET CONTEXT: 131,072
==============================================================================
Raw user tokens: 121,590
Prompt SHA256:    807db8ebcb4020e23d28f7bebafffd7518914f17a0c5a64311dcf422fbc65b24

RECORD_01: PASS | expected=9QMT-45N0-LYXI | actual=9QMT-45N0-LYXI
RECORD_02: PASS | expected=32LT-FFM9-SE4F | actual=32LT-FFM9-SE4F
RECORD_03: PASS | expected=U6VS-ID4E-DIHZ | actual=U6VS-ID4E-DIHZ
RECORD_04: PASS | expected=BO67-4KD0-QY25 | actual=BO67-4KD0-QY25
RECORD_05: PASS | expected=LVY1-08Q1-XPNQ | actual=LVY1-08Q1-XPNQ
RECORD_06: PASS | expected=BJEO-JPHG-FWQW | actual=BJEO-JPHG-FWQW
RECORD_07: PASS | expected=SLQT-VAVS-YEGK | actual=SLQT-VAVS-YEGK
RECORD_08: PASS | expected=QQSB-MP1A-T793 | actual=QQSB-MP1A-T793
RECORD_09: PASS | expected=P845-G5JE-RGS5 | actual=P845-G5JE-RGS5
RECORD_10: PASS | expected=2ZBT-WMB3-OSZW | actual=2ZBT-WMB3-OSZW

RESULT:            10/10
Elapsed:           13.82 sec
Prompt tokens API: 121602
Output tokens:     340
Reasoning chars:   407

==============================================================================
TARGET CONTEXT: 262,144
==============================================================================
Raw user tokens: 246,965
Prompt SHA256:    ca930f6572203b47a55ba8457d3e86edb4d0f7ef0cefbe0d1ff562cd15f6e980

RECORD_01: PASS | expected=99ZN-ONC2-C111 | actual=99ZN-ONC2-C111
RECORD_02: PASS | expected=YZG3-DF23-V6FD | actual=YZG3-DF23-V6FD
RECORD_03: PASS | expected=8ND5-C6PA-C5BB | actual=8ND5-C6PA-C5BB
RECORD_04: PASS | expected=60JQ-WCC4-NBEN | actual=60JQ-WCC4-NBEN
RECORD_05: PASS | expected=E74T-AKHZ-II1B | actual=E74T-AKHZ-II1B
RECORD_06: PASS | expected=J4J0-GQ4Q-LKTS | actual=J4J0-GQ4Q-LKTS
RECORD_07: PASS | expected=NL17-ZBWS-N54I | actual=NL17-ZBWS-N54I
RECORD_08: PASS | expected=TLV5-HRHH-T89G | actual=TLV5-HRHH-T89G
RECORD_09: PASS | expected=IZHS-MP7W-E39M | actual=IZHS-MP7W-E39M
RECORD_10: PASS | expected=XESJ-LY0P-AW1N | actual=XESJ-LY0P-AW1N

RESULT:            10/10
Elapsed:           27.84 sec
Prompt tokens API: 246977
Output tokens:     350
Reasoning chars:   491

==============================================================================
TARGET CONTEXT: 524,288
==============================================================================
Raw user tokens: 497,711
Prompt SHA256:    fa0da01e13447ef678336908f0d11af3bc5f0b8e879b77e2abf4a387912ab54c

RECORD_01: PASS | expected=KCST-NWBA-8XJE | actual=KCST-NWBA-8XJE
RECORD_02: PASS | expected=1YJC-34ZT-NEFR | actual=1YJC-34ZT-NEFR
RECORD_03: PASS | expected=3EYH-JLPG-OAFN | actual=3EYH-JLPG-OAFN
RECORD_04: PASS | expected=9O17-97R5-O20E | actual=9O17-97R5-O20E
RECORD_05: PASS | expected=W9MT-ZRAG-LZDK | actual=W9MT-ZRAG-LZDK
RECORD_06: PASS | expected=LLQ5-S3BV-A25R | actual=LLQ5-S3BV-A25R
RECORD_07: PASS | expected=JRDV-G9JT-91WM | actual=JRDV-G9JT-91WM
RECORD_08: PASS | expected=6PQC-XK4I-JNIV | actual=6PQC-XK4I-JNIV
RECORD_09: PASS | expected=6MGR-YFMT-6YER | actual=6MGR-YFMT-6YER
RECORD_10: PASS | expected=QB4W-B87L-XDBM | actual=QB4W-B87L-XDBM

RESULT:            10/10
Elapsed:           60.47 sec
Prompt tokens API: 497723
Output tokens:     348
Reasoning chars:   407

==============================================================================
TARGET CONTEXT: 786,432
==============================================================================
Raw user tokens: 748,456
Prompt SHA256:    6b022852730d19e68384fd2c8f00c33813771306005d76e0e2b16a0dd784cfd3

RECORD_01: PASS | expected=YNM7-51Y9-Q6WD | actual=YNM7-51Y9-Q6WD
RECORD_02: PASS | expected=TUT5-8R4O-OGO7 | actual=TUT5-8R4O-OGO7
RECORD_03: PASS | expected=SQPW-RCOG-Z9O3 | actual=SQPW-RCOG-Z9O3
RECORD_04: PASS | expected=QNU8-7USV-J1VD | actual=QNU8-7USV-J1VD
RECORD_05: PASS | expected=QY1O-6QOP-CYP1 | actual=QY1O-6QOP-CYP1
RECORD_06: PASS | expected=PZ7R-8EUD-QWGM | actual=PZ7R-8EUD-QWGM
RECORD_07: PASS | expected=RX7Z-MGGO-LZCS | actual=RX7Z-MGGO-LZCS
RECORD_08: PASS | expected=EAAL-SQFY-M51K | actual=EAAL-SQFY-M51K
RECORD_09: PASS | expected=OX63-EE2U-LASD | actual=OX63-EE2U-LASD
RECORD_10: PASS | expected=TAEW-287U-F66T | actual=TAEW-287U-F66T

RESULT:            10/10
Elapsed:           103.51 sec
Prompt tokens API: 748468
Output tokens:     356
Reasoning chars:   446

==============================================================================
TARGET CONTEXT: 1,000,000
==============================================================================
Raw user tokens: 952,739
Prompt SHA256:    3174fd416f9221b0850a587d2f862980632718caaad7bfc6c8972af576e21966

RECORD_01: PASS | expected=398B-G82K-CW0T | actual=398B-G82K-CW0T
RECORD_02: PASS | expected=M71F-OCXQ-QBFC | actual=M71F-OCXQ-QBFC
RECORD_03: PASS | expected=ZVY0-6CYO-FSIK | actual=ZVY0-6CYO-FSIK
RECORD_04: PASS | expected=VF9G-1RQ8-7RVC | actual=VF9G-1RQ8-7RVC
RECORD_05: PASS | expected=29KB-D8KC-40ZU | actual=29KB-D8KC-40ZU
RECORD_06: PASS | expected=33S6-U5WM-J1KZ | actual=33S6-U5WM-J1KZ
RECORD_07: PASS | expected=4TTH-GK3T-Q71Y | actual=4TTH-GK3T-Q71Y
RECORD_08: PASS | expected=M4A7-1KGF-C9ZT | actual=M4A7-1KGF-C9ZT
RECORD_09: PASS | expected=FEYC-MLS3-K7TL | actual=FEYC-MLS3-K7TL
RECORD_10: PASS | expected=WNG6-OO0Y-RO0J | actual=WNG6-OO0Y-RO0J

RESULT:            10/10
Elapsed:           259.08 sec
Prompt tokens API: 952751
Output tokens:     369
Reasoning chars:   445

==============================================================================
SUMMARY — FP8
==============================================================================
  131,072 target |   121,602 API prompt | 10/10 |    13.82s
  262,144 target |   246,977 API prompt | 10/10 |    27.84s
  524,288 target |   497,723 API prompt | 10/10 |    60.47s
  786,432 target |   748,468 API prompt | 10/10 |   103.51s
1,000,000 target |   952,751 API prompt | 10/10 |   259.08s

Results saved to: /home/aabduh/glm53-kv-ab/results-fp8.jsonl
Prompts saved to: /home/aabduh/glm53-kv-ab
(glm53-kv-test-venv) aabduh@threadripper-aabduh:~$
```

</details>

## Interpretation

The FP8 KV-cache path passed simple exact retrieval at every tested size, including **10/10 with 952,751 API prompt tokens**. That is strong evidence that FP8 KV is functioning correctly for distributed lookup very close to the configured 1,048,576-token maximum.

This benchmark is intentionally a retrieval test. It does **not** establish that all forms of long-context reasoning are equally strong.

## Recommended next validation

A stronger follow-up is **distributed multi-hop reasoning** at approximately 500K, 750K, and 950K tokens.

Instead of putting the final answer directly into the context, scatter dependent facts far apart:

```text
Near 5%:
Customer ORCHID uses account identifier A17.

Near 35%:
Account A17 maps to service tier TIER-C.

Near 65%:
TIER-C receives multiplier 1.35.

Near 90%:
ORCHID's base allocation is 2400 units.
```

Then ask:

```text
What is ORCHID's final allocation after applying its service-tier multiplier?
Return only the integer.
```

The required chain is:

```text
ORCHID -> A17 -> TIER-C -> 1.35 -> 2400 -> 3240
```

A good benchmark should add many decoy customers, account IDs, tiers, and multipliers. That tests long-range association and reasoning rather than simple string retrieval.


---

# Follow-up Validation — Distributed Multi-Hop Reasoning

## Purpose

After the simple distributed needle-retrieval test passed 50/50, a harder benchmark was run to test whether FP8 KV cache could preserve **relationships between facts that were widely separated across very large contexts**.

Each answer required the model to recover and combine four facts:

```text
customer
  -> account identifier
  -> service tier
  -> multiplier

customer
  -> base allocation

final allocation = base allocation * multiplier
```

The benchmark also inserted large numbers of decoy records so the requested answer could not be solved by retrieving a single nearby string.

## Multi-hop result summary

| Target context | API prompt tokens | Score | Elapsed | Output tokens | Reasoning chars |
|---:|---:|---:|---:|---:|---:|
| 524,288 | 501,844 | 10/10 | 64.50 s | 1,860 | 4,796 |
| 786,432 | 756,280 | 10/10 | 109.92 s | 2,410 | 6,665 |
| 1,000,000 | 963,574 | 10/10 | 270.44 s | 2,592 | 7,260 |

### Overall

```text
30 / 30 exact answers
100% pass rate
Largest tested API prompt: 963,574 tokens
```

Combined with the first retrieval benchmark:

```text
Simple distributed retrieval: 50/50
Distributed multi-hop reasoning: 30/30

Combined observed result: 80/80
```

## Exact multi-hop console output

<details>
<summary><strong>Expand complete FP8 multi-hop test output</strong></summary>

```text
Welcome to Ubuntu 26.04 LTS (GNU/Linux 7.0.0-30-generic x86_64)

 * Documentation:  https://docs.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/pro

Expanded Security Maintenance for Applications is not enabled.

341 updates can be applied immediately.
1 of these updates is a standard security update.
To see these additional updates run: apt list --upgradable

20 additional security updates can be applied with ESM Apps.
Learn more about enabling ESM Apps service at https://ubuntu.com/esm

Web console: https://threadripper-aabduh:9091/ or https://192.168.0.8:9091/

Last login: Fri Sep  4 16:32:37 2026 from 192.168.0.4
aabduh@threadripper-aabduh:~$ python3 -m venv ~/glm53-kv-test-venv

source ~/glm53-kv-test-venv/bin/activate
(glm53-kv-test-venv) aabduh@threadripper-aabduh:~$ KV_LABEL=FP8 python3 ~/glm53-longctx-multihop-test.py
[transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Loading tokenizer for local-inference-lab/GLM-5.3-Flash-NVFP4 ...
Ignoring corrupted tree cache file /m2-2/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/trees/46aaae8a82032f77100f2f03e9cc11b391df3b4d.json: [Errno 13] Permission denied: '/m2-2/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/trees/46aaae8a82032f77100f2f03e9cc11b391df3b4d.json'
Tokenizer loaded.

================================================================================
GLM-5.3 LONG-CONTEXT MULTI-HOP TEST — FP8
================================================================================

================================================================================
TARGET CONTEXT: 524,288
================================================================================
Raw user tokens: 501,832
Prompt SHA256:    6388b5a4edac7c6110e34fb3b4834eb75668a0786f42a927e62ad969de2c086d

Expected answers:
  CHAIN_01 | SABLE -> A1554 -> TIER-P90 -> 1.35 | base=6500 | expected=8775
  CHAIN_02 | ZEPHYR -> A7238 -> TIER-N82 -> 1.10 | base=6800 | expected=7480
  CHAIN_03 | EMBER -> A1973 -> TIER-Z20 -> 1.50 | base=6600 | expected=9900
  CHAIN_04 | LANTERN -> A8445 -> TIER-L74 -> 1.10 | base=1700 | expected=1870
  CHAIN_05 | RAVEN -> A9908 -> TIER-G17 -> 1.40 | base=1400 | expected=1960
  CHAIN_06 | MERCURY -> A9406 -> TIER-H11 -> 1.05 | base=7300 | expected=7665
  CHAIN_07 | HARBOR -> A1474 -> TIER-A90 -> 1.45 | base=6300 | expected=9135
  CHAIN_08 | QUARTZ -> A3454 -> TIER-J87 -> 1.30 | base=1500 | expected=1950
  CHAIN_09 | ORCHID -> A1592 -> TIER-R37 -> 1.05 | base=1800 | expected=1890
  CHAIN_10 | MAPLE -> A4477 -> TIER-C77 -> 1.40 | base=4900 | expected=6860

Sending request ...

Results:
CHAIN_01: PASS | expected=8775 | actual=8775
CHAIN_02: PASS | expected=7480 | actual=7480
CHAIN_03: PASS | expected=9900 | actual=9900
CHAIN_04: PASS | expected=1870 | actual=1870
CHAIN_05: PASS | expected=1960 | actual=1960
CHAIN_06: PASS | expected=7665 | actual=7665
CHAIN_07: PASS | expected=9135 | actual=9135
CHAIN_08: PASS | expected=1950 | actual=1950
CHAIN_09: PASS | expected=1890 | actual=1890
CHAIN_10: PASS | expected=6860 | actual=6860

RESULT:            10/10
Elapsed:           64.50 sec
Prompt tokens API: 501844
Output tokens:     1860
Reasoning chars:   4796

================================================================================
TARGET CONTEXT: 786,432
================================================================================
Raw user tokens: 756,268
Prompt SHA256:    ceb5616db6799321347a0d6a4ca8745dc9c2982e2bf936aac9b6cdc3713905bf

Expected answers:
  CHAIN_01 | COBALT -> A9881 -> TIER-M29 -> 1.10 | base=6400 | expected=7040
  CHAIN_02 | FALCON -> A7211 -> TIER-E84 -> 1.05 | base=6900 | expected=7245
  CHAIN_03 | VEGA -> A8770 -> TIER-N92 -> 1.25 | base=3600 | expected=4500
  CHAIN_04 | SABLE -> A2245 -> TIER-C44 -> 1.40 | base=1900 | expected=2660
  CHAIN_05 | LANTERN -> A7318 -> TIER-D29 -> 1.15 | base=4000 | expected=4600
  CHAIN_06 | ZEPHYR -> A2576 -> TIER-H10 -> 1.10 | base=3600 | expected=3960
  CHAIN_07 | ORCHID -> A4669 -> TIER-P82 -> 1.45 | base=8000 | expected=11600
  CHAIN_08 | EMBER -> A9533 -> TIER-J73 -> 1.20 | base=6700 | expected=8040
  CHAIN_09 | ATLAS -> A7809 -> TIER-C54 -> 1.45 | base=3400 | expected=4930
  CHAIN_10 | PHOENIX -> A5866 -> TIER-Z91 -> 1.35 | base=4500 | expected=6075

Sending request ...

Results:
CHAIN_01: PASS | expected=7040 | actual=7040
CHAIN_02: PASS | expected=7245 | actual=7245
CHAIN_03: PASS | expected=4500 | actual=4500
CHAIN_04: PASS | expected=2660 | actual=2660
CHAIN_05: PASS | expected=4600 | actual=4600
CHAIN_06: PASS | expected=3960 | actual=3960
CHAIN_07: PASS | expected=11600 | actual=11600
CHAIN_08: PASS | expected=8040 | actual=8040
CHAIN_09: PASS | expected=4930 | actual=4930
CHAIN_10: PASS | expected=6075 | actual=6075

RESULT:            10/10
Elapsed:           109.92 sec
Prompt tokens API: 756280
Output tokens:     2410
Reasoning chars:   6665

================================================================================
TARGET CONTEXT: 1,000,000
================================================================================
Raw user tokens: 963,562
Prompt SHA256:    96655409ab5a89ade0a00179c979cc182a022ee334114958910ed88b79b3e4c4

Expected answers:
  CHAIN_01 | EMBER -> A9320 -> TIER-M17 -> 1.25 | base=5100 | expected=6375
  CHAIN_02 | LANTERN -> A5983 -> TIER-W72 -> 1.50 | base=7800 | expected=11700
  CHAIN_03 | TUNDRA -> A5495 -> TIER-R39 -> 1.20 | base=2300 | expected=2760
  CHAIN_04 | PHOENIX -> A4695 -> TIER-S94 -> 1.25 | base=4300 | expected=5375
  CHAIN_05 | MAPLE -> A4892 -> TIER-X88 -> 1.30 | base=4400 | expected=5720
  CHAIN_06 | SABLE -> A1542 -> TIER-X38 -> 1.10 | base=6000 | expected=6600
  CHAIN_07 | NOVA -> A9981 -> TIER-H68 -> 1.50 | base=4200 | expected=6300
  CHAIN_08 | SUMMIT -> A4552 -> TIER-L78 -> 1.45 | base=5100 | expected=7395
  CHAIN_09 | CEDAR -> A5804 -> TIER-L29 -> 1.35 | base=9000 | expected=12150
  CHAIN_10 | HARBOR -> A6613 -> TIER-B42 -> 1.35 | base=6400 | expected=8640

Sending request ...

Results:
CHAIN_01: PASS | expected=6375 | actual=6375
CHAIN_02: PASS | expected=11700 | actual=11700
CHAIN_03: PASS | expected=2760 | actual=2760
CHAIN_04: PASS | expected=5375 | actual=5375
CHAIN_05: PASS | expected=5720 | actual=5720
CHAIN_06: PASS | expected=6600 | actual=6600
CHAIN_07: PASS | expected=6300 | actual=6300
CHAIN_08: PASS | expected=7395 | actual=7395
CHAIN_09: PASS | expected=12150 | actual=12150
CHAIN_10: PASS | expected=8640 | actual=8640

RESULT:            10/10
Elapsed:           270.44 sec
Prompt tokens API: 963574
Output tokens:     2592
Reasoning chars:   7260

================================================================================
SUMMARY — FP8
================================================================================
  524,288 target |   501,844 API prompt | 10/10 |    64.50s
  786,432 target |   756,280 API prompt | 10/10 |   109.92s
1,000,000 target |   963,574 API prompt | 10/10 |   270.44s

OVERALL: 30/30

Results saved to: /home/aabduh/glm53-kv-multihop/results-fp8.jsonl
Prompts saved to: /home/aabduh/glm53-kv-multihop
```

</details>

## Interpretation of the multi-hop test

The 963,574-token result is substantially stronger than a simple needle lookup.

At that context length, the model successfully recovered all four required facts for each of ten independent chains, followed the associations between them, performed the required arithmetic, and returned all ten exact answers.

Observed result:

```text
10 chains
4 linked facts per chain
40 relevant facts distributed through ~964K prompt tokens
large decoy population
10/10 exact final answers
```

This gives strong evidence that the current FP8 KV path remains functional not only for direct retrieval but also for long-distance relational association and simple reasoning near the top of the configured context window.

---

# A Harder Next Test

The next meaningful step is an **adversarial temporal graph / conflicting-record test**.

The current multi-hop benchmark is difficult, but every relevant fact is internally consistent. A stronger test should force the model to:

1. Follow a longer chain, such as 6–8 hops instead of 4.
2. Distinguish active records from obsolete records.
3. Resolve later updates that supersede earlier values.
4. Ignore plausible but incorrect decoy chains.
5. Combine facts from both the very beginning and very end of the context.
6. Perform more than one arithmetic or logical operation.
7. Answer several independent queries in the same ~950K-token request.

Example:

```text
5%:
Project ORCHID initially maps to customer C114.

14%:
Customer C114 maps to account A17.

26%:
Account A17 initially maps to region WEST.

38%:
Revision 004 supersedes that mapping:
A17 now maps to region EAST.

51%:
Region EAST maps to service group G9.

63%:
G9 initially carries multiplier 1.20.

72%:
Policy revision P18 changes G9 multiplier to 1.35,
effective after revision 004.

84%:
ORCHID has base allocation 2400.

91%:
ORCHID also has an adjustment of +125.

96%:
For calculations after P18, apply the adjustment before
multiplying by the active service multiplier.
```

Required reasoning:

```text
ORCHID
 -> C114
 -> A17
 -> latest valid region = EAST
 -> G9
 -> latest valid multiplier = 1.35

base 2400
+ adjustment 125
= 2525

2525 * 1.35
= 3408.75
```

The context would also contain obsolete WEST mappings, old multipliers, duplicate customer names, conflicting decoys, revision numbers, and unrelated records.

That is materially harder because success requires **retrieval + graph traversal + temporal precedence + contradiction resolution + ordering rules + arithmetic**, rather than retrieval and multiplication alone.

If FP8 remains essentially perfect around 950K on that test, there would be little practical evidence from these synthetic workloads that BF16 KV would improve your long-context use case.

---

# Test 3 — Adversarial Temporal / Revision Reasoning

## Purpose

The third benchmark increased difficulty beyond both direct retrieval and ordinary multi-hop reasoning.

Each requested answer required the model to:

1. Resolve a project to a customer.
2. Resolve the customer to an account.
3. Locate an older region mapping.
4. Locate a later revision that superseded that mapping.
5. Follow the active region to a service group.
6. Locate an older service-group multiplier.
7. Locate a later policy revision that superseded that multiplier.
8. Retrieve the project's base allocation.
9. Retrieve the project's adjustment.
10. Retrieve the active calculation-order rule.
11. Ignore rejected, obsolete, retired, historical, and unapproved decoy records.
12. Apply the correct operation order and calculate the final integer result.

The real facts were distributed throughout approximately one million tokens, with stages intentionally positioned across successive portions of the context.

## Result

| Metric | Result |
|---|---:|
| Raw user tokens | 966,381 |
| API prompt tokens | 966,393 |
| Output tokens | 15,760 |
| Reasoning characters | 52,328 |
| Elapsed | 60.97 s |
| Exact answers | **10/10** |

### Overall result

```text
10 / 10 exact final answers
100% pass rate
API prompt length: 966,393 tokens
```

## Exact expected and returned answers

```text
CHAIN_01: PASS | expected=9940  | actual=9940
CHAIN_02: PASS | expected=8950  | actual=8950
CHAIN_03: PASS | expected=4675  | actual=4675
CHAIN_04: PASS | expected=3105  | actual=3105
CHAIN_05: PASS | expected=4060  | actual=4060
CHAIN_06: PASS | expected=4156  | actual=4156
CHAIN_07: PASS | expected=6561  | actual=6561
CHAIN_08: PASS | expected=5465  | actual=5465
CHAIN_09: PASS | expected=9745  | actual=9745
CHAIN_10: PASS | expected=10385 | actual=10385
```

## Exact chain definitions

<details>
<summary><strong>Expand all 10 adversarial chains</strong></summary>

```text
CHAIN_01 | COBALT -> C9881 -> A6903 -> SOUTH -> G29 -> 1.15
base=8600 | adj=+50 | rule=MULTIPLY_THEN_ADJUST
(8600 * 1.15) + 50 -> 9940

CHAIN_02 | FALCON -> C5925 -> A4427 -> ATLANTIC -> G93 -> 1.25
base=7100 | adj=+75 | rule=MULTIPLY_THEN_ADJUST
(7100 * 1.25) + 75 -> 8950

CHAIN_03 | VEGA -> C4055 -> A4905 -> ATLANTIC -> G22 -> 1.10
base=3800 | adj=+450 | rule=ADJUST_THEN_MULTIPLY
(3800 + 450) * 1.10 -> 4675

CHAIN_04 | SABLE -> C9974 -> A9533 -> WEST -> G45 -> 1.35
base=2000 | adj=+300 | rule=ADJUST_THEN_MULTIPLY
(2000 + 300) * 1.35 -> 3105

CHAIN_05 | LANTERN -> C1082 -> A2655 -> SOUTH -> G82 -> 1.15
base=3400 | adj=+150 | rule=MULTIPLY_THEN_ADJUST
(3400 * 1.15) + 150 -> 4060

CHAIN_06 | ZEPHYR -> C8610 -> A1413 -> SOUTH -> G52 -> 1.25
base=3100 | adj=+225 | rule=ADJUST_THEN_MULTIPLY
(3100 + 225) * 1.25 -> 4156

CHAIN_07 | ORCHID -> C3397 -> A6027 -> NORTH -> G80 -> 1.45
base=4300 | adj=+225 | rule=ADJUST_THEN_MULTIPLY
(4300 + 225) * 1.45 -> 6561

CHAIN_08 | EMBER -> C8911 -> A3101 -> CENTRAL -> G36 -> 1.45
base=3700 | adj=+100 | rule=MULTIPLY_THEN_ADJUST
(3700 * 1.45) + 100 -> 5465

CHAIN_09 | ATLAS -> C6032 -> A9258 -> NORTH -> G59 -> 1.05
base=8900 | adj=+400 | rule=MULTIPLY_THEN_ADJUST
(8900 * 1.05) + 400 -> 9745

CHAIN_10 | PHOENIX -> C9593 -> A3845 -> PACIFIC -> G56 -> 1.20
base=8300 | adj=+425 | rule=MULTIPLY_THEN_ADJUST
(8300 * 1.20) + 425 -> 10385
```

</details>

## Prompt identity

```text
Raw user tokens: 966,381
API prompt tokens: 966,393
Prompt SHA256: 5aa96845a2afc240e47f872b938ef37df9101bb75e41743199597f9bc397adae
```

The exact prompt was saved on the host at:

```text
/home/aabduh/glm53-kv-adversarial-temporal/adversarial-temporal-context-1000000.txt
```

The result JSONL was saved at:

```text
/home/aabduh/glm53-kv-adversarial-temporal/results-fp8.jsonl
```

## Interpretation

This is the strongest synthetic FP8 long-context result in this validation set.

At 966,393 API prompt tokens, the model correctly handled:

```text
long-range retrieval
+ graph traversal
+ superseding revisions
+ obsolete/conflicting records
+ adversarial decoys
+ policy precedence
+ operation-order rules
+ arithmetic
```

and returned all 10 exact final answers after 15,760 output tokens.

## Combined FP8 validation summary

| Test | Largest API prompt | Result |
|---|---:|---:|
| Distributed exact retrieval | 952,751 | 10/10 at largest context |
| Distributed multi-hop reasoning | 963,574 | 10/10 at largest context |
| Adversarial temporal/revision reasoning | 966,393 | **10/10** |

Across all recorded successful checks in these three benchmark families:

```text
Simple retrieval checks:     50/50
Multi-hop reasoning checks:   30/30
Adversarial temporal checks:  10/10

Total successful checks:      90/90
```

These results provide strong synthetic evidence that the current FP8 KV-cache configuration preserves useful long-context information and reasoning behavior very close to the configured 1,048,576-token context limit.

---

# Test 4 — Real Linux Kernel Documentation Exam

## Corpus

A deterministic real-world corpus was built from the upstream Linux kernel documentation tree.

```text
Kernel commit:
654ae5d73c05bd2943d65636ce6cd0aa46e62f18

Corpus tokens:
950,000

Files:
395

Corpus SHA256:
46adbc3b275f91bab6f0627cf0c055e50df2979e0577218c3315f0423b6f46e2
```

The corpus was assembled from Linux documentation subtrees including memory management and related technical documentation, preserving file boundaries and source paths.

## Exam setup

The harness attempted a 25-question bank and automatically excluded questions whose source files were not present in the generated 950K corpus.

```text
Question-bank items available: 23

Unavailable source files:
Q16: Documentation/mm/pagemap.rst
Q19: Documentation/mm/shmem.rst
```

The model therefore received 23 technical questions covering areas such as:

- `mm` versus `active_mm`
- allocation profiling
- architecture page-table helper semantics
- memory balancing
- DAMON design and DAMOS
- high memory mappings
- KSM
- OOM handling
- page migration
- page_owner
- page-table checking
- physical-memory representation
- process address-space structures and locking
- split page-table locking
- swap/reclaim
- Transparent Huge Pages
- unevictable LRU
- vmapped kernel stacks
- vmalloc

The prompt required the model to answer only from the supplied corpus and cite the exact `Documentation/...` source path for every answer.

## Result

| Metric | Result |
|---|---:|
| Corpus tokens | 950,000 |
| Local raw prompt tokens | 950,450 |
| API prompt tokens | 950,462 |
| Questions requested | 23 |
| Questions answered | **23/23** |
| Questions with valid in-corpus source path | **23/23** |
| Output tokens | 18,123 |
| Reasoning characters | 65,909 |
| Elapsed | 354.49 s |
| Finish reason | **stop** |

### Structural/source-grounding score

```text
Questions answered:        23/23
Valid source-path answers: 23/23
Finish reason:             stop
```

The model completed the response naturally rather than exhausting the 32,768-token output allowance.

## Notable grounding behavior

The response did not always force a complete answer when the available corpus was weak. In several cases it explicitly limited its claim to what the supplied documentation supported.

Examples included:

- **Q12 — OOM:** it noted that `Documentation/mm/oom.rst` was a stub and supplemented only with what `Documentation/mm/page_tables.rst` actually supported.
- **Q21 — swap/reclaim:** it stated that the corpus contained limited information and cited `Documentation/mm/swap-table.rst` and `Documentation/mm/multigen_lru.rst` rather than inventing a comprehensive reclaim description.
- **Q25 — vmalloc:** it recognized that `Documentation/mm/vmalloc.rst` was a stub and cautiously inferred only what `Documentation/mm/vmalloced-kernel-stacks.rst` supported.

This is useful evidence that the model was not merely hallucinating complete answers when the real 950K-token corpus lacked enough material.

## Interpretation

This test is the closest of the validation set to an actual long-context technical workflow.

Unlike the synthetic tests, it required the model to operate over hundreds of real upstream documents, locate relevant material among 395 files, synthesize technical explanations, and return source attribution for each answer.

The current result establishes:

```text
23/23 questions answered
23/23 cited at least one valid path from the supplied corpus
natural completion at 18,123 output tokens
```

It does **not yet establish 23/23 semantic correctness**. The next step for this benchmark is to grade each answer against the exact cited documentation so that the structural/source-grounding result can be converted into a true technical accuracy score.

## Selected exam output

<details>
<summary><strong>Expand selected answers</strong></summary>

### Q01 — mm versus active_mm

The model explained that tasks with a normal userspace address space have `tsk->mm` and `tsk->active_mm` referring to the real address space, while kernel threads can have `tsk->mm == NULL` but borrow an `active_mm` while scheduled.

Source cited:

```text
Documentation/mm/active_mm.rst
```

### Q05 — DAMON design

The model described DAMON's region-based sampling, adjustable monitoring intervals, adaptive splitting/merging of regions, and its design goal of controlling monitoring overhead while preserving useful access-pattern information.

Source cited:

```text
Documentation/mm/damon/design.rst
```

### Q18 — process virtual address space

The model identified `struct mm_struct`, VMAs represented by `struct vm_area_struct`, the maple tree, `mmap_lock`, per-VMA locking, reverse-mapping locks, and split page-table locks.

Source cited:

```text
Documentation/mm/process_addrs.rst
```

### Q23 — unevictable LRU

The model explained why unevictable folios are separated from normal reclaim scanning and listed examples such as ramfs, tmpfs with `noswap`, `SHM_LOCK`, and `VM_LOCKED` pages.

Source cited:

```text
Documentation/mm/unevictable-lru.rst
```

</details>
