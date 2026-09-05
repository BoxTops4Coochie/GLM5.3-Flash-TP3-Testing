# GLM-5.3 Flash TP3 — FP8 Long-Context KV Cache Validation

## Executive Summary

**Docker image used for testing - https://github.com/infernix/rtx6kpro/blob/research/glm53-tp3-final-benchmark/models/glm-5.3-flash.md#r17-physical-tp3-compatibility-child**

**Weight used - https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4**

**DFlash Weight Used - https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2**

**Runtime Details - https://github.com/BoxTops4Coochie/GLM5.3-Flash-TP3-Testing/blob/main/GLM-5.3-Flash-TP3-DFlash2-runtime-record.md**

**Test 1 - Distributed Exact Retrieval:** Ten unique records were distributed throughout contexts ranging from roughly 122K to 953K API prompt tokens. FP8 KV returned every requested code exactly, finishing **50/50 overall** and **10/10 at 952,751 tokens**.

**Test 2 - Distributed Multi-Hop Reasoning:** Ten chains per context required linking customer → account → service tier → multiplier plus a separate base allocation, followed by arithmetic. FP8 KV finished **30/30 overall**, including **10/10 at 963,574 API prompt tokens**.

**Test 3 - Adversarial Temporal / Revision Reasoning:** At 966,393 API prompt tokens, each answer required resolving superseded records, rejecting obsolete/draft decoys, traversing a longer graph, applying the active calculation-order policy, and performing arithmetic. The successful run completed naturally and scored **10/10**, using **15,760 output tokens**.

**Test 4 - Real Linux Kernel Documentation Exam:** A deterministic **950,000-token** corpus built from **395 upstream Linux documentation files** was followed by 23 technical questions. GLM answered **23/23**, cited valid in-corpus source paths for **23/23**, and completed naturally with `finish_reason=stop`; semantic correctness can be graded separately against the cited source text.


**Test 5 - Source-Code Call-Graph / Runtime Trace Exam:** A deterministic **950,000-token** corpus built from **115 production source/launcher files** extracted from the exact immutable R17 TP3 image was queried with 25 cross-file implementation questions. Using the final batched procedure, GLM completed **25/25**, every answer cited at least one valid in-corpus source path, **21/25** answers had every cited path present in the corpus manifest, and every authoritative run finished naturally with `finish_reason=stop`. The result measures long-context source retrieval and structural/source grounding; semantic correctness remains separately gradeable against the cited code.

## Combined Results

| Test | Largest API prompt | Result |
|---|---:|---:|
| Test 1 - Distributed Exact Retrieval | 952,751 | 10/10 at largest context |
| Test 2 - Distributed Multi-Hop Reasoning | 963,574 | 10/10 at largest context |
| Test 3 - Adversarial Temporal / Revision Reasoning | 966,393 | 10/10 |
| Test 4 - Real Linux Kernel Documentation Exam | 950,462 | 23/23 answered, 23/23 valid source paths |
| Test 5 - Source-Code Call-Graph / Runtime Trace Exam | 950,362* | 25/25 answered, 25/25 with ≥1 valid source path |

\* Largest API prompt among the authoritative completed Test 5 runs. Batch 9/Q10 used a larger 90,000-token completion allowance but an API prompt of 950,237 tokens.

Synthetic exact checks:

```text
Distributed retrieval:      50/50
Multi-hop reasoning:        30/30
Adversarial temporal:       10/10

Synthetic total:            90/90
```

The Linux documentation and source-code exams are listed separately from the synthetic exact-match tests because their scores primarily reflect answer completion and source-grounding structure rather than a completed source-by-source semantic grade. For Test 5, **25/25** answers cited at least one manifest-valid source; **21/25** used only manifest-valid paths. The remaining four contained one or more extra path citations not present in the 115-file corpus manifest, while still citing valid in-corpus evidence.

---

# Test 1 - Distributed Exact Retrieval

## Purpose

This test validated the current **FP8 KV cache** configuration across progressively larger contexts, from roughly 128K to nearly 1M prompt tokens.

Each context contained ten deterministic authorization-code records spread across the sequence. The model was asked to recover all ten records in a single response, and every value was scored by exact match.

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
10. Saves the exact generated prompts so another KV-cache configuration can later be tested against byte-identical inputs.

Saved host paths:

```text
/home/aabduh/glm53-kv-ab/results-fp8.jsonl
/home/aabduh/glm53-kv-ab/context-131072.txt
/home/aabduh/glm53-kv-ab/context-262144.txt
/home/aabduh/glm53-kv-ab/context-524288.txt
/home/aabduh/glm53-kv-ab/context-786432.txt
/home/aabduh/glm53-kv-ab/context-1000000.txt
```

## Results

| Target context | API prompt tokens | Score | Elapsed |
|---:|---:|---:|---:|
| 131,072 | 121,602 | 10/10 | 13.82 s |
| 262,144 | 246,977 | 10/10 | 27.84 s |
| 524,288 | 497,723 | 10/10 | 60.47 s |
| 786,432 | 748,468 | 10/10 | 103.51 s |
| 1,000,000 | 952,751 | 10/10 | 259.08 s |

```text
Overall: 50/50
Pass rate: 100%
Largest API prompt tested: 952,751 tokens
```


<details>
<summary><strong>Full Test 1 results output</strong></summary>

```text
==============================================================================
GLM-5.3 LONG-CONTEXT KV TEST — FP8
==============================================================================

131,072 target | 121,602 API prompt | 10/10 | 13.82s
262,144 target | 246,977 API prompt | 10/10 | 27.84s
524,288 target | 497,723 API prompt | 10/10 | 60.47s
786,432 target | 748,468 API prompt | 10/10 | 103.51s
1,000,000 target | 952,751 API prompt | 10/10 | 259.08s

OVERALL: 50/50
PASS RATE: 100%
Largest API prompt tested: 952,751 tokens

Results saved to:
/home/aabduh/glm53-kv-ab/results-fp8.jsonl

Prompts saved under:
/home/aabduh/glm53-kv-ab/
```

</details>

<details>
<summary><strong>Full Test 1 script — glm53-longctx-kv-test.py</strong></summary>

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

TARGET_CONTEXTS = [131_072, 262_144, 524_288, 786_432, 1_000_000]
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

    question_lines = [f"{n['id']}: ?" for n in needles]

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

    user_text, needles, raw_tokens = build_context(target, test_index)

    sha = hashlib.sha256(
        user_text.encode("utf-8")
    ).hexdigest()

    prompt_file = OUTDIR / f"context-{target}.txt"

    if not prompt_file.exists():
        prompt_file.write_text(user_text, encoding="utf-8")
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
                {"role": "user", "content": user_text}
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

    answer_text = message.get("content") or ""
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
        (target, usage.get("prompt_tokens"), passed, elapsed)
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

---

# Test 2 - Distributed Multi-Hop Reasoning

## Purpose

This test made the task substantially harder than direct retrieval. For each of ten independent chains, the model had to recover a customer, account, service tier, multiplier, and base allocation from widely separated positions, then compute the final allocation.

The corpus also contained hundreds of decoy records so the answer could not be obtained by simply copying a nearby value.

## Results

| Target context | API prompt tokens | Score | Elapsed | Output tokens | Reasoning chars |
|---:|---:|---:|---:|---:|---:|
| 524,288 | 501,844 | 10/10 | 64.50 s | 1,860 | 4,796 |
| 786,432 | 756,280 | 10/10 | 109.92 s | 2,410 | 6,665 |
| 1,000,000 | 963,574 | 10/10 | 270.44 s | 2,592 | 7,260 |

```text
Overall: 30/30
Pass rate: 100%
Largest API prompt tested: 963,574 tokens
```


<details>
<summary><strong>Full Test 2 results output</strong></summary>

```text
================================================================================
GLM-5.3 LONG-CONTEXT MULTI-HOP TEST — FP8
================================================================================

TARGET CONTEXT: 524,288
API prompt tokens: 501,844
RESULT:            10/10
Elapsed:           64.50 sec
Output tokens:     1,860
Reasoning chars:   4,796

TARGET CONTEXT: 786,432
API prompt tokens: 756,280
RESULT:            10/10
Elapsed:           109.92 sec
Output tokens:     2,410
Reasoning chars:   6,665

TARGET CONTEXT: 1,000,000
API prompt tokens: 963,574
RESULT:            10/10
Elapsed:           270.44 sec
Output tokens:     2,592
Reasoning chars:   7,260

================================================================================
SUMMARY — FP8
================================================================================
524,288 target | 501,844 API prompt | 10/10 | 64.50s
786,432 target | 756,280 API prompt | 10/10 | 109.92s
1,000,000 target | 963,574 API prompt | 10/10 | 270.44s

OVERALL: 30/30

Results saved to:
/home/aabduh/glm53-kv-multihop/results-fp8.jsonl

Prompts saved under:
/home/aabduh/glm53-kv-multihop/
```

</details>

<details>
<summary><strong>Full Test 2 script — glm53-longctx-multihop-test.py</strong></summary>

```python
#!/usr/bin/env python3

import hashlib
import json
import os
import random
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


URL = "http://127.0.0.1:15015/v1/chat/completions"
MODEL = "GLM-5.3-Flash-TP3"
KV_LABEL = os.environ.get("KV_LABEL", "FP8")
MODEL_REPO = "local-inference-lab/GLM-5.3-Flash-NVFP4"

TARGET_CONTEXTS = [524_288, 786_432, 1_000_000]
CHAINS_PER_CONTEXT = 10
CONTENT_RESERVE = 8192
MAX_OUTPUT_TOKENS = 4096
SEED = 530054

OUTDIR = Path.home() / "glm53-kv-multihop"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESULT_FILE = OUTDIR / f"results-{KV_LABEL.lower()}.jsonl"

print(f"Loading tokenizer for {MODEL_REPO} ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    local_files_only=True,
    trust_remote_code=True,
)
print("Tokenizer loaded.")

CUSTOMER_NAMES = [
    "ORCHID", "FALCON", "JASPER", "MERCURY", "CEDAR",
    "COBALT", "LANTERN", "RAVEN", "SUMMIT", "VEGA",
    "HARBOR", "QUARTZ", "EMBER", "ATLAS", "NOVA",
    "MAPLE", "PHOENIX", "SABLE", "TUNDRA", "ZEPHYR",
]

MULTIPLIERS = [
    1.05, 1.10, 1.15, 1.20, 1.25,
    1.30, 1.35, 1.40, 1.45, 1.50,
]


def format_multiplier(x):
    return f"{x:.2f}"


def make_dataset(test_index):
    rng = random.Random(SEED + test_index)
    customers = rng.sample(CUSTOMER_NAMES, CHAINS_PER_CONTEXT)
    chains = []
    used_accounts = set()
    used_tiers = set()

    for idx, customer in enumerate(customers, 1):
        while True:
            account = f"A{rng.randint(1000, 9999)}"
            if account not in used_accounts:
                used_accounts.add(account)
                break

        while True:
            tier = f"TIER-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(10,99)}"
            if tier not in used_tiers:
                used_tiers.add(tier)
                break

        multiplier = rng.choice(MULTIPLIERS)
        base = rng.randrange(1000, 9001, 100)
        expected = round(base * multiplier)

        chains.append({
            "id": f"CHAIN_{idx:02d}",
            "customer": customer,
            "account": account,
            "tier": tier,
            "multiplier": multiplier,
            "base": base,
            "expected": expected,
        })

    return chains


def build_fact_blocks(chains, rng):
    blocks = []

    for c in chains:
        blocks.append({
            "kind": "real",
            "chain": c["id"],
            "stage": 1,
            "text": (
                f"CLIENT DIRECTORY ENTRY:\n"
                f"Customer {c['customer']} uses account identifier "
                f"{c['account']}.\n"
            ),
        })

        blocks.append({
            "kind": "real",
            "chain": c["id"],
            "stage": 2,
            "text": (
                f"ACCOUNT CLASSIFICATION ENTRY:\n"
                f"Account {c['account']} is assigned to service tier "
                f"{c['tier']}.\n"
            ),
        })

        blocks.append({
            "kind": "real",
            "chain": c["id"],
            "stage": 3,
            "text": (
                f"SERVICE-TIER POLICY ENTRY:\n"
                f"Service tier {c['tier']} uses allocation multiplier "
                f"{format_multiplier(c['multiplier'])}.\n"
            ),
        })

        blocks.append({
            "kind": "real",
            "chain": c["id"],
            "stage": 4,
            "text": (
                f"BASE ALLOCATION ENTRY:\n"
                f"Customer {c['customer']} has base allocation "
                f"{c['base']} units.\n"
            ),
        })

    decoys = []

    for i in range(200):
        fake_customer = f"DECOY_CUSTOMER_{i:04d}"
        fake_account = f"D{rng.randint(10000,99999)}"
        fake_tier = f"DECOY-TIER-{rng.randint(100,999)}"
        fake_mult = rng.choice(MULTIPLIERS)
        fake_base = rng.randrange(1000, 9001, 100)

        decoys.extend([
            (
                "CLIENT DIRECTORY ENTRY:\n"
                f"Customer {fake_customer} uses account identifier "
                f"{fake_account}.\n"
            ),
            (
                "ACCOUNT CLASSIFICATION ENTRY:\n"
                f"Account {fake_account} is assigned to service tier "
                f"{fake_tier}.\n"
            ),
            (
                "SERVICE-TIER POLICY ENTRY:\n"
                f"Service tier {fake_tier} uses allocation multiplier "
                f"{format_multiplier(fake_mult)}.\n"
            ),
            (
                "BASE ALLOCATION ENTRY:\n"
                f"Customer {fake_customer} has base allocation "
                f"{fake_base} units.\n"
            ),
        ])

    rng.shuffle(decoys)
    return blocks, decoys


def build_context(target_tokens, test_index):
    rng = random.Random(SEED + test_index)
    chains = make_dataset(test_index)
    real_blocks, decoys = build_fact_blocks(chains, rng)

    REGION_COUNT = 40
    regions = [[] for _ in range(REGION_COUNT)]

    stage_region_bases = {
        1: 1,
        2: 11,
        3: 21,
        4: 31,
    }

    for block in real_blocks:
        chain_num = int(block["chain"].split("_")[1]) - 1
        region_idx = stage_region_bases[block["stage"]] + chain_num
        region_idx = min(region_idx, REGION_COUNT - 1)
        regions[region_idx].append(block["text"])

    for i, decoy in enumerate(decoys):
        regions[i % REGION_COUNT].append(decoy)

    for region in regions:
        rng.shuffle(region)

    true_fact_text = "\n".join(
        x
        for region in regions
        for x in region
    )

    true_fact_tokens = len(
        tokenizer.encode(
            true_fact_text,
            add_special_tokens=False,
        )
    )

    query_lines = [
        f'{c["id"]}: customer={c["customer"]}'
        for c in chains
    ]

    query = (
        "\n\nEND OF ARCHIVE.\n\n"
        "For each requested chain below, determine the customer's final "
        "allocation.\n\n"
        "You must follow the relationships in the archive:\n"
        "customer -> account identifier -> service tier -> multiplier,\n"
        "then combine that multiplier with the customer's base allocation.\n\n"
        "Final allocation = base allocation * service-tier multiplier.\n\n"
        "Return ONLY a JSON object with integer final allocations, using "
        "exactly these keys:\n"
        '{"CHAIN_01":1234,"CHAIN_02":5678,...}\n\n'
        "Do not explain your reasoning. Do not include units.\n\n"
        "Requested chains:\n"
        + "\n".join(query_lines)
    )

    query_tokens = len(
        tokenizer.encode(
            query,
            add_special_tokens=False,
        )
    )

    desired_content_tokens = (
        target_tokens
        - CONTENT_RESERVE
        - query_tokens
    )

    filler_needed = desired_content_tokens - true_fact_tokens

    if filler_needed <= 0:
        raise RuntimeError(
            f"Insufficient token budget for target {target_tokens}"
        )

    filler_phrase = (
        "Routine archival telemetry indicates nominal system operation. "
        "This paragraph contains no customer assignment, account mapping, "
        "service-tier policy, multiplier, or base allocation relevant to "
        "the requested calculation. "
    )

    filler_phrase_ids = tokenizer.encode(
        filler_phrase,
        add_special_tokens=False,
    )

    repeats = (filler_needed // len(filler_phrase_ids)) + 2
    filler_ids = (filler_phrase_ids * repeats)[:filler_needed]

    filler_regions = []

    for i in range(REGION_COUNT):
        start = round(i * len(filler_ids) / REGION_COUNT)
        end = round((i + 1) * len(filler_ids) / REGION_COUNT)

        filler_regions.append(
            tokenizer.decode(
                filler_ids[start:end],
                skip_special_tokens=True,
            )
        )

    parts = []

    for i in range(REGION_COUNT):
        parts.append(
            f"\n\n===== ARCHIVE REGION {i+1:02d} =====\n\n"
        )
        parts.append(filler_regions[i])
        parts.append("\n\n")
        parts.extend(regions[i])

    context = "".join(parts)
    full_user_text = context + query

    raw_tokens = len(
        tokenizer.encode(
            full_user_text,
            add_special_tokens=False,
        )
    )

    return full_user_text, chains, raw_tokens


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
print("=" * 80)
print(f"GLM-5.3 LONG-CONTEXT MULTI-HOP TEST — {KV_LABEL}")
print("=" * 80)

for test_index, target in enumerate(TARGET_CONTEXTS):
    print()
    print("=" * 80)
    print(f"TARGET CONTEXT: {target:,}")
    print("=" * 80)

    user_text, chains, raw_tokens = build_context(
        target,
        test_index,
    )

    sha = hashlib.sha256(
        user_text.encode("utf-8")
    ).hexdigest()

    prompt_file = OUTDIR / f"multihop-context-{target}.txt"

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
    print()

    print("Expected answers:")

    for c in chains:
        print(
            f"  {c['id']} | "
            f"{c['customer']} -> "
            f"{c['account']} -> "
            f"{c['tier']} -> "
            f"{format_multiplier(c['multiplier'])} | "
            f"base={c['base']} | "
            f"expected={c['expected']}"
        )

    print()
    print("Sending request ...")

    start = time.perf_counter()

    r = requests.post(
        URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": user_text}
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
            "total": CHAINS_PER_CONTEXT,
            "error": r.text[:4000],
        }

        with RESULT_FILE.open("a") as f:
            f.write(json.dumps(record) + "\n")

        continue

    obj = r.json()
    message = obj["choices"][0]["message"]

    answer_text = message.get("content") or ""
    reasoning_text = (
        message.get("reasoning")
        or message.get("reasoning_content")
        or ""
    )

    usage = obj.get("usage", {})
    parsed = parse_answer(answer_text)

    passed = 0
    detail = []

    print()
    print("Results:")

    for c in chains:
        actual = None
        ok = False

        if isinstance(parsed, dict):
            actual = parsed.get(c["id"])

            try:
                actual_int = int(actual)
            except Exception:
                actual_int = None

            ok = actual_int == c["expected"]

        if ok:
            passed += 1

        detail.append({
            "id": c["id"],
            "customer": c["customer"],
            "account": c["account"],
            "tier": c["tier"],
            "multiplier": c["multiplier"],
            "base": c["base"],
            "expected": c["expected"],
            "actual": actual,
            "pass": ok,
        })

        print(
            f"{c['id']}: "
            f"{'PASS' if ok else 'FAIL'} | "
            f"expected={c['expected']} | "
            f"actual={actual}"
        )

    print()
    print(f"RESULT:            {passed}/{CHAINS_PER_CONTEXT}")
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
        "total": CHAINS_PER_CONTEXT,
        "details": detail,
        "answer": answer_text,
    }

    with RESULT_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    summary.append({
        "target": target,
        "api_tokens": usage.get("prompt_tokens"),
        "passed": passed,
        "elapsed": elapsed,
    })


print()
print("=" * 80)
print(f"SUMMARY — {KV_LABEL}")
print("=" * 80)

total_passed = 0
total_tests = 0

for row in summary:
    total_passed += row["passed"]
    total_tests += CHAINS_PER_CONTEXT

    print(
        f"{row['target']:>9,} target | "
        f"{row['api_tokens'] or 0:>9,} API prompt | "
        f"{row['passed']:>2}/{CHAINS_PER_CONTEXT} | "
        f"{row['elapsed']:>8.2f}s"
    )

print()
print(f"OVERALL: {total_passed}/{total_tests}")
print()
print(f"Results saved to: {RESULT_FILE}")
print(f"Prompts saved to: {OUTDIR}")
```

</details>

---

# Test 3 - Adversarial Temporal / Revision Reasoning

## Purpose

The third benchmark increased difficulty beyond both direct retrieval and ordinary multi-hop reasoning.

Each answer required the model to resolve a chain of active versus superseded records, reject plausible obsolete/draft decoys, recover an operation-order rule near the end of the context, and compute a final value.

## Result

| Metric | Result |
|---|---:|
| Raw user tokens | 966,381 |
| API prompt tokens | 966,393 |
| Output tokens | 15,760 |
| Reasoning characters | 52,328 |
| Elapsed | 60.97 s |
| Exact answers | **10/10** |

```text
Pass rate: 100%
Prompt SHA256:
5aa96845a2afc240e47f872b938ef37df9101bb75e41743199597f9bc397adae
```


<details>
<summary><strong>Full Test 3 results output</strong></summary>

```text
========================================================================================
GLM-5.3 ADVERSARIAL TEMPORAL LONG-CONTEXT TEST — FP8
========================================================================================
Raw user tokens: 966,381
Prompt SHA256:    5aa96845a2afc240e47f872b938ef37df9101bb75e41743199597f9bc397adae

Expected chains:
  CHAIN_01 | COBALT -> C9881 -> A6903 -> SOUTH -> G29 -> 1.15 | base=8600 | adj=+50 | rule=MULTIPLY_THEN_ADJUST | (8600 * 1.15) + 50 -> 9940
  CHAIN_02 | FALCON -> C5925 -> A4427 -> ATLANTIC -> G93 -> 1.25 | base=7100 | adj=+75 | rule=MULTIPLY_THEN_ADJUST | (7100 * 1.25) + 75 -> 8950
  CHAIN_03 | VEGA -> C4055 -> A4905 -> ATLANTIC -> G22 -> 1.10 | base=3800 | adj=+450 | rule=ADJUST_THEN_MULTIPLY | (3800 + 450) * 1.10 -> 4675
  CHAIN_04 | SABLE -> C9974 -> A9533 -> WEST -> G45 -> 1.35 | base=2000 | adj=+300 | rule=ADJUST_THEN_MULTIPLY | (2000 + 300) * 1.35 -> 3105
  CHAIN_05 | LANTERN -> C1082 -> A2655 -> SOUTH -> G82 -> 1.15 | base=3400 | adj=+150 | rule=MULTIPLY_THEN_ADJUST | (3400 * 1.15) + 150 -> 4060
  CHAIN_06 | ZEPHYR -> C8610 -> A1413 -> SOUTH -> G52 -> 1.25 | base=3100 | adj=+225 | rule=ADJUST_THEN_MULTIPLY | (3100 + 225) * 1.25 -> 4156
  CHAIN_07 | ORCHID -> C3397 -> A6027 -> NORTH -> G80 -> 1.45 | base=4300 | adj=+225 | rule=ADJUST_THEN_MULTIPLY | (4300 + 225) * 1.45 -> 6561
  CHAIN_08 | EMBER -> C8911 -> A3101 -> CENTRAL -> G36 -> 1.45 | base=3700 | adj=+100 | rule=MULTIPLY_THEN_ADJUST | (3700 * 1.45) + 100 -> 5465
  CHAIN_09 | ATLAS -> C6032 -> A9258 -> NORTH -> G59 -> 1.05 | base=8900 | adj=+400 | rule=MULTIPLY_THEN_ADJUST | (8900 * 1.05) + 400 -> 9745
  CHAIN_10 | PHOENIX -> C9593 -> A3845 -> PACIFIC -> G56 -> 1.20 | base=8300 | adj=+425 | rule=MULTIPLY_THEN_ADJUST | (8300 * 1.20) + 425 -> 10385

Sending ~1M-token adversarial request ...

Results:
CHAIN_01: PASS | expected=9940 | actual=9940
CHAIN_02: PASS | expected=8950 | actual=8950
CHAIN_03: PASS | expected=4675 | actual=4675
CHAIN_04: PASS | expected=3105 | actual=3105
CHAIN_05: PASS | expected=4060 | actual=4060
CHAIN_06: PASS | expected=4156 | actual=4156
CHAIN_07: PASS | expected=6561 | actual=6561
CHAIN_08: PASS | expected=5465 | actual=5465
CHAIN_09: PASS | expected=9745 | actual=9745
CHAIN_10: PASS | expected=10385 | actual=10385

========================================================================================
SUMMARY
========================================================================================
RESULT:            10/10
Elapsed:           60.97 sec
Prompt tokens API: 966393
Output tokens:     15760
Reasoning chars:   52328
Raw user tokens:   966381

Results saved to: /home/aabduh/glm53-kv-adversarial-temporal/results-fp8.jsonl
Prompt saved to:  /home/aabduh/glm53-kv-adversarial-temporal/adversarial-temporal-context-1000000.txt
```

</details>

<details>
<summary><strong>Full Test 3 script — glm53-longctx-adversarial-temporal-test.py</strong></summary>

```python
#!/usr/bin/env python3

import hashlib
import json
import os
import random
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


URL = "http://127.0.0.1:15015/v1/chat/completions"
MODEL = "GLM-5.3-Flash-TP3"
KV_LABEL = os.environ.get("KV_LABEL", "FP8")
MODEL_REPO = "local-inference-lab/GLM-5.3-Flash-NVFP4"

TARGET_CONTEXT = 1_000_000
CONTENT_RESERVE = 12_000
CHAINS = 10
MAX_OUTPUT_TOKENS = 32768
SEED = 530055

OUTDIR = Path.home() / "glm53-kv-adversarial-temporal"
OUTDIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = OUTDIR / f"results-{KV_LABEL.lower()}.jsonl"
PROMPT_FILE = OUTDIR / "adversarial-temporal-context-1000000.txt"


print(f"Loading tokenizer for {MODEL_REPO} ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    local_files_only=True,
    trust_remote_code=True,
)
print("Tokenizer loaded.")


PROJECT_NAMES = [
    "ORCHID", "FALCON", "JASPER", "MERCURY", "CEDAR",
    "COBALT", "LANTERN", "RAVEN", "SUMMIT", "VEGA",
    "HARBOR", "QUARTZ", "EMBER", "ATLAS", "NOVA",
    "MAPLE", "PHOENIX", "SABLE", "TUNDRA", "ZEPHYR",
]

REGIONS = [
    "EAST", "WEST", "NORTH", "SOUTH",
    "CENTRAL", "PACIFIC", "ATLANTIC", "MOUNTAIN",
]

MULTIPLIERS = [
    1.05, 1.10, 1.15, 1.20, 1.25,
    1.30, 1.35, 1.40, 1.45, 1.50,
]


def fmt_mult(x):
    return f"{x:.2f}"


def calc_expected(base, adjustment, multiplier, rule):
    if rule == "ADJUST_THEN_MULTIPLY":
        return round((base + adjustment) * multiplier)

    if rule == "MULTIPLY_THEN_ADJUST":
        return round(base * multiplier + adjustment)

    raise ValueError(rule)


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
        return json.loads(text[start:end + 1])
    except Exception:
        return None


rng = random.Random(SEED)
projects = rng.sample(PROJECT_NAMES, CHAINS)

chains = []
used_customers = set()
used_accounts = set()
used_groups = set()

for i, project in enumerate(projects, 1):
    while True:
        customer = f"C{rng.randint(1000, 9999)}"
        if customer not in used_customers:
            used_customers.add(customer)
            break

    while True:
        account = f"A{rng.randint(1000, 9999)}"
        if account not in used_accounts:
            used_accounts.add(account)
            break

    while True:
        service_group = f"G{rng.randint(10, 99)}"
        if service_group not in used_groups:
            used_groups.add(service_group)
            break

    old_region, new_region = rng.sample(REGIONS, 2)
    old_multiplier, new_multiplier = rng.sample(MULTIPLIERS, 2)

    base = rng.randrange(1200, 9001, 100)
    adjustment = rng.randrange(25, 501, 25)

    rule = rng.choice([
        "ADJUST_THEN_MULTIPLY",
        "MULTIPLY_THEN_ADJUST",
    ])

    region_revision = rng.randint(20, 80)
    multiplier_revision = rng.randint(100, 180)
    rule_revision = rng.randint(200, 260)

    expected = calc_expected(
        base,
        adjustment,
        new_multiplier,
        rule,
    )

    chains.append({
        "id": f"CHAIN_{i:02d}",
        "project": project,
        "customer": customer,
        "account": account,
        "old_region": old_region,
        "new_region": new_region,
        "region_revision": region_revision,
        "service_group": service_group,
        "old_multiplier": old_multiplier,
        "new_multiplier": new_multiplier,
        "multiplier_revision": multiplier_revision,
        "base": base,
        "adjustment": adjustment,
        "rule": rule,
        "rule_revision": rule_revision,
        "expected": expected,
    })


real_blocks = []

for c in chains:
    real_blocks.append({
        "stage": 1,
        "chain": c["id"],
        "text": (
            "PROJECT REGISTRY RECORD:\n"
            f"Project {c['project']} is assigned to customer "
            f"{c['customer']}.\n"
        ),
    })

    real_blocks.append({
        "stage": 2,
        "chain": c["id"],
        "text": (
            "CUSTOMER ACCOUNT RECORD:\n"
            f"Customer {c['customer']} uses account "
            f"{c['account']}.\n"
        ),
    })

    real_blocks.append({
        "stage": 3,
        "chain": c["id"],
        "text": (
            "ACCOUNT REGION RECORD — ORIGINAL:\n"
            f"Account {c['account']} maps to region "
            f"{c['old_region']}.\n"
            "This record predates later revisions.\n"
        ),
    })

    real_blocks.append({
        "stage": 4,
        "chain": c["id"],
        "text": (
            f"ACCOUNT REGION REVISION {c['region_revision']:03d}:\n"
            f"Revision {c['region_revision']:03d} supersedes all earlier "
            f"region mappings for account {c['account']}.\n"
            f"Account {c['account']} now maps to region "
            f"{c['new_region']}.\n"
        ),
    })

    real_blocks.append({
        "stage": 5,
        "chain": c["id"],
        "text": (
            "REGION SERVICE GROUP RECORD:\n"
            f"Region {c['new_region']} for project {c['project']} "
            f"maps to service group {c['service_group']}.\n"
        ),
    })

    real_blocks.append({
        "stage": 6,
        "chain": c["id"],
        "text": (
            "SERVICE GROUP POLICY — ORIGINAL:\n"
            f"Service group {c['service_group']} originally used "
            f"multiplier {fmt_mult(c['old_multiplier'])}.\n"
            "This value is obsolete if a later policy revision exists.\n"
        ),
    })

    real_blocks.append({
        "stage": 7,
        "chain": c["id"],
        "text": (
            f"SERVICE GROUP POLICY REVISION "
            f"{c['multiplier_revision']:03d}:\n"
            f"Revision {c['multiplier_revision']:03d} supersedes all "
            f"earlier multipliers for service group "
            f"{c['service_group']}.\n"
            f"The active multiplier is now "
            f"{fmt_mult(c['new_multiplier'])}.\n"
        ),
    })

    real_blocks.append({
        "stage": 8,
        "chain": c["id"],
        "text": (
            "PROJECT BASE ALLOCATION RECORD:\n"
            f"Project {c['project']} has base allocation "
            f"{c['base']} units.\n"
        ),
    })

    real_blocks.append({
        "stage": 9,
        "chain": c["id"],
        "text": (
            "PROJECT ADJUSTMENT RECORD:\n"
            f"Project {c['project']} has an adjustment of "
            f"+{c['adjustment']} units.\n"
        ),
    })

    if c["rule"] == "ADJUST_THEN_MULTIPLY":
        rule_text = (
            "Apply the project adjustment to the base allocation first, "
            "then multiply the adjusted amount by the active service-group "
            "multiplier."
        )
    else:
        rule_text = (
            "Multiply the base allocation by the active service-group "
            "multiplier first, then add the project adjustment."
        )

    real_blocks.append({
        "stage": 10,
        "chain": c["id"],
        "text": (
            f"CALCULATION POLICY REVISION {c['rule_revision']:03d}:\n"
            f"For project {c['project']}, revision "
            f"{c['rule_revision']:03d} defines the active calculation rule.\n"
            f"{rule_text}\n"
        ),
    })


decoys = []

for i in range(400):
    fake_project = f"DECOY_PROJECT_{i:04d}"
    fake_customer = f"DC{rng.randint(10000, 99999)}"
    fake_account = f"DA{rng.randint(10000, 99999)}"
    fake_region = rng.choice(REGIONS)
    fake_group = f"DG{rng.randint(100, 999)}"
    fake_mult = rng.choice(MULTIPLIERS)
    fake_base = rng.randrange(1000, 9001, 100)
    fake_adjust = rng.randrange(25, 501, 25)

    decoys.extend([
        (
            "PROJECT REGISTRY RECORD:\n"
            f"Project {fake_project} is assigned to customer "
            f"{fake_customer}.\n"
        ),
        (
            "CUSTOMER ACCOUNT RECORD:\n"
            f"Customer {fake_customer} uses account "
            f"{fake_account}.\n"
        ),
        (
            "ACCOUNT REGION RECORD:\n"
            f"Account {fake_account} maps to region "
            f"{fake_region}.\n"
        ),
        (
            "REGION SERVICE GROUP RECORD:\n"
            f"Region {fake_region} for project {fake_project} "
            f"maps to service group {fake_group}.\n"
        ),
        (
            "SERVICE GROUP POLICY RECORD:\n"
            f"Service group {fake_group} uses multiplier "
            f"{fmt_mult(fake_mult)}.\n"
        ),
        (
            "PROJECT BASE ALLOCATION RECORD:\n"
            f"Project {fake_project} has base allocation "
            f"{fake_base} units.\n"
        ),
        (
            "PROJECT ADJUSTMENT RECORD:\n"
            f"Project {fake_project} has an adjustment of "
            f"+{fake_adjust} units.\n"
        ),
    ])


for c in chains:
    decoys.extend([
        (
            "REJECTED DRAFT — DO NOT USE:\n"
            f"Draft proposal suggested project {c['project']} might use "
            f"customer C{rng.randint(1000,9999)}. "
            "This draft was rejected and never became active.\n"
        ),
        (
            "OBSOLETE ACCOUNT REGION NOTE:\n"
            f"A historical note references account {c['account']} and region "
            f"{c['old_region']}. This note is superseded by the later "
            f"account-region revision.\n"
        ),
        (
            "RETIRED POLICY SNAPSHOT:\n"
            f"An older snapshot lists service group {c['service_group']} "
            f"with multiplier {fmt_mult(c['old_multiplier'])}. "
            "This snapshot is not the active policy.\n"
        ),
        (
            "UNAPPROVED CALCULATION DRAFT:\n"
            f"A discarded draft for project {c['project']} proposed an "
            "alternate calculation order. It was never approved and must "
            "not override the active calculation-policy revision.\n"
        ),
    ])


REGION_COUNT = 100
regions = [[] for _ in range(REGION_COUNT)]

stage_bases = {
    1: 0,
    2: 10,
    3: 20,
    4: 30,
    5: 40,
    6: 50,
    7: 60,
    8: 70,
    9: 80,
    10: 90,
}

for block in real_blocks:
    chain_num = int(
        block["chain"].split("_")[1]
    ) - 1

    region_idx = (
        stage_bases[block["stage"]]
        + chain_num
    )

    regions[region_idx].append(block["text"])


rng.shuffle(decoys)

for i, decoy in enumerate(decoys):
    regions[i % REGION_COUNT].append(decoy)

for region in regions:
    rng.shuffle(region)


query_lines = [
    f"{c['id']}: project={c['project']}"
    for c in chains
]

query = (
    "\n\nEND OF ARCHIVE.\n\n"
    "For each requested project, determine the ACTIVE final allocation.\n\n"
    "You must follow the complete chain:\n"
    "project -> customer -> account -> ACTIVE region -> service group "
    "-> ACTIVE multiplier.\n\n"
    "You must also retrieve the project's base allocation, adjustment, "
    "and ACTIVE calculation-order rule.\n\n"
    "Revision and precedence rules:\n"
    "1. A later explicit revision supersedes an earlier record for the "
    "same relationship or policy.\n"
    "2. Records marked obsolete, retired, rejected, historical, draft, "
    "or unapproved are not active.\n"
    "3. Use only the final active region mapping.\n"
    "4. Use only the final active multiplier.\n"
    "5. Follow the active calculation-order rule exactly.\n\n"
    "Return ONLY a JSON object whose values are integer final allocations.\n"
    "Use exactly these keys:\n"
    '{"CHAIN_01":1234,"CHAIN_02":5678,...}\n\n'
    "Do not explain your reasoning. Do not include units.\n\n"
    "Requested chains:\n"
    + "\n".join(query_lines)
)


all_fact_text = "\n".join(
    text
    for region in regions
    for text in region
)

fact_tokens = len(
    tokenizer.encode(
        all_fact_text,
        add_special_tokens=False,
    )
)

query_tokens = len(
    tokenizer.encode(
        query,
        add_special_tokens=False,
    )
)

desired_content_tokens = (
    TARGET_CONTEXT
    - CONTENT_RESERVE
    - query_tokens
)

filler_needed = desired_content_tokens - fact_tokens

if filler_needed <= 0:
    raise RuntimeError(
        "Fact/decoy data exceeded context budget."
    )


filler_phrase = (
    "Routine archival telemetry indicates nominal system operation. "
    "This paragraph contains no authoritative project assignment, "
    "customer-account mapping, revision, service-group policy, "
    "allocation value, adjustment, or calculation rule relevant to "
    "the requested result. "
)

filler_phrase_ids = tokenizer.encode(
    filler_phrase,
    add_special_tokens=False,
)

repeats = (
    filler_needed // len(filler_phrase_ids)
) + 2

filler_ids = (
    filler_phrase_ids * repeats
)[:filler_needed]


filler_regions = []

for i in range(REGION_COUNT):
    start = round(
        i * len(filler_ids) / REGION_COUNT
    )

    end = round(
        (i + 1) * len(filler_ids) / REGION_COUNT
    )

    filler_regions.append(
        tokenizer.decode(
            filler_ids[start:end],
            skip_special_tokens=True,
        )
    )


parts = []

for i in range(REGION_COUNT):
    parts.append(
        f"\n\n===== ARCHIVE REGION {i+1:03d} =====\n\n"
    )
    parts.append(filler_regions[i])
    parts.append("\n\n")
    parts.extend(regions[i])


context = "".join(parts)
user_text = context + query

raw_tokens = len(
    tokenizer.encode(
        user_text,
        add_special_tokens=False,
    )
)

sha = hashlib.sha256(
    user_text.encode("utf-8")
).hexdigest()


if not PROMPT_FILE.exists():
    PROMPT_FILE.write_text(
        user_text,
        encoding="utf-8",
    )
else:
    existing = PROMPT_FILE.read_text(
        encoding="utf-8"
    )

    existing_sha = hashlib.sha256(
        existing.encode("utf-8")
    ).hexdigest()

    if existing_sha != sha:
        raise RuntimeError(
            "Existing prompt does not match generated prompt."
        )


print()
print("=" * 88)
print(
    f"GLM-5.3 ADVERSARIAL TEMPORAL LONG-CONTEXT TEST — {KV_LABEL}"
)
print("=" * 88)

print(f"Raw user tokens: {raw_tokens:,}")
print(f"Prompt SHA256:    {sha}")

print()
print("Expected chains:")

for c in chains:
    if c["rule"] == "ADJUST_THEN_MULTIPLY":
        equation = (
            f"({c['base']} + {c['adjustment']}) "
            f"* {fmt_mult(c['new_multiplier'])}"
        )
    else:
        equation = (
            f"({c['base']} * "
            f"{fmt_mult(c['new_multiplier'])}) "
            f"+ {c['adjustment']}"
        )

    print(
        f"  {c['id']} | "
        f"{c['project']} -> "
        f"{c['customer']} -> "
        f"{c['account']} -> "
        f"{c['new_region']} -> "
        f"{c['service_group']} -> "
        f"{fmt_mult(c['new_multiplier'])} | "
        f"base={c['base']} | "
        f"adj=+{c['adjustment']} | "
        f"rule={c['rule']} | "
        f"{equation} -> "
        f"{c['expected']}"
    )


print()
print("Sending ~1M-token adversarial request ...")

start = time.perf_counter()

r = requests.post(
    URL,
    json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": user_text}
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
    print()
    print(f"HTTP ERROR: {r.status_code}")
    print(r.text[:5000])

    record = {
        "kv": KV_LABEL,
        "target_context": TARGET_CONTEXT,
        "raw_user_tokens": raw_tokens,
        "prompt_sha256": sha,
        "http_status": r.status_code,
        "elapsed_sec": elapsed,
        "passed": 0,
        "total": CHAINS,
        "error": r.text[:5000],
    }

    with RESULT_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    raise SystemExit(1)


obj = r.json()
message = obj["choices"][0]["message"]

answer_text = message.get("content") or ""
reasoning_text = (
    message.get("reasoning")
    or message.get("reasoning_content")
    or ""
)

usage = obj.get("usage", {})
parsed = parse_answer(answer_text)


passed = 0
details = []

print()
print("Results:")

for c in chains:
    actual = None
    actual_int = None

    if isinstance(parsed, dict):
        actual = parsed.get(c["id"])

        try:
            actual_int = int(actual)
        except Exception:
            actual_int = None

    ok = actual_int == c["expected"]

    if ok:
        passed += 1

    details.append({
        "id": c["id"],
        "project": c["project"],
        "customer": c["customer"],
        "account": c["account"],
        "old_region": c["old_region"],
        "active_region": c["new_region"],
        "service_group": c["service_group"],
        "old_multiplier": c["old_multiplier"],
        "active_multiplier": c["new_multiplier"],
        "base": c["base"],
        "adjustment": c["adjustment"],
        "rule": c["rule"],
        "expected": c["expected"],
        "actual": actual,
        "pass": ok,
    })

    print(
        f"{c['id']}: "
        f"{'PASS' if ok else 'FAIL'} | "
        f"expected={c['expected']} | "
        f"actual={actual}"
    )


print()
print("=" * 88)
print("SUMMARY")
print("=" * 88)

print(f"RESULT:            {passed}/{CHAINS}")
print(f"Elapsed:           {elapsed:.2f} sec")
print(f"Prompt tokens API: {usage.get('prompt_tokens')}")
print(f"Output tokens:     {usage.get('completion_tokens')}")
print(f"Reasoning chars:   {len(reasoning_text)}")
print(f"Raw user tokens:   {raw_tokens}")


record = {
    "kv": KV_LABEL,
    "target_context": TARGET_CONTEXT,
    "raw_user_tokens": raw_tokens,
    "prompt_sha256": sha,
    "http_status": 200,
    "elapsed_sec": elapsed,
    "api_prompt_tokens": usage.get("prompt_tokens"),
    "api_completion_tokens": usage.get("completion_tokens"),
    "passed": passed,
    "total": CHAINS,
    "details": details,
    "answer": answer_text,
}

with RESULT_FILE.open("a") as f:
    f.write(json.dumps(record) + "\n")

print()
print(f"Results saved to: {RESULT_FILE}")
print(f"Prompt saved to:  {PROMPT_FILE}")
```

</details>

---

# Test 4 - Real Linux Kernel Documentation Exam

## Purpose

This test replaced synthetic facts with real upstream technical documentation. A deterministic 950,000-token corpus made from 395 Linux kernel documentation files was followed by technical questions requiring the model to locate, synthesize, and cite information from the supplied corpus.

The harness also required the model to state when the supplied corpus did not contain enough evidence rather than silently filling gaps from outside knowledge.

## Corpus identity

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

## Results

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

Two bank items were excluded automatically because their exact source files were not present in the generated 950K corpus:

```text
Q16: Documentation/mm/pagemap.rst
Q19: Documentation/mm/shmem.rst
```


<details>
<summary><strong>Full Test 4 results output</strong></summary>

```text
Loading corpus ...
Corpus SHA256: 46adbc3b275f91bab6f0627cf0c055e50df2979e0577218c3315f0423b6f46e2
Manifest files: 395

Available question-bank items: 23
Unavailable source files:
  Q16: Documentation/mm/pagemap.rst
  Q19: Documentation/mm/shmem.rst

Running 23 questions:
  Q01: Explain the distinction between mm and active_mm for a Linux task, including why a kernel thread can have no normal mm while still having an active_mm.
  Q02: How does Linux allocation profiling attribute memory allocations, and what is the intended use of the resulting information?
  Q03: What purpose do the generic page-table helper semantics serve for architecture implementations, and what kind of errors are the associated tests intended to detect?
  Q04: Describe the memory-balancing behavior documented here. What conditions cause balancing or reclaim activity and what is it trying to accomplish?
  Q05: Describe DAMON's core design. How does it monitor memory access patterns while controlling monitoring overhead?
  Q06: Within DAMON, explain the relationship between monitoring targets, address spaces or regions, sampling, aggregation, and region adaptation.
  Q07: What are DAMOS schemes and how do they turn DAMON's monitoring information into memory-management actions or policies?
  Q08: Identify one limitation, common misunderstanding, or important operational consideration described in the DAMON FAQ and explain it accurately.
  Q09: Summarize the documented expectations for DAMON development or maintenance, including the workflow or review expectations that a contributor should know.
  Q10: Why does high memory require special handling on affected architectures, and what distinction does the documentation make between permanent and temporary mappings?
  Q11: Explain how Kernel Samepage Merging identifies and merges duplicate anonymous pages, and describe the copy-on-write behavior after pages have been merged.
  Q12: What is the purpose of Linux's out-of-memory handling, and what controls or factors described in the documentation influence which task is selected when memory exhaustion occurs?
  Q13: Explain what page migration is, why Linux performs it, and what must happen to mappings or references while a page is being migrated.
  Q14: What problem is page_owner designed to diagnose, what information does it record, and what cost or tradeoff comes with enabling it?
  Q15: What classes of incorrect page-table mappings is page table check intended to detect, and when are those checks performed?
  Q17: Explain how Linux represents physical memory in terms of nodes, zones, and pages. Describe the relationship among those structures.
  Q18: Describe the major data structures and locking considerations used to represent a process virtual address space.
  Q20: What problem does split page-table locking solve, and at which page-table levels can separate locks be used?
  Q21: Describe the role of swap in Linux memory management and explain how reclaim and swap interact when anonymous memory must be evicted.
  Q22: What are Transparent Huge Pages intended to improve, and what tradeoffs or failure modes make THP policy more complicated than simply using huge pages everywhere?
  Q23: Why does Linux maintain an unevictable LRU, what kinds of pages can end up there, and why is treating those pages separately useful to reclaim?
  Q24: What are the motivations for vmapped kernel stacks and what memory-safety or debugging benefits do they provide?
  Q25: Explain the purpose of vmalloc and contrast virtually contiguous memory obtained through vmalloc with physically contiguous allocations.

Loading tokenizer for local-inference-lab/GLM-5.3-Flash-NVFP4 ...
Ignoring corrupted tree cache file /m2-2/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/trees/46aaae8a82032f77100f2f03e9cc11b391df3b4d.json: [Errno 13] Permission denied: '/m2-2/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/trees/46aaae8a82032f77100f2f03e9cc11b391df3b4d.json'
Local raw prompt tokens: 950,450
Output allowance:        32,768

Sending Linux-documentation exam ...

================================================================================
LINUX DOCUMENTATION 25-QUESTION RESULT
================================================================================
Questions requested:      23
Questions answered:       23
Valid source-path answers:23
Elapsed:                  354.49 sec
API prompt tokens:        950462
Output tokens:            18123
Reasoning chars:          65909
Finish reason:            stop

Parsed result: /home/aabduh/linux-docs-bench/linux-docs-25q-result.json
Raw response:  /home/aabduh/linux-docs-bench/linux-docs-25q-raw-response.txt

================================================================================
MODEL ANSWERS
================================================================================

Q01
The kernel distinguishes 'real' address spaces (tracked by tsk->mm) from 'anonymous' address spaces. For a task with a real address space, tsk->mm is non-NULL and tsk->active_mm always equals tsk->mm. For an anonymous process (e.g., a kernel thread), tsk->mm is NULL because it has no real address space, but tsk->active_mm records the 'borrowed' address space that is currently active; when the task is scheduled away, the borrowed address space is returned and cleared. This avoids wasting time switching VM state for tasks that do not use user mappings. Two counters support this: mm_users counts real users, and mm_count counts lazy users plus one if there are any real users.
Sources: Documentation/mm/active_mm.rst

Q02
Memory allocation profiling provides low-overhead accounting of all memory allocations tracked by file and line number. It replaces allocation calls with an alloc_hooks() macro that declares a code tag, stashes a pointer to it in task_struct, calls the real allocation function, and restores the previous tag pointer, allowing nested hooks. The resulting data is exposed via /proc/allocinfo (and an ioctl interface) showing bytes, counts, and the call site (file:line and function). The intended use is to identify memory consumers and debug allocation behavior in production.
Sources: Documentation/mm/allocation-profiling.rst

Q03
Generic MM expects architectures (with MMU) to provide helpers to create, access, and modify page table entries at various levels, and these helpers must conform to a common semantics across platforms so that generic code works uniformly. The documented tables describe the expected semantics for PTE, PMD, PUD, HugeTLB, and swap helpers. The associated tests (enabled via CONFIG_DEBUG_VM_PGTABLE) run during boot to verify that architecture implementations conform to these semantics, detecting incorrect or missing helper behavior.
Sources: Documentation/mm/arch_pgtable_helpers.rst

Q04
Memory balancing is needed for allocation requests that do not set __GFP_HIGH or __GFP_KSWAPD_RECLAIM, and for non-__GFP_IO allocations. Callers may avoid reclaim because they cannot sleep (holding spinlocks or in interrupt context) or because they are willing to fail an opportunistic high-order allocation rather than incur reclaim overhead. Balancing is driven by per-zone watermarks: when free pages fall below watermark[WMARK_MIN], the hysteric low_on_memory flag is set (until watermark[WMARK_HIGH]) and allocation requests try to free pages; independently, when free pages fall below watermark[WMARK_LOW], kswapd is woken (zone_wake_kswapd). The goal is to keep zones filled and avoid falling back to scarce zones (e.g., DMA), and page stealing from process memory and shm is done if it alleviates memory pressure on any zone below its watermark.
Sources: Documentation/mm/balance.rst

Q05
DAMON is a kernel subsystem for efficient data access monitoring and access-aware system operations. It is configured with three layers: an operations set (address-space dependent primitives), core logic (overhead/accuracy control and access-aware operations), and modules (user interfaces). To control overhead, DAMON uses region-based sampling: it groups adjacent pages assumed to have the same access frequencies into regions and checks only one page per region per sampling interval, making overhead controllable via the number of regions. It adaptively merges and splits regions based on access frequency to maintain the assumption, and tracks an age counter per region. Users set sampling interval, aggregation interval, update interval, and min/max number of regions to trade off accuracy and overhead.
Sources: Documentation/mm/damon/design.rst

Q06
DAMON monitors target address spaces (e.g., virtual address spaces of processes or the physical address space). For virtual address spaces, it constructs three regions covering every mapped area, excluding the two biggest unmapped gaps. Within each region, DAMON performs region-based sampling: for each sampling interval it picks one page per region, checks whether it was accessed, and increments the region's nr_accesses counter. After each aggregation interval it reports and clears the aggregated access frequency. To keep the assumption that pages in a region have similar access frequencies, DAMON adaptively merges adjacent regions when their access frequency difference is small and their combined size is below a threshold, and splits regions when the total number is below the user-specified maximum. An age counter tracks how long the current access pattern has been maintained.
Sources: Documentation/mm/damon/design.rst

Q07
DAMOS (Data Access Monitoring-based Operation Schemes) lets users specify desired memory-management schemes at a high level. DAMON monitors, finds regions matching a user-specified access pattern (size, access frequency, age), and applies a user-desired action to those regions for every apply_interval. Supported actions include willneed, cold, pageout, hugepage, nohugepage, collapse, lru_prio, lru_deprio, migrate_hot, migrate_cold, and stat. To bound overhead, DAMOS provides quotas (limits on time or bytes applied), prioritization mechanisms to choose which regions to act on under quotas, watermarks for conditional activation/deactivation, and filters (e.g., by address, target, anonymous, memcg, young, hugepage size, unmapped) to exclude memory from the scheme.
Sources: Documentation/mm/damon/design.rst

Q08
The FAQ clarifies that DAMON is not limited to virtual memory: its core is address-space independent, and address-space-specific operations (target region construction and access checks) can be implemented and configured by users; DAMON provides default vaddr/paddr implementations for convenience. It also notes that page-granularity monitoring is possible by setting min_nr_regions higher than the working set size divided by the page size, because region sizes are forced to be >= page size, making splitting ineffective.
Sources: Documentation/mm/damon/faq.rst

Q09
DAMON patches should be made against the mm-new tree and posted to the damon@lists.linux.dev and linux-mm@kvack.org mailing lists. Contributors should build the kernel and documents, ensure no new errors or warnings, run DAMON selftests and kunittests, and (for normal changes) run damon-tests/corr; for performance changes, measure impacts on benchmarks or real-world workloads. Patches can be sent anytime; the maintainer works flexibly (except early morning PT) and may be slow to respond, so contributors should ping if they have not heard back within a week.
Sources: Documentation/mm/damon/maintainer-profile.rst

Q10
High memory is needed when physical memory approaches or exceeds the maximum virtual memory size, making it impossible for the kernel to keep all physical memory permanently mapped. The documentation distinguishes permanent mappings (the kernel's direct map, limited by the virtual address space split, e.g., ~896 MiB on i386) from temporary mappings used to access the rest of physical memory. Temporary mappings are created via kmap_local_page()/kmap_local_folio() (preferred, thread-local and CPU-local, valid only in the acquiring context), the deprecated kmap_atomic() and kmap(), and vmap() for long-duration mappings of multiple pages.
Sources: Documentation/mm/highmem.rst

Q11
KSM is a memory-saving de-duplication feature that merges identical anonymous pages. It maintains reverse mapping information in a stable tree: if a KSM page is shared by fewer than max_page_sharing VMAs, the stable tree node points to a list of rmap_items; when sharing exceeds that threshold, the node becomes a chain linking multiple 'dups,' each holding reverse mapping information for a KSM page copy. The max_page_sharing limit prevents rmap lists from growing too large. The merged pages are write-protected (the stable tree enforces that chain/dups represent the same write-protected content), so writes to a merged page require breaking the sharing (copy-on-write), though the document focuses on the tree structure rather than the fault path.
Sources: Documentation/mm/ksm.rst

Q12
The corpus states that if the kernel fails to make room for data that must be present in physical frames, it invokes the out-of-memory (OOM) killer to make room by terminating lower priority processes until pressure reduces under a safe threshold (Documentation/mm/page_tables.rst). However, the dedicated OOM documentation (Documentation/mm/oom.rst) is a stub and the corpus does not describe specific controls or factors influencing which task is selected.
Sources: Documentation/mm/page_tables.rst, Documentation/mm/oom.rst

Q13
Page migration allows moving the physical location of pages between nodes in a NUMA system while a process is running, without changing the virtual addresses the process sees. Its main intent is to reduce memory access latency by moving pages near the processor accessing them. During migration, the folio is isolated from the LRU and its refcount is increased; page table references are converted to migration entries (decrementing mapcount; if mapcount is not zero the page is not migrated); the page is locked, writeback is completed, contents and flags are copied to the new page, the page cache tree is updated, and migration entries are replaced with real PTEs so user space can access the new page.
Sources: Documentation/mm/page_migration.rst

Q14
page_owner is designed to diagnose who allocated each page, useful for debugging memory leaks or finding memory hogs. It records allocation information such as the call stack and the order of pages for each page. The cost is that it is disabled by default; enabling it increases kernel size by several kilobytes and, when enabled at runtime, requires memory to store owner information. When disabled at runtime, the overhead is marginal (two unlikely branches in the page allocator hot path).
Sources: Documentation/mm/page_owner.rst

Q15
Page table check hardens the kernel by preventing certain memory corruptions, specifically double mappings (e.g., mapping the same physical page as both anonymous and named, or as writable in multiple places) and illegal combinations of page table entry flags (e.g., userfaultfd wr-protect bit against writable flags). The checks are performed synchronously at the time new pages become accessible from userspace—when their page table entries (PTEs, PMDs, etc.) are added to the table—so corruption is caught at the time the wrong mapping occurs.
Sources: Documentation/mm/page_table_check.rst

Q17
Linux abstracts physical memory into nodes (struct pglist_data / pg_data_t), each representing a memory bank (even on UMA systems, a single node is used). Each node contains an array of zones (struct zone), which represent ranges within memory determined by architectural constraints (e.g., ZONE_DMA, ZONE_NORMAL, ZONE_HIGHMEM, ZONE_MOVABLE, ZONE_DEVICE). Zones contain free areas managed by the buddy allocator and per-CPU pagesets. The relationship is hierarchical: nodes contain zones, and zones track free pages; the page allocator uses GFP flags to determine the highest zone from which to allocate and falls back to lower zones. Struct page objects track the status of physical page frames.
Sources: Documentation/mm/physical_memory.rst

Q18
A process virtual address space is represented by struct mm_struct, which contains a maple tree of VMAs (struct vm_area_struct). Each VMA describes a virtually contiguous range with identical attributes. Locking is layered: mmap locks (a read/write semaphore at MM granularity), VMA locks (per-VMA optimistic read locks via lock_vma_under_rcu and write locks via vma_start_write, which require the mmap write lock), and rmap locks (anon_vma locks for anonymous memory and i_mmap locks for file-backed memory). Page tables have their own locks: higher-level tables use mm->page_table_lock, while PMD and PTE tables use fine-grained split locks. Lock ordering is documented (e.g., mmap_lock before i_mmap_rwsem before page_table_lock) to prevent deadlocks.
Sources: Documentation/mm/process_addrs.rst

Q20
Split page table lock solves poor page fault scalability of multi-threaded applications caused by high contention on the single mm->page_table_lock spinlock that originally protected all page tables. It introduces separate per-table locks: at the PTE and PMD levels, each table has its own lock (accessed via helpers like pte_offset_map_lock() and pmd_lock()), while higher-level tables (PUD, P4D, PGD) remain protected by mm->page_table_lock. Split locks for PTE are enabled if CONFIG_SPLIT_PTLOCK_CPUS <= NR_CPUS; PMD split locks additionally require architecture support.
Sources: Documentation/mm/split_page_table_lock.rst

Q21
The corpus provides limited information. It describes the swap cache as a per-cluster array (swap table) that maps swap entries to folios, with states for unused, folio-bound, and shadow entries (Documentation/mm/swap-table.rst). It also notes that in the multi-gen LRU, clean file pages can be evicted regardless of swap constraints, implying anonymous pages require swap backing (Documentation/mm/multigen_lru.rst). However, the corpus does not contain a comprehensive description of how reclaim and swap interact when anonymous memory must be evicted.
Sources: Documentation/mm/swap-table.rst, Documentation/mm/multigen_lru.rst

Q22
Transparent Huge Pages (THP) are intended to reduce TLB pressure, page table overhead, and improve memory allocation efficiency and performance for certain workloads by using larger page sizes (e.g., 2MB/1GB). However, THP policy is complicated by tradeoffs: hugepage allocation can fail due to memory fragmentation, in which case regular pages must be gracefully allocated instead and mixed into the same VMA without failure or significant delay; khugepaged later relocates regular pages to hugepages when available. Refcounting and splitting are complex: split_huge_page() can fail if the page is pinned, and partial unmap requires deferred splitting. The design requires graceful fallback so components unaware of hugepages continue working.
Sources: Documentation/mm/transhuge.rst

Q23
The unevictable LRU is an additional per-node LRU list that tracks unevictable folios and hides them from vmscan. It was introduced to address scalability problems where vmscan spent excessive time scanning LRU lists looking for the small fraction of evictable pages on large-memory systems. Pages that can end up there include ramfs pages, tmpfs pages with the noswap mount option, pages mapped into SHM_LOCK'd shared memory regions, and pages mapped into VM_LOCKED (mlock()ed) VMAs. Treating them separately is useful because reclaim can deal only with pages that have a chance of being reclaimed, and it also allows unevictable pages to be migrated between nodes for defragmentation, workload management, and memory hotplug.
Sources: Documentation/mm/unevictable-lru.rst

Q24
Vmapped kernel stacks are motivated by the difficulty of debugging kernel stack overflows, which can cause hard-to-diagnose corruption and make the kernel susceptible to exploits. By allocating stacks from virtually contiguous memory with guard pages (via HAVE_ARCH_VMAP_STACK and VMAP_STACK), stack overflows cause reliable faults immediately rather than silent corruption. Requirements include sufficient vmalloc space, reliable stack operation in vmalloc space, and reasonable behavior (e.g., not instantly rebooting without logging) when the stack overflows into a guard page.
Sources: Documentation/mm/vmalloced-kernel-stacks.rst

Q25
The dedicated vmalloc documentation (Documentation/mm/vmalloc.rst) is a stub with no content. The corpus indicates indirectly that vmalloc (via __vmalloc_node_range) allocates pages from the page level allocator and maps them into contiguous kernel virtual space (Documentation/mm/vmalloced-kernel-stacks.rst), implying it provides virtual contiguity without requiring physical contiguity. However, the corpus does not provide a general explanation of vmalloc's purpose or an explicit contrast with physically contiguous allocations.
Sources: Documentation/mm/vmalloc.rst, Documentation/mm/vmalloced-kernel-stacks.rst
```

</details>

<details>
<summary><strong>Linux setup steps — clone docs, prepare venv, build corpus, and record revision</strong></summary>

### 1. Sparse-clone Linux documentation

```bash
cd ~

git clone \
  --depth 1 \
  --filter=blob:none \
  --sparse \
  https://github.com/torvalds/linux.git \
  linux-docs

cd ~/linux-docs

git sparse-checkout set Documentation
```

Verify:

```bash
du -sh ~/linux-docs/Documentation
find ~/linux-docs/Documentation -type f | wc -l
```

Observed:

```text
85M     /home/aabduh/linux-docs/Documentation
11438
```

### 2. Create and activate the tokenizer/test venv

```bash
python3 -m venv ~/glm53-kv-test-venv
source ~/glm53-kv-test-venv/bin/activate
pip install transformers requests
```

### 3. Record the exact Linux revision

```bash
cd ~/linux-docs

git rev-parse HEAD
git describe --tags --always
```

Observed:

```text
654ae5d73c05bd2943d65636ce6cd0aa46e62f18
654ae5d7
```

### 4. Build the deterministic 950K-token corpus

```bash
source ~/glm53-kv-test-venv/bin/activate
python3 ~/build-linux-docs-corpus.py
```

Observed:

```text
Candidate files: 1424

========================================================================
LINUX DOCUMENTATION CORPUS BUILT
========================================================================
Tokens:   950,000
Files:    395
SHA256:   46adbc3b275f91bab6f0627cf0c055e50df2979e0577218c3315f0423b6f46e2
Corpus:   /home/aabduh/linux-docs-bench/linux-docs-950k.txt
Manifest: /home/aabduh/linux-docs-bench/linux-docs-950k-manifest.json
```

### 5. Verify the corpus manifest

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path.home() / "linux-docs-bench/linux-docs-950k-manifest.json"
m = json.loads(p.read_text())

print("tokens:", m["actual_tokens"])
print("files:", len(m["files"]))
print("sha256:", m["sha256"])

print("\nFirst 10 files:")
for x in m["files"][:10]:
    print(x)
PY
```

### 6. Run the Linux documentation exam

```bash
source ~/glm53-kv-test-venv/bin/activate
KV_LABEL=FP8 python3 ~/glm53-linux-docs-25q.py
```

</details>

<details>
<summary><strong>Full Linux corpus-builder script — build-linux-docs-corpus.py</strong></summary>

```python
#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

DOC_ROOT = Path.home() / "linux-docs" / "Documentation"
OUTDIR = Path.home() / "linux-docs-bench"
OUTDIR.mkdir(parents=True, exist_ok=True)

MODEL_REPO = "local-inference-lab/GLM-5.3-Flash-NVFP4"
TARGET_TOKENS = 950_000

SUBTREES = [
    "mm",
    "networking",
    "scheduler",
    "filesystems",
    "admin-guide",
    "core-api",
    "driver-api",
    "locking",
    "process",
    "security",
]

ALLOWED_SUFFIXES = {
    ".rst",
    ".txt",
    ".md",
}

print(f"Loading tokenizer for {MODEL_REPO} ...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    local_files_only=True,
    trust_remote_code=True,
)

print("Tokenizer loaded.")

files = []

for subtree in SUBTREES:
    root = DOC_ROOT / subtree

    if not root.exists():
        continue

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue

        if p.suffix.lower() not in ALLOWED_SUFFIXES:
            continue

        files.append(p)

print(f"Candidate files: {len(files)}")

parts = []
manifest = []
token_total = 0

for p in files:
    rel = p.relative_to(DOC_ROOT)

    try:
        text = p.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as e:
        print(f"Skipping {rel}: {e}")
        continue

    block = (
        "\n\n"
        "============================================================\n"
        f"FILE: Documentation/{rel}\n"
        "============================================================\n\n"
        f"{text}\n"
    )

    ids = tokenizer.encode(
        block,
        add_special_tokens=False,
    )

    remaining = TARGET_TOKENS - token_total

    if remaining <= 0:
        break

    if len(ids) <= remaining:
        parts.append(block)

        manifest.append({
            "path": f"Documentation/{rel}",
            "tokens": len(ids),
            "complete": True,
        })

        token_total += len(ids)

    else:
        partial_ids = ids[:remaining]

        partial_text = tokenizer.decode(
            partial_ids,
            skip_special_tokens=True,
        )

        parts.append(partial_text)

        manifest.append({
            "path": f"Documentation/{rel}",
            "tokens": len(partial_ids),
            "complete": False,
        })

        token_total += len(partial_ids)
        break

corpus = "".join(parts)

sha = hashlib.sha256(
    corpus.encode("utf-8")
).hexdigest()

corpus_path = OUTDIR / "linux-docs-950k.txt"
manifest_path = OUTDIR / "linux-docs-950k-manifest.json"

corpus_path.write_text(
    corpus,
    encoding="utf-8",
)

manifest_data = {
    "target_tokens": TARGET_TOKENS,
    "actual_tokens": token_total,
    "sha256": sha,
    "subtrees": SUBTREES,
    "files": manifest,
}

manifest_path.write_text(
    json.dumps(
        manifest_data,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("=" * 72)
print("LINUX DOCUMENTATION CORPUS BUILT")
print("=" * 72)
print(f"Tokens:   {token_total:,}")
print(f"Files:    {len(manifest)}")
print(f"SHA256:   {sha}")
print(f"Corpus:   {corpus_path}")
print(f"Manifest: {manifest_path}")
```

</details>

<details>
<summary><strong>Full Test 4 exam script — glm53-linux-docs-25q.py</strong></summary>

```python
#!/usr/bin/env python3

import hashlib
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


URL = "http://127.0.0.1:15015/v1/chat/completions"
MODEL = "GLM-5.3-Flash-TP3"

CORPUS = Path.home() / "linux-docs-bench/linux-docs-950k.txt"
MANIFEST = Path.home() / "linux-docs-bench/linux-docs-950k-manifest.json"

OUTDIR = Path.home() / "linux-docs-bench"
OUTDIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = OUTDIR / "linux-docs-25q-result.json"
RAW_FILE = OUTDIR / "linux-docs-25q-raw-response.txt"

MODEL_REPO = "local-inference-lab/GLM-5.3-Flash-NVFP4"
MAX_OUTPUT_TOKENS = 32768

EXPECTED_CORPUS_SHA = (
    "46adbc3b275f91bab6f0627cf0c055e50df2979e0577218c3315f0423b6f46e2"
)

EXPECTED_KERNEL_COMMIT = (
    "654ae5d73c05bd2943d65636ce6cd0aa46e62f18"
)


QUESTION_BANK = [
    {
        "id": "Q01",
        "source_hint": "Documentation/mm/active_mm.rst",
        "question":
            "Explain the distinction between mm and active_mm for a Linux task, "
            "including why a kernel thread can have no normal mm while still "
            "having an active_mm."
    },
    {
        "id": "Q02",
        "source_hint": "Documentation/mm/allocation-profiling.rst",
        "question":
            "How does Linux allocation profiling attribute memory allocations, "
            "and what is the intended use of the resulting information?"
    },
    {
        "id": "Q03",
        "source_hint": "Documentation/mm/arch_pgtable_helpers.rst",
        "question":
            "What purpose do the generic page-table helper semantics serve for "
            "architecture implementations, and what kind of errors are the "
            "associated tests intended to detect?"
    },
    {
        "id": "Q04",
        "source_hint": "Documentation/mm/balance.rst",
        "question":
            "Describe the memory-balancing behavior documented here. What "
            "conditions cause balancing or reclaim activity and what is it "
            "trying to accomplish?"
    },
    {
        "id": "Q05",
        "source_hint": "Documentation/mm/damon/design.rst",
        "question":
            "Describe DAMON's core design. How does it monitor memory access "
            "patterns while controlling monitoring overhead?"
    },
    {
        "id": "Q06",
        "source_hint": "Documentation/mm/damon/design.rst",
        "question":
            "Within DAMON, explain the relationship between monitoring targets, "
            "address spaces or regions, sampling, aggregation, and region "
            "adaptation."
    },
    {
        "id": "Q07",
        "source_hint": "Documentation/mm/damon/design.rst",
        "question":
            "What are DAMOS schemes and how do they turn DAMON's monitoring "
            "information into memory-management actions or policies?"
    },
    {
        "id": "Q08",
        "source_hint": "Documentation/mm/damon/faq.rst",
        "question":
            "Identify one limitation, common misunderstanding, or important "
            "operational consideration described in the DAMON FAQ and explain "
            "it accurately."
    },
    {
        "id": "Q09",
        "source_hint": "Documentation/mm/damon/maintainer-profile.rst",
        "question":
            "Summarize the documented expectations for DAMON development or "
            "maintenance, including the workflow or review expectations that a "
            "contributor should know."
    },
    {
        "id": "Q10",
        "source_hint": "Documentation/mm/highmem.rst",
        "question":
            "Why does high memory require special handling on affected "
            "architectures, and what distinction does the documentation make "
            "between permanent and temporary mappings?"
    },
    {
        "id": "Q11",
        "source_hint": "Documentation/mm/ksm.rst",
        "question":
            "Explain how Kernel Samepage Merging identifies and merges duplicate "
            "anonymous pages, and describe the copy-on-write behavior after pages "
            "have been merged."
    },
    {
        "id": "Q12",
        "source_hint": "Documentation/mm/oom.rst",
        "question":
            "What is the purpose of Linux's out-of-memory handling, and what "
            "controls or factors described in the documentation influence which "
            "task is selected when memory exhaustion occurs?"
    },
    {
        "id": "Q13",
        "source_hint": "Documentation/mm/page_migration.rst",
        "question":
            "Explain what page migration is, why Linux performs it, and what "
            "must happen to mappings or references while a page is being migrated."
    },
    {
        "id": "Q14",
        "source_hint": "Documentation/mm/page_owner.rst",
        "question":
            "What problem is page_owner designed to diagnose, what information "
            "does it record, and what cost or tradeoff comes with enabling it?"
    },
    {
        "id": "Q15",
        "source_hint": "Documentation/mm/page_table_check.rst",
        "question":
            "What classes of incorrect page-table mappings is page table check "
            "intended to detect, and when are those checks performed?"
    },
    {
        "id": "Q16",
        "source_hint": "Documentation/mm/pagemap.rst",
        "question":
            "Describe what information userspace can obtain through the pagemap "
            "interfaces and why access to some physical-page information is "
            "restricted."
    },
    {
        "id": "Q17",
        "source_hint": "Documentation/mm/physical_memory.rst",
        "question":
            "Explain how Linux represents physical memory in terms of nodes, "
            "zones, and pages. Describe the relationship among those structures."
    },
    {
        "id": "Q18",
        "source_hint": "Documentation/mm/process_addrs.rst",
        "question":
            "Describe the major data structures and locking considerations used "
            "to represent a process virtual address space."
    },
    {
        "id": "Q19",
        "source_hint": "Documentation/mm/shmem.rst",
        "question":
            "Explain the role of shmem or tmpfs in Linux memory management and "
            "how its pages differ from ordinary anonymous memory or persistent "
            "filesystem-backed data."
    },
    {
        "id": "Q20",
        "source_hint": "Documentation/mm/split_page_table_lock.rst",
        "question":
            "What problem does split page-table locking solve, and at which "
            "page-table levels can separate locks be used?"
    },
    {
        "id": "Q21",
        "source_hint": "Documentation/mm/swap.rst",
        "question":
            "Describe the role of swap in Linux memory management and explain "
            "how reclaim and swap interact when anonymous memory must be evicted."
    },
    {
        "id": "Q22",
        "source_hint": "Documentation/mm/transhuge.rst",
        "question":
            "What are Transparent Huge Pages intended to improve, and what "
            "tradeoffs or failure modes make THP policy more complicated than "
            "simply using huge pages everywhere?"
    },
    {
        "id": "Q23",
        "source_hint": "Documentation/mm/unevictable-lru.rst",
        "question":
            "Why does Linux maintain an unevictable LRU, what kinds of pages can "
            "end up there, and why is treating those pages separately useful to "
            "reclaim?"
    },
    {
        "id": "Q24",
        "source_hint": "Documentation/mm/vmalloced-kernel-stacks.rst",
        "question":
            "What are the motivations for vmapped kernel stacks and what "
            "memory-safety or debugging benefits do they provide?"
    },
    {
        "id": "Q25",
        "source_hint": "Documentation/mm/vmalloc.rst",
        "question":
            "Explain the purpose of vmalloc and contrast virtually contiguous "
            "memory obtained through vmalloc with physically contiguous "
            "allocations."
    },
]


print("Loading corpus ...")

corpus = CORPUS.read_text(
    encoding="utf-8",
    errors="replace",
)

manifest = json.loads(
    MANIFEST.read_text(encoding="utf-8")
)

corpus_sha = hashlib.sha256(
    corpus.encode("utf-8")
).hexdigest()

print(f"Corpus SHA256: {corpus_sha}")

if corpus_sha != EXPECTED_CORPUS_SHA:
    raise RuntimeError(
        "Corpus SHA does not match the previously generated 950K corpus."
    )

manifest_paths = {
    item["path"]
    for item in manifest["files"]
}

print(f"Manifest files: {len(manifest_paths)}")


selected = []
missing = []

for q in QUESTION_BANK:
    if q["source_hint"] in manifest_paths:
        selected.append(q)
    else:
        missing.append(q)

print()
print(f"Available question-bank items: {len(selected)}")

if missing:
    print("Unavailable source files:")
    for q in missing:
        print(f"  {q['id']}: {q['source_hint']}")

if len(selected) < 20:
    raise RuntimeError(
        f"Only {len(selected)} of the 25 question sources exist in this "
        "950K corpus. Need at least 20. See missing list above."
    )

selected = selected[:25]

print()
print(f"Running {len(selected)} questions:")
for q in selected:
    print(f"  {q['id']}: {q['question']}")


question_text = []

for q in selected:
    question_text.append(
        f"{q['id']}. {q['question']}"
    )

instructions = """
You are given a corpus consisting only of Linux kernel documentation.

Answer every question using ONLY information supported by the supplied
documentation corpus.

For each question:

1. Give a technically precise answer.
2. Give the exact Documentation/... file path or paths that support it.
3. Do not rely on outside Linux knowledge if the corpus does not support a
   claim.
4. If the corpus does not contain enough information, say
   "INSUFFICIENT CORPUS EVIDENCE".
5. Keep each answer concise but complete.
6. Do not skip any question.

Return valid JSON using exactly this general schema:

{
  "Q01": {
    "answer": "...",
    "sources": ["Documentation/..."]
  },
  "Q02": {
    "answer": "...",
    "sources": ["Documentation/..."]
  }
}

Do not wrap the JSON in Markdown fences.
"""

user_text = (
    "================ BEGIN LINUX DOCUMENTATION CORPUS ================\n\n"
    + corpus
    + "\n\n================ END LINUX DOCUMENTATION CORPUS ==================\n\n"
    + instructions
    + "\n\nQUESTIONS:\n\n"
    + "\n\n".join(question_text)
)


print()
print(f"Loading tokenizer for {MODEL_REPO} ...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    local_files_only=True,
    trust_remote_code=True,
)

prompt_tokens_local = len(
    tokenizer.encode(
        user_text,
        add_special_tokens=False,
    )
)

print(f"Local raw prompt tokens: {prompt_tokens_local:,}")
print(f"Output allowance:        {MAX_OUTPUT_TOKENS:,}")

if prompt_tokens_local + MAX_OUTPUT_TOKENS >= 1_048_576:
    print()
    print(
        "WARNING: raw prompt + full output allowance approaches/exceeds "
        "the configured context window."
    )


payload = {
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
}

print()
print("Sending Linux-documentation exam ...")

start = time.perf_counter()

r = requests.post(
    URL,
    json=payload,
    timeout=1800,
)

elapsed = time.perf_counter() - start

r.raise_for_status()

obj = r.json()

choice = obj["choices"][0]
message = choice["message"]

content = message.get("content") or ""
reasoning = (
    message.get("reasoning")
    or message.get("reasoning_content")
    or ""
)

finish_reason = choice.get("finish_reason")
usage = obj.get("usage", {})

RAW_FILE.write_text(
    content,
    encoding="utf-8",
)


parsed = None

try:
    parsed = json.loads(content)
except Exception:
    start_json = content.find("{")
    end_json = content.rfind("}")

    if start_json != -1 and end_json > start_json:
        try:
            parsed = json.loads(
                content[start_json:end_json + 1]
            )
        except Exception:
            pass


answered = 0
source_path_hits = 0
unknown_source_paths = []

if isinstance(parsed, dict):
    for q in selected:
        ans = parsed.get(q["id"])

        if not isinstance(ans, dict):
            continue

        answer_text = str(
            ans.get("answer", "")
        ).strip()

        if answer_text:
            answered += 1

        sources = ans.get("sources", [])

        if isinstance(sources, list):
            valid_for_question = False

            for source in sources:
                if source in manifest_paths:
                    valid_for_question = True
                else:
                    unknown_source_paths.append(
                        (q["id"], source)
                    )

            if valid_for_question:
                source_path_hits += 1


result = {
    "kernel_commit": EXPECTED_KERNEL_COMMIT,
    "corpus_sha256": corpus_sha,
    "manifest_files": len(manifest_paths),
    "questions": selected,
    "elapsed_sec": elapsed,
    "finish_reason": finish_reason,
    "usage": usage,
    "reasoning_chars": len(reasoning),
    "answered_questions": answered,
    "questions_with_manifest_source": source_path_hits,
    "unknown_source_paths": unknown_source_paths,
    "response": parsed if parsed is not None else content,
}

RESULT_FILE.write_text(
    json.dumps(
        result,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 80)
print("LINUX DOCUMENTATION 25-QUESTION RESULT")
print("=" * 80)

print(f"Questions requested:      {len(selected)}")
print(f"Questions answered:       {answered}")
print(f"Valid source-path answers:{source_path_hits}")
print(f"Elapsed:                  {elapsed:.2f} sec")
print(f"API prompt tokens:        {usage.get('prompt_tokens')}")
print(f"Output tokens:            {usage.get('completion_tokens')}")
print(f"Reasoning chars:          {len(reasoning)}")
print(f"Finish reason:            {finish_reason}")

if unknown_source_paths:
    print()
    print("Model cited paths absent from the corpus manifest:")
    for qid, source in unknown_source_paths:
        print(f"  {qid}: {source}")

print()
print(f"Parsed result: {RESULT_FILE}")
print(f"Raw response:  {RAW_FILE}")

print()
print("=" * 80)
print("MODEL ANSWERS")
print("=" * 80)

if isinstance(parsed, dict):
    for q in selected:
        qid = q["id"]
        ans = parsed.get(qid)

        print()
        print("-" * 80)
        print(f"{qid}: {q['question']}")
        print("-" * 80)

        if isinstance(ans, dict):
            print(ans.get("answer"))
            print()
            print(
                "Sources:",
                ", ".join(ans.get("sources", []))
            )
        else:
            print("MISSING / UNPARSEABLE ANSWER")

else:
    print(content)
```

</details>

<details>
<summary><strong>Test 4 run command and observed summary</strong></summary>

```bash
source ~/glm53-kv-test-venv/bin/activate
KV_LABEL=FP8 python3 ~/glm53-linux-docs-25q.py
```

Observed:

```text
================================================================================
LINUX DOCUMENTATION 25-QUESTION RESULT
================================================================================
Questions requested:      23
Questions answered:       23
Valid source-path answers:23
Elapsed:                  354.49 sec
API prompt tokens:        950462
Output tokens:            18123
Reasoning chars:          65909
Finish reason:            stop
```

Saved files:

```text
/home/aabduh/linux-docs-bench/linux-docs-25q-result.json
/home/aabduh/linux-docs-bench/linux-docs-25q-raw-response.txt
```

</details>

## Interpretation

This test is the closest in the validation set to a real long-context technical workflow. Unlike the synthetic tests, it required GLM to operate over hundreds of real upstream documents, locate relevant information among 395 files, synthesize technical explanations, and return source attribution for each answer.

Current structural/source-grounding result:

```text
23/23 questions answered
23/23 cited at least one valid path from the supplied corpus
finish_reason=stop
```

A separate semantic grading pass can convert this into a source-verified technical accuracy score.

---

# Test 5 - Source-Code Call-Graph / Runtime Trace Exam

## Purpose

This test evaluates whether GLM-5.3-Flash can reason over an almost one-million-token corpus of real runtime source code rather than documentation or synthetic retrieval facts. The corpus was extracted from the exact immutable R17 TP3 image and contains GLM5Next, KDA/GDN, B12X, tensor/expert-parallel, model-loader, quantization, speculative-decoding, CUDA-graph, prefix-cache, KV-cache, and launcher/runtime-policy code.

The final procedure uses batched requests against the **same exact 950,000-token source corpus**. Batching changes only the number of questions requested per completion; every included answer still receives essentially the full source corpus in its prompt. Earlier abandoned/truncated runs are intentionally omitted from this section so they cannot be confused with the authoritative results.

## Exact runtime image

```text
infernix/vllm@sha256:187432cdb974645089bdcb9eb526a5e7b99ff00f25493698f037f24722884d4c
```

Docker inspection:

```text
ID=sha256:187432cdb974645089bdcb9eb526a5e7b99ff00f25493698f037f24722884d4c
RepoDigests=["infernix/vllm@sha256:187432cdb974645089bdcb9eb526a5e7b99ff00f25493698f037f24722884d4c"]
```

## Source corpus identity

```text
Corpus tokens: 950,000
Files:         115
SHA256:        4f825adda226bbf18d6c52dc73b7421511a995f030a2fb1b38d47b8ea81ef774
Required anchors present: 24/24
```

## Authoritative status

The final authoritative Test 5 result covers **all 25 questions**. Every included run completed naturally with `finish_reason=stop`; earlier truncated/abandoned attempts are intentionally omitted so they cannot be confused with the final result.

| Question(s) | Authoritative run | Result |
|---|---|---|
| Q01-Q04 | Batch 1 | 4/4 answered, all cited paths valid, `stop` |
| Q05-Q08 | Batch 2 | 4/4 answered, all cited paths valid, `stop` |
| Q09 | Batch 8 | 1/1 answered, all cited paths valid, `stop` |
| Q10 | Batch 9 | 1/1 answered, all cited paths valid, `stop` |
| Q11 | Batch 10 | 1/1 answered, all cited paths valid, `stop` |
| Q12 | Batch 11 | 1/1 answered, >=1 valid source path, `stop` |
| Q13-Q16 | Batch 4 | 4/4 answered, all cited paths valid, `stop` |
| Q17-Q20 | Batch 5 | 4/4 answered, >=1 valid source path, `stop` |
| Q21-Q23 | Batch 6 | 3/3 answered, >=1 valid source path, `stop` |
| Q24 | Batch 12 | 1/1 answered, all cited paths valid, `stop` |
| Q25 | Batch 13 | 1/1 answered, all cited paths valid, `stop` |

Across the authoritative runs:

```text
Questions completed:             25/25
Completed with >=1 valid source: 25/25
All cited paths valid:           21/25
Included-run finish reason:      stop
Corpus tokens per request:       950,000
API prompt tokens per request:   ~950,235-950,362
Default generation cap:          64,000
Q10 generation cap:              90,000
```

The four questions that cited at least one path outside the 115-file manifest still cited valid in-corpus sources as well. This remains a **structural/source-grounding result**, not a claim that every answer has been independently line-by-line semantically graded.

<details>
<summary><strong>Final Test 5 setup - source extraction</strong></summary>

```bash
mkdir -p ~/glm53-source-bench/extracted
rm -rf ~/glm53-source-bench/extracted/*

IMG='infernix/vllm@sha256:187432cdb974645089bdcb9eb526a5e7b99ff00f25493698f037f24722884d4c'

cid=$(sudo docker create "$IMG")

sudo docker cp "$cid":/opt/glm53-flash/vllm \
  ~/glm53-source-bench/extracted/vllm

sudo docker cp "$cid":/opt/glm53-flash/b12x \
  ~/glm53-source-bench/extracted/b12x

sudo docker cp "$cid":/usr/local/bin \
  ~/glm53-source-bench/extracted/usr-local-bin

sudo docker cp "$cid":/usr/local/libexec \
  ~/glm53-source-bench/extracted/usr-local-libexec 2>/dev/null || true

sudo docker rm "$cid"

sudo chown -R "$USER:$USER" ~/glm53-source-bench
```

Verification:

```bash
du -sh ~/glm53-source-bench/extracted/*
find ~/glm53-source-bench/extracted -type f | wc -l

sudo docker image inspect "$IMG" \
  --format 'ID={{.Id}} RepoDigests={{json .RepoDigests}}'
```

Observed:

```text
50M     /home/aabduh/glm53-source-bench/extracted/b12x
72M     /home/aabduh/glm53-source-bench/extracted/usr-local-bin
28K     /home/aabduh/glm53-source-bench/extracted/usr-local-libexec
815M    /home/aabduh/glm53-source-bench/extracted/vllm
14712
```

</details>

<details>
<summary><strong>Final corpus-builder script - build-glm53-source-corpus.py</strong></summary>

```python
#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path.home() / "glm53-source-bench" / "extracted"
OUTDIR = Path.home() / "glm53-source-bench"
OUTDIR.mkdir(parents=True, exist_ok=True)

MODEL_REPO = "local-inference-lab/GLM-5.3-Flash-NVFP4"
TARGET_TOKENS = 950_000

ALLOWED = {
    ".py", ".sh", ".bash", ".cu", ".cuh", ".cc", ".cpp", ".c",
    ".h", ".hpp", ".yaml", ".yml", ".toml", ".json",
}

REQUIRED_ANCHORS = [
    "GLM53_KDA_DECODE_BACKEND", "GLM53_KDA_PREFILL_BACKEND",
    "B12X_PCIE_ALLREDUCE", "VLLM_PCIE_ALLREDUCE_BACKEND",
    "expert_parallel", "SPECULATOR", "prefix_caching",
    "serve-glm53-flash-tp3-r17", "glm5next", "kimi_gdn", "B12X",
    "tensor_parallel", "ColumnParallelLinear", "RowParallelLinear",
    "weight_loader", "load_weights", "quant", "fp4", "moe", "MTP",
    "dflash", "mm_encoder_tp_mode", "cudagraph", "kv_cache",
]

PRIORITY_PATTERNS = [
    "serve-glm53", "glm5next", "kimi_gdn", "gdn", "kda", "b12x",
    "parallel_state", "distributed", "allreduce", "all_reduce",
    "tensor_parallel", "expert_parallel", "moe", "model_loader", "loader",
    "quant", "linear", "attention", "spec", "draft", "mtp", "dflash",
    "scheduler", "kv_cache", "prefix", "cudagraph",
]

print(f"Loading tokenizer for {MODEL_REPO} ...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    local_files_only=True,
    trust_remote_code=True,
)
print("Tokenizer loaded.")

candidates = []
for p in ROOT.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in ALLOWED:
        continue
    lower = str(p).lower()
    if any(x in lower for x in [
        "/.git/", "/__pycache__/", "/build/", "/dist/", "/node_modules/",
        "/tests/", "/test/", "/benchmarks/", "/benchmark/",
    ]):
        continue
    candidates.append(p)

print(f"Production candidate files: {len(candidates):,}")

records = []
anchor_hits = {anchor: [] for anchor in REQUIRED_ANCHORS}

for p in candidates:
    rel = p.relative_to(ROOT)
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue

    src_lower = src.lower()
    hits = []
    for anchor in REQUIRED_ANCHORS:
        if anchor.lower() in src_lower:
            hits.append(anchor)
            anchor_hits[anchor].append(str(rel))

    records.append({"path": p, "rel": rel, "src": src, "hits": hits})

print()
print("=" * 80)
print("ANCHOR DISCOVERY")
print("=" * 80)

for anchor in REQUIRED_ANCHORS:
    paths = anchor_hits[anchor]
    print()
    print(f"{anchor}: {len(paths)} file(s)")
    for path in paths[:8]:
        print(f"  {path}")
    if len(paths) > 8:
        print(f"  ... +{len(paths) - 8} more")

missing_on_disk = [
    anchor for anchor, paths in anchor_hits.items() if not paths
]

if missing_on_disk:
    print()
    print("WARNING: these anchors were not found anywhere in the")
    print("extracted production source:")
    for anchor in missing_on_disk:
        print(f"  {anchor}")


def base_priority(record):
    s = str(record["rel"]).lower()
    if "usr-local-bin" in s:
        return -3000
    if "usr-local-libexec" in s:
        return -2900
    if record["hits"]:
        return -2000
    for i, pattern in enumerate(PRIORITY_PATTERNS):
        if pattern in s:
            return i
    if s.startswith("b12x/"):
        return 200
    if s.startswith("vllm/"):
        return 300
    return 1000


records.sort(key=lambda r: (
    base_priority(r), -len(r["hits"]), str(r["rel"]),
))

parts = []
manifest = []
token_total = 0

for record in records:
    rel = record["rel"]
    src = record["src"]
    block = (
        "\n\n"
        "======================================================================\n"
        f"FILE: {rel}\n"
        "======================================================================\n\n"
        f"{src}\n"
    )

    ids = tokenizer.encode(block, add_special_tokens=False)
    remaining = TARGET_TOKENS - token_total
    if remaining <= 0:
        break

    if len(ids) <= remaining:
        parts.append(block)
        manifest.append({
            "path": str(rel), "tokens": len(ids), "complete": True,
            "priority": base_priority(record), "anchor_hits": record["hits"],
        })
        token_total += len(ids)
    else:
        partial_ids = ids[:remaining]
        parts.append(tokenizer.decode(partial_ids, skip_special_tokens=True))
        manifest.append({
            "path": str(rel), "tokens": len(partial_ids), "complete": False,
            "priority": base_priority(record), "anchor_hits": record["hits"],
        })
        token_total += len(partial_ids)
        break

corpus = "".join(parts)
sha = hashlib.sha256(corpus.encode("utf-8")).hexdigest()

corpus_path = OUTDIR / "glm53-source-950k.txt"
manifest_path = OUTDIR / "glm53-source-950k-manifest.json"

corpus_path.write_text(corpus, encoding="utf-8")
manifest_path.write_text(json.dumps({
    "target_tokens": TARGET_TOKENS,
    "actual_tokens": token_total,
    "sha256": sha,
    "missing_anchors_in_extracted_source": missing_on_disk,
    "files": manifest,
}, indent=2), encoding="utf-8")

corpus_lower = corpus.lower()
present = [a for a in REQUIRED_ANCHORS if a.lower() in corpus_lower]
absent = [a for a in REQUIRED_ANCHORS if a.lower() not in corpus_lower]

print()
print("=" * 80)
print("GLM-5.3 SOURCE CORPUS BUILT")
print("=" * 80)
print(f"Tokens:   {token_total:,}")
print(f"Files:    {len(manifest):,}")
print(f"SHA256:   {sha}")
print(f"Corpus:   {corpus_path}")
print(f"Manifest: {manifest_path}")
print()
print(f"Required anchors present in final corpus: {len(present)}/{len(REQUIRED_ANCHORS)}")

if absent:
    print()
    print("Still absent from final corpus:")
    for anchor in absent:
        print(f"  {anchor}")

print()
print("First 40 included files:")
for x in manifest[:40]:
    hits = ", ".join(x["anchor_hits"]) if x["anchor_hits"] else "-"
    print(f"{x['tokens']:>7,} tok | {x['path']} | anchors={hits}")
```

Run:

```bash
source ~/glm53-kv-test-venv/bin/activate
python3 ~/build-glm53-source-corpus.py
```

Expected identity:

```text
Tokens:   950,000
Files:    115
SHA256:   4f825adda226bbf18d6c52dc73b7421511a995f030a2fb1b38d47b8ea81ef774
Required anchors present in final corpus: 24/24
```

</details>

<details>
<summary><strong>Final batched exam script - glm53-source-trace-batched.py</strong></summary>

```python
#!/usr/bin/env python3

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

URL = "http://127.0.0.1:15015/v1/chat/completions"
MODEL = "GLM-5.3-Flash-TP3"
ROOT = Path.home() / "glm53-source-bench"
CORPUS = ROOT / "glm53-source-950k.txt"
MANIFEST = ROOT / "glm53-source-950k-manifest.json"
MODEL_REPO = "local-inference-lab/GLM-5.3-Flash-NVFP4"
DEFAULT_MAX_OUTPUT_TOKENS = 64000
Q10_MAX_OUTPUT_TOKENS = 90000

QUESTIONS = [
    {"id":"Q01","anchors":["GLM5","forward"],"q":"Trace the top-level forward path for the GLM model represented in this corpus. Identify the principal model classes/functions involved and explain how execution progresses from the model entry point into its transformer layers."},
    {"id":"Q02","anchors":["glm5next"],"q":"Explain how the GLM5Next implementation constructs its layers. Identify where layer types are selected and which components are instantiated for attention, feed-forward/MoE, and any linear-attention or GDN path."},
    {"id":"Q03","anchors":["kimi_gdn"],"q":"Trace the Kimi/GDN linear-attention path from the GLM model into the implementation that performs the operation. Identify the relevant classes/functions and their source files."},
    {"id":"Q04","anchors":["GLM53_KDA_DECODE_BACKEND"],"q":"Trace GLM53_KDA_DECODE_BACKEND from configuration or launcher handling to the code that selects a decode implementation. List each important decision point and the resulting backend."},
    {"id":"Q05","anchors":["GLM53_KDA_PREFILL_BACKEND"],"q":"Trace GLM53_KDA_PREFILL_BACKEND through the runtime. Explain how prefill backend selection differs from decode backend selection and identify the final implementation called."},
    {"id":"Q06","anchors":["B12X"],"q":"Identify every major place where B12X participates in this GLM runtime. Separate attention, linear, MoE, KDA, and collective/all-reduce uses where supported by the corpus."},
    {"id":"Q07","anchors":["B12X_PCIE_ALLREDUCE"],"q":"Trace the B12X PCIe all-reduce configuration from launcher or environment variables into runtime collective selection. Explain the conditions under which the B12X path is used."},
    {"id":"Q08","anchors":["VLLM_PCIE_ALLREDUCE_BACKEND"],"q":"Explain how VLLM_PCIE_ALLREDUCE_BACKEND affects collective dispatch in this source tree, including the relevant functions or classes and fallback behavior visible in the corpus."},
    {"id":"Q09","anchors":["tensor_parallel"],"q":"Trace tensor-parallel world-size information from distributed initialization into model layer construction. Give at least two concrete examples where TP world size changes tensor or module geometry."},
    {"id":"Q10","anchors":["expert_parallel"],"q":"Trace expert-parallel configuration through the model and MoE stack. Explain where EP rank/world size are obtained and how experts are assigned or partitioned."},
    {"id":"Q11","anchors":["ColumnParallelLinear"],"q":"Explain the implementation contract of ColumnParallelLinear in this corpus: how the output dimension is partitioned, how weights are represented per rank, and how its weight loader handles sharded parameters."},
    {"id":"Q12","anchors":["RowParallelLinear"],"q":"Contrast RowParallelLinear with ColumnParallelLinear. Trace where communication occurs and explain which tensor dimension is assumed to be partitioned in each case."},
    {"id":"Q13","anchors":["weight_loader"],"q":"Trace model checkpoint loading for at least one GLM attention or GDN projection from the generic loader into the destination parameter. Identify any custom loader logic, shard IDs, packing, padding, or rank-local slicing involved."},
    {"id":"Q14","anchors":["load_weights"],"q":"Locate the GLM model's load_weights implementation and explain how checkpoint parameter names are transformed or mapped to runtime module parameters. Give several concrete mappings."},
    {"id":"Q15","anchors":["quant"],"q":"Trace how quantization configuration reaches at least two GLM linear or MoE modules. Explain which objects decide the concrete quantized implementation."},
    {"id":"Q16","anchors":["fp4"],"q":"Identify the NVFP4/FP4-specific execution paths represented in the supplied source. Explain where FP4 affects weight loading or kernel/backend selection rather than merely configuration."},
    {"id":"Q17","anchors":["moe"],"q":"Trace one token through the GLM MoE path at a code-structure level: router/gating, expert selection, expert execution, and combination of expert outputs. Cite all important source files."},
    {"id":"Q18","anchors":["MTP"],"q":"Trace speculative MTP configuration from the launcher/runtime into vLLM's speculative decoding implementation. Identify what changes when MTP is enabled and where speculative tokens are produced or verified."},
    {"id":"Q19","anchors":["dflash"],"q":"Trace the DFlash/DFlash2 speculative path represented in this image. Explain how its draft model is selected and how this path differs structurally from the MTP path."},
    {"id":"Q20","anchors":["SPECULATOR"],"q":"Starting only from SPECULATOR in the launcher, reconstruct the supported speculation modes and the runtime arguments or configuration generated for each mode."},
    {"id":"Q21","anchors":["mm_encoder_tp_mode"],"q":"Trace mm_encoder_tp_mode through configuration validation and model/runtime setup. Explain what the 'weights' mode changes relative to other modes visible in the corpus."},
    {"id":"Q22","anchors":["cudagraph"],"q":"Trace CUDA graph configuration from launcher/runtime settings to the code that controls graph capture. Explain how capture sizes and graph mode affect which requests can use captured graphs."},
    {"id":"Q23","anchors":["prefix_caching"],"q":"Trace prefix-caching enablement from runtime configuration into the scheduler/cache path. Identify which major components consume this setting and what behavior changes."},
    {"id":"Q24","anchors":["kv_cache"],"q":"Trace KV-cache dtype/configuration into cache allocation and attention execution. Identify where FP8 KV is represented and which components must agree on its layout or dtype."},
    {"id":"Q25","anchors":["serve-glm53-flash-tp3-r17"],"q":"Reconstruct the complete TP3 R17 launch policy from the supplied launcher and runtime source: locked model/revision, TP/DCP/EP, scheduler limits, KV dtype, attention/backend selections, collectives, graph mode, prefix caching, multimodal TP mode, and speculation handling. Then identify which of those settings are enforced by launcher policy versus interpreted deeper in vLLM."},
]

# Final reproducible batch map. Each request still gets the same 950K corpus.
BATCHES = {
    1: ("Q01", "Q04"),
    2: ("Q05", "Q08"),
    4: ("Q13", "Q16"),
    5: ("Q17", "Q20"),
    6: ("Q21", "Q23"),
    8: ("Q09", "Q09"),
    9: ("Q10", "Q10"),
    10: ("Q11", "Q11"),
    11: ("Q12", "Q12"),
    12: ("Q24", "Q24"),
    13: ("Q25", "Q25"),
}


def get_batch_questions(batch_num):
    start_id, end_id = BATCHES[batch_num]
    start_num = int(start_id[1:])
    end_num = int(end_id[1:])
    return [q for q in QUESTIONS if start_num <= int(q["id"][1:]) <= end_num]


parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, choices=sorted(BATCHES), required=True)
args = parser.parse_args()
BATCH = args.batch

# Q10 is substantially more reasoning-heavy than the other questions.
# Keep the same exact 950K corpus, but give only Batch 9 a 90K
# completion allowance. All other batches remain at 64K.
MAX_OUTPUT_TOKENS = (
    Q10_MAX_OUTPUT_TOKENS
    if BATCH == 9
    else DEFAULT_MAX_OUTPUT_TOKENS
)

RESULT_FILE = ROOT / f"glm53-source-trace-batch{BATCH}-result.json"
RAW_FILE = ROOT / f"glm53-source-trace-batch{BATCH}-raw.txt"

print("Loading source corpus ...")
corpus = CORPUS.read_text(encoding="utf-8", errors="replace")
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
sha = hashlib.sha256(corpus.encode("utf-8")).hexdigest()

print(f"Corpus tokens manifest: {manifest['actual_tokens']:,}")
print(f"Corpus SHA256:          {sha}")
print(f"Manifest files:         {len(manifest['files']):,}")
if sha != manifest["sha256"]:
    raise RuntimeError("Corpus SHA does not match manifest.")

batch_questions = get_batch_questions(BATCH)
corpus_lower = corpus.lower()
selected = []
skipped = []

for item in batch_questions:
    missing = [a for a in item["anchors"] if a.lower() not in corpus_lower]
    if missing:
        skipped.append((item["id"], missing))
    else:
        selected.append(item)

print()
print("=" * 88)
print(f"SOURCE TRACE BATCH {BATCH}")
print("=" * 88)
print(f"Batch range: {BATCHES[BATCH][0]}-{BATCHES[BATCH][1]}")
print(f"Questions available: {len(selected)}/{len(batch_questions)}")

if skipped:
    print("Skipped because anchors were absent:")
    for qid, missing in skipped:
        print(f"  {qid}: " + ", ".join(missing))
if not selected:
    raise RuntimeError("No supported questions in this batch.")

print()
print("Running:")
for item in selected:
    print(f"  {item['id']}: {item['q']}")

questions = "\n\n".join(f"{x['id']}. {x['q']}" for x in selected)

instructions = """
You are analyzing the EXACT source-code corpus supplied above.

Answer using ONLY code, comments, configuration, and launcher logic present
in that corpus. Do not fill missing implementation details using outside
knowledge of vLLM, CUDA, B12X, GLM, or speculative decoding.

This is a source-tracing exam, not a general knowledge exam.

For every answer:
1. Trace the implementation across files rather than merely defining the feature.
2. Name concrete classes, functions, variables, environment variables, or methods when the corpus provides them.
3. Give the exact corpus-relative source path(s) supporting the answer.
4. Distinguish launcher/configuration policy from actual runtime implementation.
5. If some requested part cannot be established from the corpus, explicitly say "INSUFFICIENT CORPUS EVIDENCE" for that part rather than guessing.
6. Answer every question in this batch.
7. Keep each answer detailed enough to establish the code path, but avoid repeating unrelated implementation details.

Return valid JSON using exactly this structure:
{
  "Q01": {
    "answer": "...",
    "sources": ["vllm/...", "b12x/..."]
  }
}
Return ONLY the JSON object.
Do not wrap the JSON in Markdown fences.
"""

user_text = (
    "================ BEGIN EXACT SOURCE CORPUS ================\n\n"
    + corpus
    + "\n\n================ END EXACT SOURCE CORPUS ==================\n\n"
    + instructions
    + "\n\nQUESTIONS FOR THIS BATCH:\n\n"
    + questions
)

print()
print(f"Loading tokenizer for {MODEL_REPO} ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    local_files_only=True,
    trust_remote_code=True,
)
raw_tokens = len(tokenizer.encode(user_text, add_special_tokens=False))
print(f"Local raw prompt tokens: {raw_tokens:,}")
print(f"Output allowance:        {MAX_OUTPUT_TOKENS:,}")
print(f"Maximum potential total: {raw_tokens + MAX_OUTPUT_TOKENS:,}")

if raw_tokens + MAX_OUTPUT_TOKENS >= 1_048_576:
    raise RuntimeError(
        "Prompt + output allowance reaches/exceeds configured 1,048,576-token context limit."
    )

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": user_text}],
    "temperature": 0,
    "top_p": 1,
    "seed": 0,
    "max_tokens": MAX_OUTPUT_TOKENS,
}

print()
print(f"Sending ~950K source-tracing batch {BATCH} ...")
t0 = time.perf_counter()
r = requests.post(URL, json=payload, timeout=1800)
elapsed = time.perf_counter() - t0
r.raise_for_status()
obj = r.json()
choice = obj["choices"][0]
message = choice["message"]
content = message.get("content") or ""
reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
finish_reason = choice.get("finish_reason")
usage = obj.get("usage", {})

RAW_FILE.write_text(content, encoding="utf-8")

parsed = None
try:
    parsed = json.loads(content)
except Exception:
    begin = content.find("{")
    end = content.rfind("}")
    if begin >= 0 and end > begin:
        try:
            parsed = json.loads(content[begin:end + 1])
        except Exception:
            pass

manifest_paths = {x["path"] for x in manifest["files"]}
answered = 0
valid_source_answers = 0
all_source_paths_valid = 0
unknown_paths = []
per_question = []

if isinstance(parsed, dict):
    for item in selected:
        qid = item["id"]
        answer = parsed.get(qid)
        answer_present = False
        valid_any = False
        valid_all = False
        sources = []

        if isinstance(answer, dict):
            answer_text = str(answer.get("answer", "")).strip()
            answer_present = bool(answer_text)
            if answer_present:
                answered += 1
            sources = answer.get("sources", [])
            if isinstance(sources, list) and sources:
                valid_flags = []
                for source in sources:
                    is_valid = source in manifest_paths
                    valid_flags.append(is_valid)
                    if not is_valid:
                        unknown_paths.append((qid, source))
                valid_any = any(valid_flags)
                valid_all = all(valid_flags)
                if valid_any:
                    valid_source_answers += 1
                if valid_all:
                    all_source_paths_valid += 1

        per_question.append({
            "id": qid,
            "answered": answer_present,
            "has_valid_source": valid_any,
            "all_cited_sources_valid": valid_all,
            "sources": sources,
        })

result = {
    "batch": BATCH,
    "batch_range": BATCHES[BATCH],
    "corpus_sha256": sha,
    "corpus_tokens": manifest["actual_tokens"],
    "questions_selected": selected,
    "questions_skipped": skipped,
    "elapsed_sec": elapsed,
    "finish_reason": finish_reason,
    "usage": usage,
    "reasoning_chars": len(reasoning),
    "answered_questions": answered,
    "questions_with_valid_manifest_source": valid_source_answers,
    "questions_with_all_cited_sources_valid": all_source_paths_valid,
    "unknown_source_paths": unknown_paths,
    "per_question": per_question,
    "response": parsed if parsed is not None else content,
}

RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")

print()
print("=" * 88)
print(f"GLM-5.3 SOURCE-TRACE BATCH {BATCH} RESULT")
print("=" * 88)
print(f"Questions selected:             {len(selected)}")
print(f"Questions answered:             {answered}")
print(f"With >=1 valid source path:     {valid_source_answers}")
print(f"All cited paths valid:          {all_source_paths_valid}")
print(f"Elapsed:                        {elapsed:.2f} sec")
print(f"API prompt tokens:              {usage.get('prompt_tokens')}")
print(f"Output tokens:                  {usage.get('completion_tokens')}")
print(f"Reasoning chars:                {len(reasoning)}")
print(f"Finish reason:                  {finish_reason}")

if unknown_paths:
    print()
    print("Cited paths absent from corpus manifest:")
    for qid, source in unknown_paths:
        print(f"  {qid}: {source}")

print()
print(f"Result JSON: {RESULT_FILE}")
print(f"Raw output:  {RAW_FILE}")
print()
print("=" * 88)
print("MODEL ANSWERS")
print("=" * 88)

if isinstance(parsed, dict):
    for item in selected:
        qid = item["id"]
        print()
        print("-" * 88)
        print(f"{qid}: {item['q']}")
        print("-" * 88)
        ans = parsed.get(qid)
        if not isinstance(ans, dict):
            print("MISSING / UNPARSEABLE")
            continue
        print(ans.get("answer", ""))
        print()
        print("Sources:", ", ".join(ans.get("sources", [])))
else:
    print(content)
```

</details>

<details>
<summary><strong>Final Test 5 setup and run process</strong></summary>

The final procedure below is the reproducible configuration used for the authoritative result. It incorporates the later batching and Q10 output-budget adjustment directly; there are no intermediate patch steps to apply.

### One-command source extraction

```bash
bash -lc '
set -euo pipefail

ROOT="$HOME/glm53-source-bench"
EXTRACT="$ROOT/extracted"
IMG="infernix/vllm@sha256:187432cdb974645089bdcb9eb526a5e7b99ff00f25493698f037f24722884d4c"

mkdir -p "$EXTRACT"
rm -rf "$EXTRACT"/*

cid=$(sudo docker create "$IMG")
trap '\''sudo docker rm -f "$cid" >/dev/null 2>&1 || true'\'' EXIT

sudo docker cp "$cid":/opt/glm53-flash/vllm "$EXTRACT/vllm"
sudo docker cp "$cid":/opt/glm53-flash/b12x "$EXTRACT/b12x"
sudo docker cp "$cid":/usr/local/bin "$EXTRACT/usr-local-bin"
sudo docker cp "$cid":/usr/local/libexec "$EXTRACT/usr-local-libexec" 2>/dev/null || true

sudo docker rm "$cid"
trap - EXIT

sudo chown -R "$USER:$USER" "$ROOT"

du -sh "$EXTRACT"/*
find "$EXTRACT" -type f | wc -l

sudo docker image inspect "$IMG" \
  --format '\''ID={{.Id}} RepoDigests={{json .RepoDigests}}'\''
'
```

### Build the exact 950K corpus

```bash
source ~/glm53-kv-test-venv/bin/activate
python3 ~/build-glm53-source-corpus.py
```

Expected identity:

```text
Tokens:   950,000
Files:    115
SHA256:   4f825adda226bbf18d6c52dc73b7421511a995f030a2fb1b38d47b8ea81ef774
Required anchors present in final corpus: 24/24
```

### Final batch map

```text
Batch 1:  Q01-Q04
Batch 2:  Q05-Q08
Batch 4:  Q13-Q16
Batch 5:  Q17-Q20
Batch 6:  Q21-Q23
Batch 8:  Q09
Batch 9:  Q10
Batch 10: Q11
Batch 11: Q12
Batch 12: Q24
Batch 13: Q25
```

The script uses a 64,000-token completion allowance by default and 90,000 only for Batch 9 / Q10:

```python
DEFAULT_MAX_OUTPUT_TOKENS = 64000
Q10_MAX_OUTPUT_TOKENS = 90000

MAX_OUTPUT_TOKENS = (
    Q10_MAX_OUTPUT_TOKENS
    if BATCH == 9
    else DEFAULT_MAX_OUTPUT_TOKENS
)
```

### Run the entire authoritative suite in one loop

```bash
source ~/glm53-kv-test-venv/bin/activate

for b in 1 2 4 5 6 8 9 10 11 12 13; do
  echo
  echo "================ SOURCE TRACE BATCH $b ================"
  python3 ~/glm53-source-trace-batched.py --batch "$b" || break
done
```

### Run any batch individually

Replace `9` with any batch number from the map above:

```bash
source ~/glm53-kv-test-venv/bin/activate
python3 ~/glm53-source-trace-batched.py --batch 9
```

For Q10 / Batch 9 the observed token envelope was:

```text
Local raw prompt tokens: 950,225
Output allowance:         90,000
Maximum potential total: 1,040,225
Configured context limit: 1,048,576
```

That preserves the identical 950K source corpus while leaving approximately 8.3K tokens of context headroom.

</details>

<details>
<summary><strong>Full authoritative Test 5 result output</strong></summary>

Earlier truncated attempts are intentionally excluded. The entries below are only from naturally completed (`finish_reason=stop`) authoritative runs. Together they cover Q01-Q25.

## Batch 1 - Q01-Q04

```text
GLM-5.3 SOURCE-TRACE BATCH 1 RESULT
Questions selected:             4
Questions answered:             4
With >=1 valid source path:     4
All cited paths valid:          4
Elapsed:                        112.98 sec
API prompt tokens:              950362
Output tokens:                  20736
Reasoning chars:                55770
Finish reason:                  stop
```

### Q01

The GLM model in this corpus is Glm5NextForCausalLM / Glm5NextForConditionalGeneration in vllm/vllm/models/glm5next/nvidia/model.py. Configuration precedes construction: VllmConfig.__post_init__ (vllm/vllm/config/vllm.py) calls apply_glm53_tp3_target_geometry(self.model_config, self.parallel_config) from vllm/vllm/transformers_utils/configs/glm53_tp3.py before any other validation, materializing the TP3 physical geometry (num_attention_heads 64->72, linear_num_heads 64->66, moe_intermediate_size 2048, glm53_tp3_shared_expert_intermediate_size 2112, vocab padding 192) when TP=3 on a GLM-5.3 config. Entry points: (1) Glm5NextForCausalLM.forward just calls self.model(...), i.e. Glm5NextModel; (2) the multimodal wrapper Glm5NextForConditionalGeneration (registered with Glm5NextMultiModalProcessor) builds self.visual = Glm5NextVisionTransformer and self.language_model = init_vllm_registered_model(..., architectures=['Glm5NextForCausalLM']) and its forward delegates to self.language_model(...). Glm5NextModel.forward: on the first PP rank it takes inputs_embeds or embed_input_ids(input_ids) -> self.embed_tokens(input_ids) (VocabParallelEmbedding, with padding_size=config.glm53_tp3_vocab_padding_size when glm53_tp3_padding); on later ranks it reads intermediate_tensors['hidden_states']/['residual'] and resets post/comb to None (deferred mHC state is not propagated across PP). It records full_num_tokens = positions.shape[0], applies sp_shard when is_sequence_parallel, captures the first aux hidden state if start_layer is in aux_hidden_state_layers, then iterates the fixed slice self._active_layers = self.layers[start_layer:end_layer]: hidden_states, residual, post, comb = layer(positions, hidden_states, residual, post, comb). Each element is a Glm5NextDecoderLayer. After a layer whose index+1 is an aux layer, _prepare_aux_hidden_state applies layer.hc_post (when mHC) and then hc_contract(aux, layer.n) for dflash_capture or flatten(1). Before any return, _l2pf.join_all() rejoins the L2-prefetch side stream (CUDA-graph capture safety). Non-last PP ranks return IntermediateTensors({'hidden_states','residual'}) (the class declares supports_pp: Literal[False]); the last rank all-gathers under SP, applies self.norm(hidden_states), and returns hidden_states optionally paired with aux_hidden_states. Glm5NextDecoderLayer.forward has two regimes: (a) non-mHC or MTP layers - residual=hidden_states, input_layernorm, self.self_attn(hidden_states=..., positions=...), optional output_indices index_select (MTP-only selective outputs), post_attention_layernorm(attn_output, residual), self.mlp(...); MTP layers return the unsummed (hidden_states, residual) pair so shared_head.norm can fuse add+norm; (b) mHC layers - hc_pre/hc_fused_post_pre (B12xMHCResidual.run_pre/run_post_pre when available, else MHCPreOp/MHCFusedPostPreOp), sp_all_gather before attention, self.self_attn, sp_reduce_scatter, fused post-attn hc_post + pre-FFN hc_pre, MLP (Glm5NextMoE with already_sequence_parallel), and the last mHC layer finishes with hc_post + hc_contract. Logits come from Glm5NextForCausalLM.compute_logits -> self.logits_processor(self.lm_head, hidden_states). The MTP draft path (Glm5NextMultiTokenPredictorLayer.forward) reuses the same layer contract via self.mtp_block(positions=..., hidden_states=..., residual=None, output_indices=...) followed by shared_head.norm(hidden_states, residual=residual).

Sources: vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/config/vllm.py, vllm/vllm/transformers_utils/configs/glm53_tp3.py

### Q02

Construction lives in Glm5NextModel.__init__ and Glm5NextDecoderLayer.__init__ (vllm/vllm/models/glm5next/nvidia/model.py). Glm5NextModel.__init__ sets self.is_v32 = getattr(config, 'index_topk', None) is not None; when true it allocates the shared topk_indices_buffer ([max_num_batched_tokens, topk_tokens + (kpool-1 if kpool>1 else 0)], int32) and pool_topk_indices_buffer ([max_num_batched_tokens, topk_tokens // kpool]); otherwise no topk buffer. It builds embed_tokens with padding_size=config.glm53_tp3_vocab_padding_size when glm53_tp3_padding, then calls make_layers(config.num_hidden_layers, get_layer, prefix=f'{prefix}.layers') where get_layer parses layer_idx = int(prefix.rsplit('.', 1)[1]) and constructs Glm5NextDecoderLayer(vllm_config, config, layer_idx, prefix, topk_indices_buffer, pool_topk_indices_buffer). The active slice is cached as self._active_layers, and the L2-prefetch chain links each layer to its successor via object.__setattr__(layer, '_l2pf_next', active[(i+1) % len(active)]). Finally norm = RMSNorm(hidden_size, rms_norm_eps) on the last PP rank and an assert that num_attention_heads % world_size == 0. Glm5NextDecoderLayer.__init__ selects the attention kind with self.layer_kind = 'kda' if config.is_kda_layer(layer_idx) else 'mla'. KDA layers instantiate Glm5NextLinearAttention (imported 'from .kda import Glm5NextLinearAttention', i.e. vllm/vllm/models/glm5next/nvidia/kda.py - that file is not in the corpus). MLA layers instantiate Glm5NextMLAAttention (from .attention) with mla_quant_config set only when quant_config.get_name() == 'modelopt_mixed' ('Mixed ModelOpt checkpoints describe each projection independently; unlisted projections remain BF16'), plus topk_indices_buffer/pool_topk_indices_buffer, skip_rope=getattr(config, 'mla_nope', False) and is_mtp_layer. FFN selection: mlp_layer_types = config.mlp_layer_types; mlp_type = mlp_layer_types[layer_idx] when in range, else the last entry (MTP layers sit past num_hidden_layers and default to the last base layer's type, sparse for these checkpoints). If self.is_moe and num_experts and mlp_type == 'sparse', the layer builds Glm5NextMoE: GateLinear router with router_dtype = _get_moe_router_dtype(config) (imported from deepseek_v2; fp32 for glm_moe_dsa), e_score_correction_bias for topk_method == 'noaux_tc', EPLB fields, and FusedMoEFactory(use_grouped_topk=True, num_expert_group=getattr(config,'n_group',1), topk_group=getattr(config,'topk_group',1), routed_scaling_factor, swiglu_limit, router_logits_dtype=gate.out_dtype). Shared experts are a Glm5NextMLP whose intermediate_size = getattr(config, 'glm53_tp3_shared_expert_intermediate_size', checkpoint_intermediate_size) with loaded_intermediate_size=checkpoint_intermediate_size (TP3 pads 2048->2112). Otherwise a dense Glm5NextMLP with swiglu_limit selecting SiluAndMulWithClamp vs SiluAndMul. Both norms are RMSNorm(config.hidden_size, eps=config.rms_norm_eps). mHC components are created only when config.mhc and not is_mtp_layer: mhc_num_residual_streams/mhc_tau/hc_eps/mhc_sinkhorn_iterations/mhc_post_mult_value, hc_attn_fn/hc_attn_base/hc_attn_scale and hc_ffn_fn/hc_ffn_base/hc_ffn_scale parameters, MHCPreOp/MHCPostOp/MHCFusedPostPreOp, and - on CUDA SM120-family when get_b12x_mhc() (vllm/utils/b12x) reports support - a B12xMHCResidual(hidden_size, hc_mult, rms_eps, hc_eps, sinkhorn_iters) from vllm.models.deepseek_v4.nvidia.b12x. Under sequence parallel (parallel_config.use_sequence_parallel_moe) on non-MTP layers, self.self_attn.o_proj.reduce_results = False so the decoder-layer reduce_scatter completes the partial sum (DSv4 pattern). L2 prefetch state (_l2pf_next, _l2pf_ready, _l2pf_plan_b/_l2pf_plan_c) is initialized for the lazy _l2pf_build_plans. The linear-attention/GDN path is the KDA branch above; its shared machinery (KimiGatedDeltaNetAttention, PluggableLayer 'kimi_gated_delta_net_attention') lives in vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py and is reached through the '.in_proj_qkvgfab'/'.conv1d' weight mappings ('KDA: reuse the shared Kimi projected-GDN layer') in Glm5NextModel.load_weights.

Sources: vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/transformers_utils/configs/glm53_tp3.py, vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py

### Q03

Selection: Glm5NextDecoderLayer.__init__ (vllm/vllm/models/glm5next/nvidia/model.py) sets self.layer_kind = 'kda' if config.is_kda_layer(layer_idx) else 'mla' and, for KDA layers, builds self.self_attn = Glm5NextLinearAttention(config=config, vllm_config=vllm_config, prefix=f'{prefix}.self_attn'), imported at module top as 'from .kda import Glm5NextLinearAttention' (vllm/vllm/models/glm5next/nvidia/kda.py). That wrapper file is not part of the corpus, so its internal composition is INSUFFICIENT CORPUS EVIDENCE; what the corpus establishes is that the GLM5Next KDA layer reuses the shared Kimi projected-GDN machinery: Glm5NextModel.load_weights maps checkpoint '.q_proj'/'.k_proj'/'.v_proj'/'.b_proj'/'.f_a_proj' into '.in_proj_qkvgfab' shards 0-4 and '.q_conv1d'/'.k_conv1d'/'.v_conv1d' into '.conv1d' shards 0-2 under the comment 'KDA: reuse the shared Kimi projected-GDN layer', with _load_glm5next_fused_conv1d feeding each conv shard through the parameter's weight_loader. The shared implementation is KimiGatedDeltaNetAttention (PluggableLayer registered as 'kimi_gated_delta_net_attention') in vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py. Its forward() projects through in_proj_qkvgfab, splits mixed_qkv / g_proj_states / f_a / beta (full-rank gate) or mixed_qkv/beta/f_a plus g_b_proj(g_a_proj(...)), computes g1 = f_b_proj(f_a) rearranged to (1, n, h, d) and g2 = rearrange(g_proj_states), allocates core_attn_out, calls the eager-break _forward(mixed_qkv, g1, g2, beta, core_attn_out), then o_proj. _forward reads GDNAttentionMetadata from get_forward_context().attn_metadata[self.prefix], splits spec-decode multi-query rows from non-spec rows, runs the short convolution with causal_conv1d_update / causal_conv1d_fn (vllm/vllm/model_executor/layers/mamba/ops/causal_conv1d.py), and dispatches the KDA recurrence to: (1) b12x decode - _can_use_b12x_kda_decode(m) -> _run_b12x_kda_decode_post_conv, which binds api.bind_kda(plan, scratch=..., mixed_qkv=..., raw_g=..., raw_beta=..., z=..., A_log, dt_bias, norm_weight, recurrent_state, query_start_loc, num_accepted_tokens, state_indices, num_seqs, num_tokens, output) and runs api.run_kda(binding, lower_bound=self.gate_lower_bound, eps=self.o_norm.eps, scale=self.head_dim**-0.5) using get_b12x_gdn_decode() from vllm.utils.b12x; (2) prefill backends resolved by resolve_kda_prefill_backend - 'b12x' via _run_b12x_kda_prefill/get_b12x_kda_prefill, 'flashkda' via _flashkda_prefill (vllm._flashkda_C), or 'triton' via chunk_kda_with_fused_gate; (3) decode fallback fused_recurrent_kda / fused_recurrent_kda_packed_decode. The vendor kernels are imported per platform from vllm.models.kimi_k3.nvidia.ops.third_party.kda (CUDA) or vllm.models.kimi_k3.amd.ops.third_party.kda (ROCm). The Kimi-K3 ROCm layer KimiK3DeltaAttention in vllm/vllm/models/kimi_k3/amd/kda.py is the sibling that owns those third-party kernels and adds the fused ops.fused_kda_decode decode path. State plumbing is shared too: Glm5NextForCausalLM.get_mamba_state_shape_from_config / get_mamba_state_dtype_from_config / get_mamba_state_copy_func call MambaStateShapeCalculator.kda_state_shape(tp_size, hf_config.linear_num_heads, hf_config.linear_head_dim, conv_kernel_size=hf_config.linear_conv_kernel_dim, num_spec=num_spec), MambaStateDtypeCalculator.kda_state_dtype and MambaStateCopyFuncCalculator.kda_state_copy_func, and the layer's own get_state_shape/get_state_dtype delegate to the same calculators.

Sources: vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py, vllm/vllm/models/kimi_k3/amd/kda.py

### Q04

Launcher/policy layer: usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh (and its identical R17 copy usr-local-libexec/serve-glm53-flash-tp3-r17-dense.sh) reads kda_decode_backend=${GLM53_KDA_DECODE_BACKEND:-auto}, validates it with 'case ... in auto | b12x | triton' (else 'GLM53_KDA_DECODE_BACKEND must be auto, b12x, or triton', exit 2), and injects it into the engine as --additional-config '{"glm53_kda_decode_backend":"${kda_decode_backend}","kda_prefill_backend":"${kda_prefill_backend}"}'. The R17 TP3 dispatcher usr-local-bin/serve-glm53-flash-tp3-r17.sh locks the value with lock_env GLM53_KDA_DECODE_BACKEND b12x (plus lock_env GLM53_KDA_PREFILL_BACKEND flashkda and lock_env GLM53_R17_REQUIRE_RUNTIME_PROOF 1) before exec'ing the capture launcher, so the R17 TP3 chain always reaches the engine with glm53_kda_decode_backend='b12x'. The JSON lands in VllmConfig.additional_config (field declared in vllm/vllm/config/vllm.py). Note vllm/vllm/engine/arg_utils.py only exposes a --kda-prefill-backend CLI (choices auto/triton/flashkda/b12x, written into additional_config['kda_prefill_backend']); there is no decode CLI, so the decode backend is additional-config-only. Runtime selection: the decode machinery is in KimiGatedDeltaNetAttention (vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py). Decision points: (1) the class flag enable_b12x_kda_decode (False on the shared Kimi GDN layer) plus b12x_kda_null_state_index gate _initialize_b12x_kda_decode, which additionally requires gate_lower_bound is not None, head_dim == 128, model dtype bfloat16, recurrent state dtype in (bfloat16, float32), current_platform.is_cuda(), and get_b12x_gdn_decode() exposing bind_kda/run_kda with api.is_supported(device); it sizes max_tokens = max_num_seqs * (num_spec + 1) and refuses state_index_columns > 8. (2) bind_kv_cache builds the plan via _make_b12x_kda_plan(max_state_slots=recurrent_state.shape[0]) with api.Caps(..., gate_activation='sigmoid', qk_l2norm=True, null_state_index=self.b12x_kda_null_state_index, kda_metadata_validation='trusted') and takes the scratch from get_b12x_scratch_buffers(plan). (3) Per batch, _can_use_b12x_kda_decode(m) requires the plan/scratch, num_prefills == 0, and either a spec-decode-only batch (spec_state_indices_tensor, spec_query_start_loc, num_accepted_tokens) or a plain decode batch; when true, _run_b12x_kda_decode_post_conv stages shared metadata tensors in forward-context 'b12x_kda_metadata_tensors' and calls api.run_kda(binding, lower_bound=self.gate_lower_bound, eps=self.o_norm.eps, scale=self.head_dim**-0.5). Otherwise the fallback fused_recurrent_kda / fused_recurrent_kda_packed_decode from vllm.models.kimi_k3.{nvidia,amd}.ops.third_party.kda runs. The exact site that reads additional_config['glm53_kda_decode_backend'] and flips enable_b12x_kda_decode for the GLM-5.3 layer would be vllm/vllm/models/glm5next/nvidia/kda.py, which is not in the corpus - INSUFFICIENT CORPUS EVIDENCE for that read; the corpus only shows the shared layer's default False and the launcher's additional-config emission. The sibling prefill decision is resolve_kda_prefill_backend(backend, head_dim, dtype, lower_bound, state_dtype): 'b12x' requires is_b12x_kda_prefill_supported (b12x package, CUDA, head_dim 128, bf16, float32 state, bounded gate); 'flashkda' requires is_flashkda_supported (CUDA SM90/SM10x/SM12x, bf16, head_dim 128, bounded gate); 'auto' never selects b12x and picks flashkda when supported, else triton. Finally the R17 receipt: log_glm53_r17_tp3_runtime_proof in vllm/vllm/v1/worker/utils.py, triggered by GLM53_R17_REQUIRE_RUNTIME_PROOF=1 with TP=3 and a GLM-5.3 config, computes kda_decode_backend = 'b12x' only when every Glm5NextLinearAttention module has _b12x_kda_api set (else 'fallback') and kda_prefill_backend from the layers' kda_prefill_backend attribute (must be exactly {'flashkda'}), and raises RuntimeError('GLM-5.3 R17 TP3 runtime proof failed: ...') unless the proof equals the expected {'collective_backend': 'b12x_pcie_oneshot', 'expert_parallel_size': 3, 'kda_decode_backend': 'b12x', 'kda_prefill_backend': 'flashkda', 'mm_encoder_tp_mode': 'weights'}. Net result: R17 TP3 runs KDA decode on the b12x backend with flashkda prefill; the generic launcher's 'auto' leaves decode to the layer-level b12x enablement and prefill to flashkda-else-triton.

Sources: usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, usr-local-libexec/serve-glm53-flash-tp3-r17-dense.sh, usr-local-bin/serve-glm53-flash-tp3-r17.sh, vllm/vllm/config/vllm.py, vllm/vllm/engine/arg_utils.py, vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py, vllm/vllm/v1/worker/utils.py

## Batch 2 - Q05-Q08

```text
GLM-5.3 SOURCE-TRACE BATCH 2 RESULT
Questions selected:             4
Questions answered:             4
With >=1 valid source path:     4
All cited paths valid:          4
Elapsed:                        259.19 sec
API prompt tokens:              950350
Output tokens:                  47065
Reasoning chars:                118055
Finish reason:                  stop
```

### Q05

CONFIG/LAUNCHER LAYER: usr-local-bin/serve-glm53-flash-tp3-r17.sh locks the prefill backend with `lock_env GLM53_KDA_PREFILL_BACKEND flashkada` (and separately `lock_env GLM53_KDA_DECODE_BACKEND b12x`). The libexec launchers (usr-local-libexec/serve-glm53-flash-tp3-r17-dense.sh and the identical serve-glm53-flash-nvfp4-dflash2.sh) read `kda_prefill_backend=${GLM53_KDA_PREFILL_BACKEND:-flashkada}`, validate it against `auto | flashkda | triton` only (b12x is deliberately NOT an accepted prefill value here), and embed both values into `--additional-config '{"glm53_kda_decode_backend":"...","kda_prefill_backend":"..."}'`. vllm/vllm/engine/arg_utils.py also exposes a CLI path: `--kda-prefill-backend` (choices auto/triton/flashkda/b12x) whose value `create_engine_config` writes into `additional_config["kda_prefill_backend"]`. RUNTIME SELECTION (PREFILL): in vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py, `KimiGatedDeltaNetAttention.__init__` reads `additional_config.get("kda_prefill_backend", "auto")` and calls `resolve_kda_prefill_backend(backend, head_dim, dtype, gate_lower_bound, state_dtype)`. That resolver's docstring states '``auto`` never selects ``b12x``; that backend must be requested by name until its serving qualification lands': b12x requires `is_b12x_kda_prefill_supported` (b12x package via `get_b12x_kda_prefill()`, CUDA, head_dim=128, bf16, float32 recurrent state, bounded gate, `api.is_supported(device)`) else RuntimeError; flashkda requires `is_flashkda_supported` (CUDA capability major 9/10/12, head_dim=128, bf16, bounded gate) else RuntimeError; otherwise flashkda when supported and not explicitly triton, else triton. The resolved value is stored as `self.kda_prefill_backend` and also shapes the cache contract: `get_kv_cache_spec` sets `num_prefill_checkpoint_blocks=int(self.kda_prefill_backend in ("flashkda", "b12x"))`, enabling mid-sequence recurrent/conv checkpoints. PREFILL EXECUTION (in `_forward`, non-spec path): (1) b12x -> `_run_b12x_kda_prefill`, which binds `api.bind(plan, scratch=..., q/k/v/raw_g/raw_beta, A_log, dt_bias, recurrent_state, cu_seqlens, initial_state_indices, final_state_indices=state_indices, checkpoint_state_indices, checkpoint_offsets, num_seqs, num_tokens, output=...)` and runs `api.run(binding, lower_bound=..., max_live_tokens=..., max_live_seqs=...)` straight against the recurrent-state pool by slot index, with the conv checkpoint stored by `_store_kda_conv_checkpoint`/`_store_cache_checkpoints_kernel`; (2) flashkda -> `_flashkda_prefill` calling `torch.ops._flashkda_C.fwd(...)` with workspace/final-state/checkpoint buffers borrowed from `current_workspace_manager().get_simultaneous(*self._flashkda_buffer_specs)`; (3) triton -> `chunk_kda_with_fused_gate(...)` from `vllm.models.kimi_k3.nvidia.ops.third_party.kda` (amd copy on ROCm). DECODE SELECTION DIFFERS: the decode backend is a separate config key (`GLM53_KDA_DECODE_BACKEND` -> `glm53_kda_decode_backend` additional-config key; no `--kda-*` CLI exists for it in the corpus — arg_utils' `--gdn-decode-kernel` is the Qwen GDN analog) and is dispatched dynamically per batch, not resolved once at init: `_can_use_b12x_kda_decode(m)` requires the b12x plan/scratch bound, `num_prefills == 0`, a spec-only or plain-decode batch, and state-index width <= `_b12x_kda_state_index_columns`; when eligible, `_run_b12x_kda_decode_post_conv` runs `api.bind_kda(...)` + `api.run_kda(binding, lower_bound=..., eps=..., scale=...)` (with `retain_cuda_graph_capture_resource(binding)` for capture safety and shared metadata tensors cached under forward-context key `b12x_kda_metadata_tensors`); the fallback is the vendor `fused_recurrent_kda` / `fused_recurrent_kda_packed_decode` after `causal_conv1d_update`. The b12x decode init is gated on the class flag `enable_b12x_kda_decode` (False on the shared Kimi-Linear layer); the GLM-5.3 `Glm5NextLinearAttention` subclass that consumes `glm53_kda_decode_backend` lives in vllm/vllm/models/glm5next/nvidia/kda.py, which is NOT in the corpus — INSUFFICIENT CORPUS EVIDENCE for that file's exact code, though the runtime proof establishes the outcome. VERIFICATION: vllm/vllm/v1/worker/utils.py `log_glm53_r17_tp3_runtime_proof` (triggered by `GLM53_R17_REQUIRE_RUNTIME_PROOF=1`, called from both model runners' `load_model`) requires `kda_prefill_backend` to be a single value in {"flashkda","triton"} (expected `flashkda`) and `kda_decode_backend == "b12x"`, the latter verified by `getattr(layer, "_b12x_kda_api", None) is not None` on every `Glm5NextLinearAttention` module; mismatch raises `RuntimeError("GLM-5.3 R17 TP3 runtime proof failed: ...")`. On ROCm, vllm/vllm/models/kimi_k3/amd/kda.py asserts `backend == "triton"` ('The ROCm Kimi-K3 KDA layer only supports the Triton KDA prefill backend'). SUMMARY OF THE DIFFERENCE: prefill = one static per-layer resolution at model init that also affects the KV-cache spec and checkpointing; decode = per-batch metadata-driven dispatch between the b12x fused decode op and vendor recurrent kernels.

Sources: usr-local-bin/serve-glm53-flash-tp3-r17.sh, usr-local-libexec/serve-glm53-flash-tp3-r17-dense.sh, usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, vllm/vllm/engine/arg_utils.py, vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py, vllm/vllm/v1/worker/utils.py, vllm/vllm/models/kimi_k3/amd/kda.py

### Q06

ATTENTION: (a) `--attention-backend B12X` on every GLM-5.3-Flash launcher; the R17 TP3 launcher locks `ATTENTION_BACKEND B12X` and `MTP_ATTENTION_BACKEND B12X` for the draft. (b) Backend implementation in vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py: `B12xMLASparseBackend`, `B12xMLASparseImpl`, `B12xGLM5NextMLASparseBackend`, and `B12xGLMDSAMLASparseBackend`, with B12X planning/binding/run APIs and packed fp8_ds_mla/nvfp4_ds_mla caches. (c) DCP full-CKV gather and related B12X gather controls. (d) Fused MLA query GEMM via `can_implement_bf16_mla_query` / `prewarm_bf16_mla_query` / `run_bf16_mla_query`. LINEAR: `--linear-backend b12x`, `VLLM_USE_B12X_FP8_GEMM`, dense split-K/tiny-decode knobs, and b12x integration plugin paths. MoE: `--moe-backend b12x` / `VLLM_USE_B12X_MOE`, `VLLM_B12X_MOE_FP4_FORCE_A16`, A8/A16 activation controls, plus the GLM-5.3 distinction where target NVFP4 experts use B12X while MTP MXFP8 uses Humming. KDA/GDN: R17 locks `GLM53_KDA_DECODE_BACKEND=b12x`; shared GDN/KDA code contains B12X decode and optional B12X prefill paths; Qwen GDN has analogous B12X decode support. COLLECTIVE/ALL-REDUCE: `B12X_PCIE_ALLREDUCE=1`, `VLLM_ENABLE_PCIE_ALLREDUCE=1`, `VLLM_PCIE_ALLREDUCE_BACKEND=b12x`, size thresholds, `B12xPcieAllReduce`, runtime-proof classification, health checks, and fused allreduce+RMSNorm gating. MHC/HYPER-CONNECTIONS: `get_b12x_mhc()` -> `B12xMHCResidual` in Glm5Next and DeepSeek-V4. SPARSE INDEXER/MISC: sparse-indexer toggles, B12X warmup units, compile caches, DCP A2A, WO projection, and PCIe calibration fingerprints.

Sources: usr-local-libexec/serve-glm53-flash-tp3-r17-dense.sh, usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, usr-local-bin/serve-glm53-flash-nvfp4.sh, usr-local-bin/serve-glm52-v16.sh, usr-local-bin/serve-glm51.sh, usr-local-bin/serve-ds4-flash.sh, usr-local-bin/serve-ds4-pro-tp10.sh, usr-local-bin/serve-kimi-k3-qsrt.sh, vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py, vllm/vllm/model_executor/layers/attention/mla_attention.py, vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py, vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py, vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/models/deepseek_v4/nvidia/model.py, vllm/vllm/v1/worker/utils.py, vllm/vllm/v1/worker/gpu_worker.py, vllm/vllm/envs.py, vllm/vllm/config/vllm.py, usr-local-bin/glm52-pcie-runtime-env.sh, usr-local-bin/glm52-dcp-prefill-policy.sh, b12x/b12x/integration/vllm/plugin.py, b12x/b12x/tools/inspect_model_policy.py

### Q07

1) LAUNCHER POLICY. usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh reads `b12x_pcie_allreduce=${B12X_PCIE_ALLREDUCE:-1}`, validates 0|1, then exports `VLLM_ENABLE_PCIE_ALLREDUCE` and `VLLM_PCIE_ALLREDUCE_BACKEND=b12x`; when 0 it appends `--disable-custom-all-reduce`. usr-local-bin/serve-glm53-flash-tp3-r17.sh locks `B12X_PCIE_ALLREDUCE=1`, `VLLM_ENABLE_PCIE_ALLREDUCE=1`, `VLLM_PCIE_ALLREDUCE_BACKEND=b12x`, `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=84KB`, `B12X_PCIE_ALLREDUCE_ALGORITHM=auto`. Other launchers show B12X vs NCCL/vLLM-custom modes and multi-node RDMA disabling the PCIe path. 2) ENV DEFINITIONS in vllm/vllm/envs.py expose enable/backend/one-shot-size/DMA/RoCE controls. 3) RUNTIME SELECTION: `init_worker_distributed_environment` wires custom-all-reduce enablement; the TP communicator carries `b12x_ar_comm`, an instance of `B12xPcieAllReduce`. `log_glm53_r17_tp3_runtime_proof` classifies the live collective as `b12x_pcie_oneshot` iff the communicator is an active `B12xPcieAllReduce`, not disabled, world_size==3, `_runtime` is non-None, and `allreduce_max_bytes > 0`; otherwise it is `fallback` and the R17 proof fails. Health checks run after model steps. Fusion: with `VLLM_ENABLE_PCIE_ALLREDUCE=1`, `enable_allreduce_rms_fusion` requires backend b12x. 4) CONDITIONS: PCIe all-reduce enabled, backend b12x, TP>1 on the supported topology, custom all-reduce not disabled, and for the R17 proof exactly world size 3 plus an initialized runtime and positive one-shot threshold.

Sources: usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, usr-local-bin/serve-glm53-flash-tp3-r17.sh, usr-local-bin/glm52-pcie-runtime-env.sh, usr-local-bin/serve-glm52-v16.sh, usr-local-bin/serve-glm52-v19.sh, usr-local-bin/serve-ds4-flash.sh, usr-local-bin/serve-ds4-pro-tp10.sh, vllm/scripts/serve-glm53-nvfp4-tp4-rdma.sh, vllm/vllm/envs.py, vllm/vllm/config/vllm.py, vllm/vllm/v1/worker/utils.py, vllm/vllm/v1/worker/gpu_worker.py, vllm/vllm/engine/arg_utils.py, usr-local-bin/glm52-pcie-calibration.py

### Q08

DEFINITION: vllm/vllm/envs.py declares `VLLM_PCIE_ALLREDUCE_BACKEND` with b12x as the default and only allowed value. EFFECT 1: `enable_allreduce_rms_fusion(cfg)` returns false for TP<=1 or non-CUDA; with PCIe all-reduce enabled it returns true only when backend == b12x, making the O2/O3 fused-allreduce-RMSNorm path B12X-gated. EFFECT 2: `log_glm53_r17_tp3_runtime_proof` reads the TP device communicator's `b12x_ar_comm` and reports `b12x_pcie_oneshot` only for an active world-size-3 `B12xPcieAllReduce` with a live runtime and positive max bytes; otherwise `fallback`. EFFECT 3: worker health checking calls the B12X communicator's `check_health`. Visible fallbacks include launcher switches to NCCL, vLLM custom all-reduce, or disabled custom all-reduce; multi-node RDMA launchers disable the PCIe path; large transfers can fall back according to DMA thresholds. The source also exposes one alternative launcher string (`flashinfer-ipc`) that conflicts with the env whitelist if that variable is read.

Sources: vllm/vllm/envs.py, vllm/vllm/config/vllm.py, vllm/vllm/v1/worker/utils.py, vllm/vllm/v1/worker/gpu_worker.py, usr-local-bin/serve-ds4-flash.sh, usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, vllm/vllm/engine/arg_utils.py, vllm/vllm/models/minimax_m3/nvidia/model.py, vllm/vllm/model_executor/layers/linear.py, usr-local-bin/glm52-pcie-runtime-env.sh, vllm/scripts/serve-glm53-nvfp4-tp4-rdma.sh

## Batch 8 - Q09

```text
GLM-5.3 SOURCE-TRACE BATCH 8 RESULT
Questions selected:             1
Questions answered:             1
With >=1 valid source path:     1
All cited paths valid:          1
Elapsed:                        215.07 sec
API prompt tokens:              950238
Output tokens:                  42732
Reasoning chars:                152180
Finish reason:                  stop
```

### Q09

TRACE: launcher/config -> distributed init -> layer construction

1) TP-SIZE ORIGIN (configuration policy, not runtime). Launchers only translate environment variables into the CLI flag: usr-local-bin/serve-glm52-v16.sh builds cmd=(vllm serve ... --tensor-parallel-size "${TP}") with TP="${TP:-8}", and usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh uses tp=${TP:-4} then --tensor-parallel-size "${tp}". In vllm/vllm/engine/arg_utils.py, EngineArgs.tensor_parallel_size defaults to ParallelConfig.tensor_parallel_size, is registered as the CLI flag --tensor-parallel-size / -tp via parallel_group.add_argument, and create_engine_config() constructs ParallelConfig(..., tensor_parallel_size=self.tensor_parallel_size, ...).

2) DISTRIBUTED INITIALIZATION (runtime). In vllm/vllm/v1/worker/gpu_worker.py, Worker.init_device() calls init_worker_distributed_environment(self.vllm_config, self.rank, self.distributed_init_method, self.local_rank, current_platform.dist_backend) before the memory snapshot (comment: 'Initialize the distributed environment BEFORE taking memory snapshot'). That helper calls init_distributed_environment(parallel_config.world_size, rank, init_method, local_rank, backend, timeout) and then ensure_model_parallel_initialized(parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size, parallel_config.prefill_context_parallel_size, parallel_config.decode_context_parallel_size). After this the TP process group exists and the accessor pair get_tensor_model_parallel_world_size() / get_tensor_model_parallel_rank() (imported from vllm.distributed by every model file) returns the group geometry. INSUFFICIENT CORPUS EVIDENCE for the internals of ensure_model_parallel_initialized / get_tensor_model_parallel_world_size and for ParallelConfig.world_size / use_sequence_parallel_moe themselves (vllm/distributed/parallel_state.py and vllm/config/parallel.py are not in the corpus); the call sites and the consumer API are what the corpus establishes.

3) CONSUMPTION AT LAYER CONSTRUCTION. GPUModelRunner.load_model (vllm/vllm/v1/worker/gpu/model_runner.py) -> model_loader.load_model(vllm_config=..., model_config=...) builds the model tree (e.g. Glm5NextForCausalLM -> Glm5NextModel -> Glm5NextDecoderLayer -> Glm5NextMoE / Glm5NextMLP). The canonical sink is LinearBase.__init__ in vllm/vllm/model_executor/layers/linear.py: self.tp_rank = tp_rank if tp_rank is not None else get_tensor_model_parallel_rank(); self.tp_size = tp_size if tp_size is not None else get_tensor_model_parallel_world_size(), with per-partition geometry: ColumnParallelLinear.__init__ sets self.output_size_per_partition = divide(output_size, self.tp_size); RowParallelLinear.__init__ sets self.input_size_per_partition = divide(input_size, self.tp_size); MergedColumnParallelLinear.__init__ asserts all(output_size % self.tp_size == 0 for output_size in output_sizes). Model classes also call get_tensor_model_parallel_world_size() directly: Glm5NextModel.__init__ and KimiLinearModel.__init__ both assert config.num_attention_heads % world_size == 0.

CONCRETE EXAMPLES WHERE TP WORLD SIZE CHANGES TENSOR OR MODULE GEOMETRY:

Example 1 - QKV head sharding and KV-head replication (vllm/vllm/model_executor/layers/linear.py, QKVParallelLinear.__init__): tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1; self.num_heads = divide(self.total_num_heads, tp_size); if tp_size >= self.total_num_kv_heads then self.num_kv_heads = 1 and self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads), else self.num_kv_heads = divide(self.total_num_kv_heads, tp_size) with num_kv_head_replicas = 1; output_sizes = [num_heads*head_size*tp_size, num_kv_heads*head_size*tp_size, num_kv_heads*v_head_size*tp_size]. TP size sets per-rank head counts, decides KV sharding vs replication, and the weight loader slices q with shard_rank = self.tp_rank and k/v with shard_rank = self.tp_rank // self.num_kv_head_replicas.

Example 2 - Block-FP8 MLP intermediate padding (vllm/vllm/models/deepseek_v4/nvidia/model.py, DeepseekV4MLP.__init__): block-FP8 shards round the per-rank block count up so the linear's even TP split stays block-aligned, with trailing ranks zero-filled by load_weights. tp_size = get_tensor_model_parallel_world_size(); n_local = cdiv(intermediate_size // block_size[0], tp_size); intermediate_size = n_local * block_size[0] * tp_size.

Example 3 - MoE local expert partitioning (same file): DeepseekV4MoE.__init__ stores self.tp_size = get_tensor_model_parallel_world_size(); _init_fused_moe_experts computes self.n_local_physical_experts = self.n_physical_experts // self.tp_size, self.experts_start_idx = self.tp_rank * self.n_local_experts, self.experts_end_idx = self.experts_start_idx + self.n_local_experts.

Example 4 - GLM-5.3 TP3 physical geometry rewrite (vllm/vllm/transformers_utils/configs/glm53_tp3.py, apply_glm53_tp3_target_geometry; invoked from VllmConfig.__post_init__ in vllm/vllm/config/vllm.py and from SpeculativeConfig.__post_init__ / _apply_glm53_tp3_draft_geometry in vllm/vllm/config/speculative.py). The function returns early unless parallel_config.tensor_parallel_size == 3, then rewrites the HF config before any layer is built: original_num_attention_heads=64 -> num_attention_heads=72, original_num_key_value_heads=64 -> 72, original_linear_num_heads=64 -> linear_num_heads=66, glm53_tp3_shared_expert_intermediate_size=2112, glm53_tp3_mtp_projection_size=4098, glm53_tp3_vocab_padding_size=192, glm53_tp3_vocab_storage_size=154944. Downstream geometry: Glm5NextMoE reads the padded shared-expert size; embedding/lm-head use vocab padding; MTP eh_proj uses the padded projection size.

Example 5 - Kimi MoE intermediate padding (vllm/vllm/models/kimi_k3/nvidia/model.py, KimiMoE.__init__): self.tp_size = get_tensor_model_parallel_world_size(); if self.tp_size > 1 and not vllm_config.parallel_config.enable_expert_parallel: moe_intermediate_per_partition = moe_intermediate_size // self.tp_size; if below the minimum, self.padded_moe_intermediate_size = min_moe_intermediate_per_partition * self.tp_size.

Example 6 - TP>1 gates sequence parallelism, which changes module geometry. vllm/vllm/models/deepseek_v4/nvidia/model.py _use_sequence_parallel requires parallel_config.tensor_parallel_size > 1; DeepseekV4DecoderLayer then sets self.attn.wo_b.reduce_results = False and DeepseekV4MLP passes disable_tp=is_sequence_parallel so MLP weights are replicated instead of sharded. Glm5Next follows the same pattern for its attention output projection and shared-expert MLP.

Example 7 - Direct tensor-shape dependence in specialized modules: InklingSinkExperts sizes intermediate_pp = d_mlp // tp_size; MiniMaxM3Model derives num_index_heads from sparse heads // tp_size; OAIAttention sizes sinks = num_attention_heads // tp_size; BailingMoeV3KimiDeltaAttention derives local_num_heads and local projection sizes from tp_size.

Example 8 - Weight-loading geometry derived from TP: DeepseekV4Model.load_weights uses n_local_head = n_head // tp_size and head_rank_start = n_local_head * tp_rank for rank-local attn_sink rows; shared-expert weights are padded to TP-aligned multiples.

Example 9 - Replicated-shard sizing helper (vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py, _KimiGDNMergedColumnParallelLinear): output_sizes[replicated_shard_id] *= tp_size so every rank reserves full rows for the replicated shard. KimiGatedDeltaNetAttention validates the GLM-5.3 TP3 logical64/physical66 contract and derives local heads/projection sizes by dividing through self.tp_size.

Example 10 - Config-level validation and kernel gating on TP size: ModelConfig.verify_with_parallel_config rejects attention heads not divisible by tensor_parallel_size; get_num_attention_heads/get_num_kv_heads return per-rank counts; Kimi GEMM+reduce-scatter all-reduce support and fused allreduce-RMSNorm also gate on TP size.

LAUNCHER vs RUNTIME: shell launchers only emit --tensor-parallel-size; group construction, world-size plumbing, and geometry changes are runtime vLLM behavior.

Sources: usr-local-bin/serve-glm52-v16.sh, usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, vllm/vllm/engine/arg_utils.py, vllm/vllm/v1/worker/gpu_worker.py, vllm/vllm/v1/worker/gpu/model_runner.py, vllm/vllm/model_executor/layers/linear.py, vllm/vllm/models/deepseek_v4/nvidia/model.py, vllm/vllm/models/deepseek_v4/nvidia/dspark.py, vllm/vllm/models/deepseek_v4/amd/model.py, vllm/vllm/models/deepseek_v4/xpu/model.py, vllm/vllm/models/kimi_k3/nvidia/model.py, vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/models/glm5next/nvidia/mtp.py, vllm/vllm/transformers_utils/configs/glm53_tp3.py, vllm/vllm/config/vllm.py, vllm/vllm/config/model.py, vllm/vllm/config/speculative.py, vllm/vllm/model_executor/models/transformers/base.py, vllm/vllm/models/inkling/nvidia/moe.py, vllm/vllm/models/minimax_m3/nvidia/model.py, vllm/vllm/model_executor/models/gpt_oss.py, vllm/vllm/model_executor/models/bailing_moe_v3.py, vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py, vllm/vllm/model_executor/models/mllama4.py, vllm/vllm/model_executor/models/glm4_1v.py

## Batch 9 - Q10

```text
GLM-5.3 SOURCE-TRACE BATCH 9 RESULT
Questions selected:             1
Questions answered:             1
With >=1 valid source path:     1
All cited paths valid:          1
Elapsed:                        151.25 sec
API prompt tokens:              950237
Output tokens:                  30145
Reasoning chars:                92669
Finish reason:                  stop
```

### Q10

EXPERT-PARALLEL CONFIGURATION TRACE

(1) Configuration entry points (CLI/config policy). In vllm/vllm/engine/arg_utils.py, EngineArgs declares 'enable_expert_parallel: bool = ParallelConfig.enable_expert_parallel' and registers the CLI flags '--enable-expert-parallel'/'-ep' via parallel_group.add_argument, alongside the related knobs '--enable-ep-weight-filter' (enable_ep_weight_filter), '--expert-placement-strategy' (expert_placement_strategy), '--enable-eplb' and '--eplb-config'. create_engine_config forwards all of them into ParallelConfig(... enable_expert_parallel=self.enable_expert_parallel, enable_ep_weight_filter=..., eplb_config=self.eplb_config, enable_eplb=self.enable_eplb, expert_placement_strategy=...).

(2) Launcher policy. usr-local-bin/serve-glm53-flash-tp3-r17.sh unconditionally appends '--enable-expert-parallel' on its exec line ('exec "${capture_launcher}" "$@" --enable-expert-parallel --mm-encoder-tp-mode weights'), and usr-local-bin/serve-dots3-note-prev.sh passes '--enable-expert-parallel' to vllm serve. EP is mandatory for the GLM-5.3 TP3 contract: vllm/vllm/transformers_utils/configs/glm53_tp3.py::apply_glm53_tp3_target_geometry raises 'GLM-5.3 physical TP3 requires expert parallelism because the released routed-expert width is not divisible by 3.' when 'not parallel_config.enable_expert_parallel', and the runtime receipt log_glm53_r17_tp3_runtime_proof in vllm/vllm/v1/worker/utils.py requires 'expert_parallel_size': 3, computed as 'get_ep_group().world_size if parallel_config.enable_expert_parallel else 1'. For drafts, vllm/vllm/config/speculative.py::_apply_glm53_tp3_draft_geometry sets 'draft.enable_expert_parallel = (target.enable_expert_parallel if self.method == 'mtp' else False)' — the comment states MTP participates in the target MoE's topology while DFlash is dense and must not join the routed-expert EP group.

(3) Where EP rank/world size are obtained. Every MoE module reads the EP process group via get_ep_group() (imported from vllm.distributed): 'self.ep_group = get_ep_group().device_group', 'self.ep_size = self.ep_group.size()' (or '.world_size'), 'self.ep_rank = get_ep_group().rank_in_group'. This exact pattern appears in Glm5NextMoE.__init__ (vllm/vllm/models/glm5next/nvidia/model.py), DeepseekV2MoE.__init__ (vllm/vllm/model_executor/models/deepseek_v2.py), DeepseekV4MoE._init_mega_moe_experts (vllm/vllm/models/deepseek_v4/nvidia/model.py), the MegaMoE branch of KimiMoE.__init__ (vllm/vllm/models/kimi_k3/nvidia/model.py), HYV3MoEFused.__init__ (vllm/vllm/model_executor/models/hy_v3.py), OpenPanguMoE.__init__ (vllm/vllm/model_executor/models/openpangu.py), MiMoV2MoE.__init__ (vllm/vllm/model_executor/models/mimo_v2.py), NemotronHMoE.__init__ (vllm/vllm/model_executor/models/nemotron_h.py), AXK1MoE.__init__ (vllm/vllm/model_executor/models/AXK1.py), and FusedMoEBlock.__init__ (vllm/vllm/model_executor/models/step3p5.py).

The flattened EP geometry is documented in vllm/vllm/model_executor/model_loader/default_loader.py::DefaultModelLoader._init_ep_weight_filter: 'EP size/rank computation mirrors FusedMoEParallelConfig.make(): ep_size = dp_size * pcp_size * tp_size (flattened); ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank', computed from parallel_config.data_parallel_size/tensor_parallel_size/prefill_context_parallel_size and get_dp_group().rank_in_group, get_tensor_model_parallel_rank(), get_pcp_group().rank_in_group. vllm/vllm/models/inkling/nvidia/moe.py::_inkling_moe_ep_size (and the AMD twin in vllm/vllm/models/inkling/amd/moe.py) implements the same flattening: 'world = get_tensor_model_parallel_world_size() * get_dp_group().world_size * get_pcp_group().world_size' when parallel_config.enable_expert_parallel, else 1. vllm/vllm/model_executor/models/gpt_oss.py uses FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(tp_size=..., dp_size=..., dp_rank=..., pcp_size=..., pcp_rank=...) for the non-EP path and get_ep_group().world_size / .rank_in_group for the EP path in _load_weights_mxfp4 / _load_weights_quark.

(4) How experts are assigned/partitioned.
- MegaMoE (EP-sharded): DeepseekV4MoE._init_mega_moe_experts computes 'n_local_physical_experts = n_physical_experts // ep_size', 'physical_expert_start = ep_rank * n_local_physical_experts', 'physical_expert_end = start + local', and builds DeepseekV4MegaMoEExperts(vllm_config, num_experts=n_physical_experts, num_local_experts=n_local_physical_experts, experts_start_idx=physical_expert_start, num_logical_experts=n_logical_experts, ...). MegaMoE hard-requires EP. DeepseekV4MegaMoEExperts._map_global_expert_id maps a logical expert id to this rank's local slot offsets, handling EPLB redundant replicas; weight_loader copies the checkpoint tensor into the local expert slot.
- Kimi-K3 MegaMoE requires num_experts % ep_size == 0 and builds KimiK3MegaMoEExperts with num_local_experts=num_experts // ep_size and experts_start_idx=ep_rank * num_local_experts.
- Fused-MoE paths may instead partition experts by TP rank, e.g. n_local_experts=n_routed_experts // tp_size and experts_start_idx=tp_rank * n_local_experts.
- fused_moe_make_expert_params_mapping and model-specific mappings convert checkpoint expert tensors into per-expert/per-shard destinations.
- DefaultModelLoader._init_ep_weight_filter calls compute_local_expert_ids(num_experts, ep_size, ep_rank, placement=parallel_config.expert_placement_strategy) so non-local expert tensors can be skipped before disk reads; this filtering is disabled under EPLB because redundant physical slots may need logical experts outside the default partition.
- Inkling pads its routed-expert count to an EP-divisible count because its kernels assume equal contiguous per-rank slabs, then maps global expert ids to local slots and zeroes padding experts after load.

(5) EPLB (expert-placement load balancing). MoE modules derive n_logical_experts, n_physical_experts = logical + redundant, and rank-local physical-expert ranges. At dispatch, MegaMoE implementations call eplb_map_to_physical_and_record(...) with logical-to-physical maps and replica counts. V1 initializes EplbState, registers target and drafter MoE models, updates mappings, and steps the balancer. V2 wraps the equivalent lifecycle in EPLBController. MixtureOfExperts protocol methods keep expert metadata synchronized after remaps.

(6) Related runtime wiring. MoE backends validate EP compatibility; both model runners can query the EP all-to-all manager for fault-tolerance state; sequence-parallel MoE depends on enable_expert_parallel together with TP/DP topology; and shared-expert fusion paths can be enabled or disabled based on EP/SP topology.

INSUFFICIENT CORPUS EVIDENCE: the definitions of ParallelConfig, FusedMoEParallelConfig.make / flatten_tp_across_dp_and_pcp, compute_local_expert_ids, EplbState/EplbLayerState, and the EP all2all manager are referenced but their bodies are not in the corpus; only their call sites and documented contracts are visible.

Sources: vllm/vllm/engine/arg_utils.py, vllm/vllm/model_executor/model_loader/default_loader.py, vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/models/glm5next/nvidia/mtp.py, vllm/vllm/model_executor/models/deepseek_v2.py, vllm/vllm/models/deepseek_v4/nvidia/model.py, vllm/vllm/models/deepseek_v4/amd/model.py, vllm/vllm/models/deepseek_v4/xpu/model.py, vllm/vllm/models/kimi_k3/nvidia/model.py, vllm/vllm/model_executor/models/hy_v3.py, vllm/vllm/model_executor/models/openpangu.py, vllm/vllm/model_executor/models/mimo_v2.py, vllm/vllm/model_executor/models/nemotron_h.py, vllm/vllm/model_executor/models/AXK1.py, vllm/vllm/model_executor/models/step3p5.py, vllm/vllm/model_executor/models/gpt_oss.py, vllm/vllm/models/inkling/nvidia/moe.py, vllm/vllm/models/inkling/amd/moe.py, vllm/vllm/models/minimax_m3/amd/model.py, vllm/vllm/models/minimax_m3/nvidia/model.py, vllm/vllm/transformers_utils/configs/glm53_tp3.py, vllm/vllm/config/speculative.py, vllm/vllm/v1/worker/utils.py, vllm/vllm/v1/worker/gpu_model_runner.py, vllm/vllm/v1/worker/gpu/model_runner.py, usr-local-bin/serve-glm53-flash-tp3-r17.sh, usr-local-bin/serve-dots3-note-prev.sh

## Batch 10 - Q11

```text
GLM-5.3 SOURCE-TRACE BATCH 10 RESULT
Questions selected:             1
Questions answered:             1
With >=1 valid source path:     1
All cited paths valid:          1
Elapsed:                        49.03 sec
API prompt tokens:              950244
Output tokens:                  8032
Reasoning chars:                26781
Finish reason:                  stop
```

### Q11

All of the following is in vllm/vllm/model_executor/layers/linear.py.

PARTITIONING CONTRACT. ColumnParallelLinear implements Y = XA + b with A parallelized along its second (output) dimension: 'A is parallelized along its second dimension as A = [A_1, ..., A_p]'. In __init__, the TP coordinates come from get_tensor_model_parallel_rank()/get_tensor_model_parallel_world_size() unless explicit tp_rank/tp_size kwargs are supplied; disable_tp=True forces (tp_rank, tp_size) = (0, 1), i.e. a replicated layer. The input dimension is never sharded: self.input_size_per_partition = input_size. The output dimension is divided: self.output_size_per_partition = divide(output_size, self.tp_size), and self.output_partition_sizes = [self.output_size_per_partition]; subclasses that define an output_sizes attribute build output_partition_sizes as [divide(output_size, self.tp_size) for output_size in self.output_sizes]. The checkpoint's logical axis is recorded as self.loaded_output_size; self._allow_loaded_output_padding is set when loaded_output_size is explicitly smaller than output_size, enabling destination-local zero padding of the TP shard tail. gather_output controls whether forward() all-gathers the per-rank partial outputs across TP when tp_size > 1; otherwise each rank keeps Y_i = XA_i.

PER-RANK WEIGHT REPRESENTATION. Weights are created by self.quant_method.create_weights(layer, input_size_per_partition, output_partition_sizes, input_size, output_size, params_dtype, weight_loader=...), where the loader is weight_loader_v2 for supported quant methods and v1 otherwise. For the unquantized path, UnquantizedLinearMethod.create_weights allocates a ModelWeightParameter of shape [sum(output_partition_sizes), input_size_per_partition] with input_dim=1, output_dim=0 — each rank holds the full-width input rows and only its output shard. Bias is a Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype)) with output_dim=0 and the same weight_loader. LinearBase.update_param_tp_status() reconciles every child parameter's tp_rank/tp_size to the layer's values.

WEIGHT-LOADER CONTRACT FOR SHARDED PARAMETERS. The v1 weight_loader(param, loaded_weight) validates packed checkpoint axes, computes expected loaded size when output padding is allowed, takes shard_size = param_data.shape[output_dim], start_idx = self.tp_rank * shard_size, and calls load_tensor_parallel_weight(..., start_idx, allow_padding=..., expected_loaded_size=...). The v2 loader mirrors this through BasevLLMParameter.load_column_parallel_weight. Block-scale, Marlin, and packed-parameter shard adjustments occur before narrowing/copying. MergedColumnParallelLinear resolves per-shard offsets/sizes and loads individual fused shards; QKVParallelLinear partitions by heads and replicates KV heads when needed, with q shard rank = tp_rank and k/v shard rank = tp_rank // num_kv_head_replicas.

INSUFFICIENT CORPUS EVIDENCE: the internal mechanics of load_tensor_parallel_weight and BasevLLMParameter.load_column_parallel_weight live in vllm/model_executor/parameter.py, which is not included; only their call signatures and validation inputs are visible.

Sources: vllm/vllm/model_executor/layers/linear.py

## Batch 11 - Q12

```text
GLM-5.3 SOURCE-TRACE BATCH 11 RESULT
Questions selected:             1
Questions answered:             1
With >=1 valid source path:     1
All cited paths valid:          0
Elapsed:                        38.03 sec
API prompt tokens:              950235
Output tokens:                  7219
Reasoning chars:                24947
Finish reason:                  stop
Cited path absent from manifest: vllm/vllm/models/deepseek_v2.py
```

### Q12

Both classes live in vllm/vllm/model_executor/layers/linear.py and share the LinearBase constructor, but they partition opposite axes of Y = XA + b and therefore communicate at opposite ends of the layer.

ColumnParallelLinear — output-dim partition, optional output all-gather: A is partitioned along its second/output dimension; input_size_per_partition = input_size, output_size_per_partition = divide(output_size, tp_size). forward() communicates only when gather_output and tp_size > 1, via tensor_model_parallel_all_gather(output_parallel). The default gather_output=False path is collective-free. Its weight loader slices along param.output_dim and uses a rank-local start offset.

RowParallelLinear — input-dim partition, output all-reduce: A is partitioned along its first/input dimension and X along its last dimension; input_size_per_partition = divide(input_size, tp_size), output_size_per_partition = output_size. If input_is_parallel=True the caller provides an already-split tensor; otherwise the layer splits the input along the last dimension. Bias is fused only on rank 0. If reduce_results and tp_size > 1, output = tensor_model_parallel_all_reduce(output_parallel); reduce_results defaults to True. Immediately before that all-reduce an optional `_l2_prefetch_pre_reduce_hook` can fire, which GLM-5.3 uses for L2 weight prefetch. Its loader slices along param.input_dim with start_idx = tp_rank * shard_size. disable_tp forces rank0/size1, replicating the weights and removing collectives.

Typical pairing: a column-parallel producer such as q_b_proj/qkv_proj/gate_up_proj emits sharded outputs that feed a row-parallel consumer such as o_proj/down_proj, whose all-reduce restores the full hidden state.

Sources: vllm/vllm/model_executor/layers/linear.py, vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/models/deepseek_v2.py, vllm/vllm/models/deepseek_v4/nvidia/model.py, vllm/vllm/models/kimi_k3/nvidia/model.py

## Batch 4 - Q13-Q16

```text
GLM-5.3 SOURCE-TRACE BATCH 4 RESULT
Questions selected:             4
Questions answered:             4
With >=1 valid source path:     4
All cited paths valid:          4
Elapsed:                        164.40 sec
API prompt tokens:              950348
Output tokens:                  26606
Reasoning chars:                71812
Finish reason:                  stop
```

### Q13

Generic entry: `DefaultModelLoader.load_weights` (vllm/vllm/model_executor/model_loader/default_loader.py) calls `model.load_weights(self.get_all_weights(model_config, model))`. `get_all_weights` builds a `DefaultModelLoader.Source` with `weight_name_prefixes=getattr(model, 'checkpoint_weight_name_prefixes', None)` (Glm5NextMTP defines this in vllm/vllm/models/glm5next/nvidia/mtp.py to restrict loading to MTP-layer tensors, and `_prepare_weights` filters files via `filter_safetensors_files_by_weight_name_prefixes`), and `_get_weights_iterator` selects `instanttensor_weights_iterator(..., copy=self.instanttensor_copy, distributed=self.instanttensor_distributed, indexed_tensor_files=indexed_tensor_files)` for `load_format == 'instanttensor'` — the format every GLM launcher passes (`--load-format instanttensor`). `_init_ep_weight_filter` pre-computes `local_expert_ids` so non-local routed-expert tensors are skipped before disk reads under `--enable-expert-parallel`.

GLM MLA projection (`fused_qkv_a_proj`): `Glm5NextForCausalLM.load_weights` runs `AutoWeightsLoader(self).load_weights(weights)` then `process_weights_after_loading()`; the real mapping lives in `Glm5NextModel.load_weights` (vllm/vllm/models/glm5next/nvidia/model.py). Checkpoint `.q_a_proj` / `.kv_a_proj_with_mqa` match stacked entries `('.fused_qkv_a_proj', '.q_a_proj', 0)` / `('.fused_qkv_a_proj', '.kv_a_proj_with_mqa', 1)` and are loaded via `weight_loader(param, loaded_weight, shard_id)`. The destination is a `MergedColumnParallelLinear`; its `weight_loader` computes rank-local shard offsets/sizes, adjusts block-scale/packing geometry, narrows the destination parameter, and calls `load_tensor_parallel_weight` with a rank-local checkpoint start index.

Custom GLM loader logic includes: NoPE padding for `kv_a_proj_with_mqa`; legacy block-FP8 attention-projection buffering/dequantization through `_try_load_fp8_attn_proj` / `_dequant_fp8_block`; MXFP8 BF16 indexer-projection loading through `_try_load_mxfp8_bf16_attn_proj` / `dequant_mxfp8_to_bf16`; fused KDA conv loading through `_load_glm5next_fused_conv1d`; Kimi/GDN merged projection `_KimiGDNMergedColumnParallelLinear` with a replicated f_a shard; rank-local tail loaders for dt_bias/A_log; and zero-padding for physical TP3 destination axes through the generic linear padded-axis validation/loaders.

Sources: vllm/vllm/model_executor/model_loader/default_loader.py, vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/models/glm5next/nvidia/mtp.py, vllm/vllm/model_executor/layers/linear.py, vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py, vllm/vllm/transformers_utils/configs/glm53_tp3.py

### Q14

The GLM load_weights implementation is `Glm5NextModel.load_weights` in vllm/vllm/models/glm5next/nvidia/model.py, reached through `Glm5NextForCausalLM.load_weights`. Name transformations include `_remap_glm5next_weight_name` for KDA forget-gate and mHC names; skipping target-side speculative-layer weights and rotary-cache tensors; stacked mappings `.gate_proj/.up_proj -> .gate_up_proj`, `.q_a_proj/.kv_a_proj_with_mqa -> .fused_qkv_a_proj`, `.q_proj/.k_proj/.v_proj/.b_proj/.f_a_proj -> .in_proj_qkvgfab`, and `.q_conv1d/.k_conv1d/.v_conv1d -> .conv1d`; routed-expert mapping through `fused_moe_make_expert_params_mapping`; KV-scale remapping; legacy FP8 and MXFP8 conversion; and NoPE row padding for `kv_a_proj_with_mqa`.

The MTP variant `Glm5NextMTP.load_weights` applies a `WeightsMapper`, rewrites speculative layer names into `.mtp_block.`, hoists shared weights, and requires all MTP layer weights. The multimodal wrapper inherits GLM4V prefix mapping for visual/language/lm_head paths.

Sources: vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/models/glm5next/nvidia/mtp.py, vllm/vllm/model_executor/models/glm4_1v.py

### Q15

`ModelConfig._verify_quantization` reads the checkpoint/model quantization config, lets registered quantization methods override the configured method, and stores the selected method. `VllmConfig.__post_init__` then instantiates `self.quant_config` through `get_quant_config`, validates device capability and supported activation dtype, and allows the quant config to update itself from the model/HF config.

GLM MLA attention receives `quant_config` only when the active method name is `modelopt_mixed`, because mixed ModelOpt checkpoints describe each projection independently. `MLAAttention` stores that quant config, initializes KV-cache quantization, and later asks `quant_config.get_quant_method(self,prefix)` while processing loaded weights. `Glm5NextMoE` passes the same quant config into the shared-expert `Glm5NextMLP` and `FusedMoEFactory`.

The concrete linear implementation is selected by `LinearBase.__init__`: no quant config -> UnquantizedLinearMethod; otherwise `quant_config.get_quant_method(self,prefix)` decides the concrete method. Supported methods include ModelOpt NVFP4/W4A16/FP8 variants, Humming, compressed-tensors, Qutlass NVFP4, and others; that method also decides whether v1 or v2 weight loading is used.

Sources: vllm/vllm/config/model.py, vllm/vllm/config/vllm.py, vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/model_executor/layers/linear.py, vllm/vllm/model_executor/layers/attention/mla_attention.py, b12x/b12x/integration/vllm/plugin.py, usr-local-libexec/serve-glm53-flash-tp3-r17-dense.sh, vllm/serve-glm53-flash-nvfp4.sh

### Q16

FP4/NVFP4 affects actual storage, loading, kernel policy, and cache execution. Runtime paths visible in the supplied source include packed FP4 expert weights/scales; ModelOpt NVFP4 checkpoint-name/layout conversions; FP4-specific expert scale suffixes and packed uint8 handling; MXFP4/NVFP4 MoE loader paths in GPT-OSS/Inkling; `VLLM_B12X_MOE_FP4_FORCE_A16` switching B12X NVFP4 routed experts between W4A4 and W4A16 activation contracts; B12X MoE policy selection keyed by quant_mode/source format; NVFP4 linear quant methods using the v2 loader; dynamic FP4 output quantization in MLA through `ops.scaled_fp4_quant`; and B12X sparse-MLA `nvfp4_ds_mla` cache records/kernels with 304/368-byte packed records and dedicated cache-update functions.

Sources: vllm/vllm/envs.py, vllm/vllm/config/vllm.py, vllm/vllm/config/model.py, vllm/vllm/model_executor/layers/linear.py, vllm/vllm/model_executor/layers/attention/mla_attention.py, vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py, vllm/vllm/models/deepseek_v4/nvidia/model.py, vllm/vllm/models/deepseek_v4/nvidia/dspark.py, vllm/vllm/models/deepseek_v4/nvidia/mtp.py, vllm/vllm/models/deepseek_v4/amd/dspark.py, vllm/vllm/models/inkling/nvidia/moe.py, vllm/vllm/models/inkling/amd/moe.py, vllm/vllm/model_executor/models/gpt_oss.py, vllm/vllm/model_executor/models/mllama4.py, vllm/vllm/model_executor/models/config.py, b12x/b12x/tools/inspect_model_policy.py, vllm/serve-glm53-flash-nvfp4.sh, usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh

## Batch 5 - Q17-Q20

```text
GLM-5.3 SOURCE-TRACE BATCH 5 RESULT
Questions selected:             4
Questions answered:             4
With >=1 valid source path:     4
All cited paths valid:          2
Elapsed:                        161.54 sec
API prompt tokens:              950350
Output tokens:                  25642
Reasoning chars:                64268
Finish reason:                  stop
Cited paths absent from manifest:
  Q17: vllm/vllm/models/deepseek_v2.py
  Q19: .github/mergify.yml
```

### Q17

Code-structure trace of one token through the GLM (Glm5Next) MoE path:

1) Layer dispatch — `Glm5NextDecoderLayer.__init__` in vllm/vllm/models/glm5next/nvidia/model.py reads the model's MoE configuration and chooses `Glm5NextMoE` for sparse MoE layers, otherwise `Glm5NextMLP`.

2) Router/gating — `Glm5NextMoE.__init__` creates a `GateLinear` router with output size equal to the routed-expert count. The router dtype comes from `_get_moe_router_dtype(config)` and `e_score_correction_bias` is present for `topk_method == "noaux_tc"`.

3) Expert selection — the constructor builds `FusedMoEFactory` with the routed-expert count, top-k, grouped-top-k configuration, expert groups, scoring function, routed scaling factor, router-logit dtype, EPLB fields, correction bias, and shared experts.

4) Expert execution — `Glm5NextMoE.forward` optionally sequence-parallel chunks the input, evaluates the gate, then calls `self.experts(hidden_states=hidden_states, router_logits=router_logits)`. Under sequence parallel it all-gathers the result and trims it back to the original token count. The concrete fused-MoE kernel body is outside the included corpus, but B12X MoE policy presets are present in `b12x/b12x/tools/inspect_model_policy.py`.

5) Shared experts / combination — shared experts are a `Glm5NextMLP` with `reduce_results=False`, passed into `FusedMoEFactory`; the GLM TP3 physical shared-expert size can be larger than the checkpoint size, with `loaded_intermediate_size` retaining the logical checkpoint width. The decoder then returns the MoE output into the residual/mHC path.

6) Launcher/config wiring — the GLM launchers pass the target MoE backend and linear backend, with `VLLM_B12X_MOE_FP4_FORCE_A16` selecting the target NVFP4 activation contract. The GLM-5.3 MTP draft uses Humming for its MXFP8 expert while the target's NVFP4 routed experts use the target MoE path.

7) Weight loading / MTP participation — `Glm5NextModel.load_weights` uses `fused_moe_make_expert_params_mapping` to route checkpoint expert tensors to local fused-expert parameters. `Glm5NextMTP.set_moe_parameters` collects MTP MoE blocks, and GLM TP3 draft geometry keeps MTP in target EP topology while DFlash remains dense.

Sources: vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/models/glm5next/nvidia/mtp.py, vllm/vllm/models/deepseek_v2.py, vllm/vllm/transformers_utils/configs/glm53_tp3.py, b12x/b12x/tools/inspect_model_policy.py, usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, usr-local-bin/serve-glm53-flash-tp3-r17.sh, usr-local-bin/serve-glm53-flash-nvfp4.sh

### Q18

MTP configuration trace, launcher -> runtime:

1) The launcher emits a `--speculative-config` for `method:"mtp"`, with num_speculative_tokens plus MTP-specific MoE and attention backends. Some related launchers add probabilistic/local-argmax/adaptive-depth controls.

2) `EngineArgs.create_speculative_config` turns the JSON into `SpeculativeConfig`, injecting target model and parallel configuration.

3) For MTP, `SpeculativeConfig` derives the drafter from the target checkpoint, inherits quantization, rewrites GLM5Next to the MTP architecture, carries the requested MTP kernel backends, and applies GLM TP3 draft geometry.

4) `VllmConfig` exposes speculative-token and lookahead counts and adapts graph/scheduling behavior when speculative decoding is enabled.

5) `Scheduler` reserves lookahead/draft space, stores draft token IDs, and on model output calculates accepted/rejected speculative tokens and rolls back rejected placeholders/tokens.

6) The GPU model runner initializes the speculator/proposer. Draft tokens are generated by `self.speculator.propose(...)` / `self.drafter.propose(...)`, written into request-state draft buffers, and then verified by the rejection sampler against target logits.

7) `llm_base_proposer.py` shows the repeated-forward MTP behavior: MTP is not parallel drafting, so multiple speculative tokens are produced by sequential draft forwards. The GLM MTP implementation is `Glm5NextMTP` / `Glm5NextMultiTokenPredictor` / `Glm5NextMultiTokenPredictorLayer`.

Sources: usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, usr-local-libexec/serve-glm53-flash-tp3-r17-dense.sh, usr-local-bin/serve-glm51.sh, usr-local-bin/serve-glm52.sh, usr-local-bin/serve-glm52-v16.sh, usr-local-bin/serve-ds4-flash.sh, usr-local-bin/serve-ds4-pro-tp10.sh, vllm/serve-glm53.sh, vllm/serve-glm53-flash-nvfp4.sh, vllm/vllm/engine/arg_utils.py, vllm/vllm/config/speculative.py, vllm/vllm/config/vllm.py, vllm/vllm/v1/core/sched/scheduler.py, vllm/vllm/v1/worker/gpu/model_runner.py, vllm/vllm/v1/worker/gpu_model_runner.py, vllm/vllm/v1/spec_decode/llm_base_proposer.py, vllm/vllm/models/glm5next/nvidia/mtp.py, vllm/vllm/models/deepseek_v4/nvidia/mtp.py

### Q19

DFlash/DFlash2 speculative path in this image:

1) Draft-model selection — launchers define a separate DFlash model/revision, draft KV dtype, and draft attention backend. `dflash|dflash2` emits a speculative config using `method:"dflash"`, a separate draft model, and seven draft tokens by default for the DFlash2 eight-token block.

2) Runtime selection — `SpeculativeConfig` creates a separate draft `ModelConfig`, auto-detects DFlash architecture, and sets `parallel_drafting=True` for DFlash/DSpark. The draft gets its own attention backend and KV-cache dtype. GLM TP3 draft geometry pads the DFlash heads/vocab while explicitly disabling expert parallelism for the dense draft.

3) DFlash2 vs DFlash1 — both use method `dflash`; DFlash2 is identified by the draft architecture. The V2 runner is required for the DFlash2 candidate selector.

4) Structural difference from MTP — DFlash uses a separate draft checkpoint, produces K draft tokens in parallel from one draft forward, reserves extra draft query slots, owns independent draft KV state, and uses GLM aux hidden-state capture/contracting for DFlash. MTP derives its draft from the target and repeatedly advances the MTP layer sequentially while sharing target state/topology.

5) Execution plumbing — V1 uses a DFlash proposer, V2 uses the speculator path, and verification still converges on the shared target rejection sampler/scheduler acceptance accounting. Capture wrappers enlarge/shape graph sizes because DFlash K7 verifies eight target rows per request.

Sources: usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, usr-local-libexec/serve-glm53-flash-tp3-r17-dense.sh, usr-local-bin/serve-glm53-flash-tp3-r17.sh, usr-local-bin/serve-glm53-flash-nvfp4.sh, usr-local-bin/serve-glm53-flash-tp3-r17-capture.sh, usr-local-bin/serve-glm53-flash-lmcache.sh, usr-local-bin/serve-mimo25-pro-dflash.sh, usr-local-bin/serve-laguna-s21-nvfp4.sh, usr-local-bin/serve-qwen36.sh, vllm/vllm/config/speculative.py, vllm/vllm/config/vllm.py, vllm/vllm/transformers_utils/configs/glm53_tp3.py, vllm/vllm/v1/spec_decode/llm_base_proposer.py, vllm/vllm/v1/worker/gpu_model_runner.py, vllm/vllm/v1/worker/gpu/model_runner.py, vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/v1/core/sched/scheduler.py, .github/mergify.yml

### Q20

The R17 GLM-5.3 launcher accepts `SPECULATOR=mtp|dflash|dflash2`.

MTP: depth comes from NUM_SPECULATIVE_TOKENS/MTP. When positive, the generated config uses `method:"mtp"`, target revision, depth, MTP MoE backend, and MTP attention backend. Runtime `SpeculativeConfig` derives the MTP drafter from the target checkpoint.

DFlash / DFlash2: default depth is seven. The config uses `method:"dflash"`, separate draft model/revision, draft KV-cache dtype, and draft attention backend. DFlash2 behavior is identified from the draft checkpoint architecture, not a separate runtime method string.

The R17 dispatcher additionally locks the target and draft model/revisions and prevents callers from overriding the speculative config. The capture wrapper and LMCache wrapper derive graph/scheduling budgets from the selected speculation mode. Runtime `EngineArgs` converts the emitted JSON to `SpeculativeConfig`; DFlash is parallel drafting with independent draft KV, whereas MTP is target-derived sequential drafting.

Sources: usr-local-bin/serve-glm53-flash-tp3-r17.sh, usr-local-bin/serve-glm53-flash-tp3-r17-capture.sh, usr-local-bin/serve-glm53-flash-nvfp4.sh, usr-local-bin/serve-glm53-flash-lmcache.sh, usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, usr-local-libexec/serve-glm53-flash-tp3-r17-dense.sh, vllm/serve-glm53-flash-nvfp4.sh, vllm/vllm/engine/arg_utils.py, vllm/vllm/config/speculative.py, vllm/vllm/config/vllm.py, vllm/vllm/v1/spec_decode/llm_base_proposer.py

## Batch 6 - Q21-Q23

```text
GLM-5.3 SOURCE-TRACE BATCH 6 RESULT
Questions selected:             3
Questions answered:             3
With >=1 valid source path:     3
All cited paths valid:          2
Elapsed:                        256.03 sec
API prompt tokens:              950301
Output tokens:                  44253
Reasoning chars:                139743
Finish reason:                  stop
Cited paths absent from corpus manifest:
  Q21: vllm/vllm/models/glm4_1v.py
  Q21: vllm/vllm/models/mllama4.py
```

### Q21

Configuration surface: `--mm-encoder-tp-mode` is registered in vllm/vllm/engine/arg_utils.py, exposed through EngineArgs, and forwarded into ModelConfig. `ModelConfig.__post_init__` normalizes unsupported `data` mode back to `weights`, then stores the result in MultiModalConfig.

Model consumption: multimodal wrappers derive `self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == 'data'`. In data mode, the vision tower uses replicated weights (`tp_size=1`/ReplicatedLinear or disable_tp) while input media/examples are sharded across ranks through DP-sharded vision helpers. In weights mode, the encoder retains normal TP weight sharding and every rank participates in the same encoder input.

GLM-5.3 TP3 specialization: `apply_glm53_tp3_target_geometry` applies vision geometry padding only when the multimodal config is in `weights` mode, validating logical vision heads/intermediate/projection shapes and rewriting them to the physical TP3-compatible geometry. The R17 launcher itself requires and locks `MM_ENCODER_TP_MODE=weights`, rejects CLI overrides, and appends `--mm-encoder-tp-mode weights`. `log_glm53_r17_tp3_runtime_proof` then re-reads the constructed runtime config and requires `mm_encoder_tp_mode=='weights'`, failing startup otherwise.

Sources: vllm/vllm/engine/arg_utils.py, vllm/vllm/config/model.py, vllm/vllm/config/vllm.py, vllm/vllm/config/speculative.py, vllm/vllm/transformers_utils/configs/glm53_tp3.py, vllm/vllm/models/glm4_1v.py, vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/models/minimax_m3/nvidia/model.py, vllm/vllm/models/minimax_m3/amd/model.py, vllm/vllm/models/mllama4.py, vllm/vllm/models/qwen3_8_flash_next/model.py, vllm/vllm/models/kimi_k3/nvidia/model.py, vllm/vllm/v1/worker/utils.py, vllm/vllm/v1/worker/gpu_model_runner.py, vllm/vllm/v1/worker/gpu/model_runner.py, usr-local-bin/serve-glm53-flash-tp3-r17.sh

### Q22

Launcher/runtime settings: GLM launchers pass `MAX_CUDAGRAPH_CAPTURE_SIZE`, `CUDAGRAPH_MODE`, and explicit capture-size lists. The R17 capture wrapper has default sizes extending through 256 but the R17 dispatcher locks `CUDAGRAPH_MODE=FULL`, `MAX_CUDAGRAPH_CAPTURE_SIZE=16`, and `CUDAGRAPH_CAPTURE_SIZES='1 2 4 8 16'`, while preventing CLI overrides. The wrapper validates the list and injects `--cudagraph-capture-sizes` into the base launcher.

Engine plumbing: EngineArgs exposes `--cudagraph-capture-sizes` and `--max-cudagraph-capture-size`, checks conflicts with values embedded in CompilationConfig, and builds the final compilation configuration. VllmConfig applies optimization-level defaults, eager/compile constraints, dynamic-speculative-decoding constraints, encoder-decoder/pooling/KV-connector restrictions, then resolves capture sizes. The contract explicitly states that a batch at or below a captured size can use the closest padded graph; batches larger than the largest captured size do not use a captured graph.

Attention/backend support further constrains the mode and sizes. V1 uses a CudagraphDispatcher and capture descriptors; V2 uses ModelCudaGraphManager. Runtime dispatch chooses FULL replay, PIECEWISE execution, or eager/NONE depending on the resolved batch descriptor and incompatibilities such as cascade attention/encoder output. Worker warmup profiles compile/capture memory and captures the model after B12X/kernel warmup.

Sources: usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, usr-local-bin/serve-glm53-flash-nvfp4-dflash2.sh, usr-local-bin/serve-glm53-flash-tp3-r17.sh, usr-local-bin/serve-glm53-flash-tp3-r17-capture.sh, usr-local-bin/serve-glm52.sh, usr-local-bin/serve-glm52-v16.sh, usr-local-bin/serve-ds4-flash.sh, usr-local-bin/serve-ds4-flash-spark.sh, usr-local-bin/serve-ds4-pro-tp10.sh, usr-local-bin/serve-minimax-m3-mxfp8.sh, vllm/vllm/engine/arg_utils.py, vllm/vllm/config/vllm.py, vllm/vllm/v1/worker/gpu_model_runner.py, vllm/vllm/v1/worker/gpu/model_runner.py, vllm/vllm/v1/worker/gpu_worker.py, vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py, vllm/vllm/v1/spec_decode/llm_base_proposer.py, vllm/vllm/v1/worker/utils.py

### Q23

Launcher surface: GLM and related launchers enable prefix caching directly or gate it through an environment variable. EngineArgs carries `enable_prefix_caching`, hash algorithm, retention interval, and prefix-match unit. The default is resolved from model support and becomes `CacheConfig.enable_prefix_caching`.

Model/config coupling: ModelConfig determines whether the architecture supports prefix caching. VllmConfig disables it for encoder-only multimodal mode and validates related cache features. Mamba/hybrid configs couple prefix caching to the recurrent-cache mode and block size: with prefix caching they use aligned/checkpoint-aware behavior; without it the Mamba cache mode is forced to none and block sizing changes.

Scheduler consumption: Scheduler constructs `KVCacheManager(..., enable_caching=self.cache_config.enable_prefix_caching, ...)`, asks the manager for computed blocks, records prefix-hit stats, caches completed blocks, handles remote/local prefix hits, respects retention/hash boundaries, and computes common-prefix blocks for cascade attention. Speculative decoders that share/pollute target KV can cause the scheduler to back off the last matching block. Partial hits can require copy-on-write block copies, which the model runner applies before the forward pass.

Worker consumption: KV block zeroing/copies are applied to newly allocated/copy-on-write blocks, and common-prefix block counts feed cascade-attention prefix length. The KV-cache-manager internals themselves are not in the corpus, so allocation/eviction/hash internals beyond these call sites are INSUFFICIENT CORPUS EVIDENCE.

Sources: usr-local-libexec/serve-glm53-flash-nvfp4-dflash2.sh, usr-local-bin/serve-glm53-flash-nvfp4.sh, usr-local-bin/serve-ds4-flash.sh, usr-local-bin/serve-ds4-flash-spark.sh, usr-local-bin/serve-glm53-flash-lmcache.sh, usr-local-bin/serve-kimi-k3-qsrt.sh, vllm/serve-ds4-flash.sh, vllm/vllm/engine/arg_utils.py, vllm/vllm/config/model.py, vllm/vllm/config/vllm.py, vllm/vllm/model_executor/models/config.py, vllm/vllm/v1/core/sched/scheduler.py, vllm/vllm/v1/worker/gpu_model_runner.py, vllm/vllm/v1/worker/gpu/model_runner.py, vllm/vllm/v1/worker/utils.py, vllm/vllm/v1/worker/gpu_worker.py, vllm/vllm/envs.py

## Batch 12 - Q24

```text
GLM-5.3 SOURCE-TRACE BATCH 12 RESULT
Questions selected:             1
Questions answered:             1
With >=1 valid source path:     1
All cited paths valid:          1
Elapsed:                        156.72 sec
API prompt tokens:              950237
Output tokens:                  26379
Reasoning chars:                76729
Finish reason:                  stop
```

### Q24

KV-cache dtype flows from CLI to allocation to attention execution through a chain of resolution, canonicalization, spec generation, and binding steps.

1) CONFIGURATION ENTRY AND RESOLUTION. EngineArgs declares `kv_cache_dtype`, exposes `--kv-cache-dtype`, resolves it through `resolve_kv_cache_dtype_string`, and constructs CacheConfig. VllmConfig validates specialized MLA cache dtype choices and KV-transfer/offload compatibility.

2) PER-LAYER OVERRIDE AND BACKEND CANONICALIZATION. `MLAAttention.__init__` reads CacheConfig.cache_dtype, applies skip-layer overrides, selects the attention backend, and canonicalizes sparse MLA backends: B12X/FlashInfer sparse MLA with ordinary FP8 can be normalized to `fp8_ds_mla`. The normalized value is written back so the cache manager and layer agree. Packed sparse MLA types are `fp8_ds_mla` / `nvfp4_ds_mla`; ordinary quantized caches are viewed as the platform FP8 dtype while packed records remain uint8.

3) SPEC GENERATION. GPUModelRunner collects each attention module's KV-cache spec and allows the backend to customize it. MLA specs carry block size, head size, torch dtype, cache dtype string, quant mode, alignment, and custom packed-record byte sizes. B12X sparse MLA advertises supported cache dtypes/layouts, enforces GLM recipe compatibility, and rewrites the spec for 528-byte FP8 or 304/368-byte NVFP4 packed records and index-page alignment.

4) ALLOCATION AND BINDING. `allocate_kv_cache` creates one int8 backing allocation and per-layer views according to the spec/layout. `bind_kv_cache` installs views into the forward context; MLAAttention and the B12X implementation validate shape, dtype, record size, and finalized page size. Common kernel block sizes must divide the scheduler/manager block size. Zeroing/copy-on-write operate in byte-addressed form over the allocated views.

5) FP8 REPRESENTATION AT RUNTIME. V1/V2 runners derive the torch dtype from CacheConfig. MLAAttention tracks the string dtype and per-layer scales and passes them into unified MLA cache-update ops. B12X packed sparse MLA stores the cache as uint8 records and dispatches dedicated FP8/NVFP4 concat-and-cache functions. DeepSeek-V4 similarly maps `fp8_ds_mla` to uint8 packed storage while plain fp8 uses float8_e4m3fn; insert/read kernels branch on that representation. Other MLA implementations selectively view ordinary FP8 caches as float8 while leaving packed formats native.

6) DRAFT-MODEL KV DTYPE. SpeculativeConfig allows a draft-specific kv_cache_dtype; when absent the draft inherits the target. DFlash launchers commonly override this to `auto`/BF16-compatible layouts rather than target packed-MLA formats.

7) LAUNCHER POLICY. The GLM launchers validate/pass target KV_CACHE_DTYPE and the R17 TP3 profile locks it to `fp8`.

8) COMPONENTS THAT MUST AGREE: CacheConfig.cache_dtype; per-layer attention kv_cache_dtype; AttentionSpec dtype/layout/record bytes/alignment/block size; allocated views; backend bind checks; kernel page size; cache update/read kernel; per-layer quant scales; resolved layout; kernel block size; and draft-vs-target cache dtype.

INSUFFICIENT CORPUS EVIDENCE: generic non-MLA Attention internals and several core KV spec/cache classes are referenced but their defining files are absent.

Sources: vllm/vllm/engine/arg_utils.py, vllm/vllm/config/vllm.py, vllm/vllm/model_executor/models/config.py, vllm/vllm/model_executor/layers/attention/mla_attention.py, vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py, vllm/vllm/models/deepseek_v4/attention.py, vllm/vllm/models/deepseek_v4/nvidia/dspark.py, vllm/vllm/models/deepseek_v32/attention.py, vllm/vllm/models/minimax_m3/nvidia/model.py, vllm/vllm/models/minimax_m3/amd/model.py, vllm/vllm/v1/worker/gpu_model_runner.py, vllm/vllm/v1/worker/gpu/model_runner.py, vllm/vllm/v1/worker/utils.py, vllm/vllm/v1/worker/gpu_worker.py, vllm/vllm/config/speculative.py, vllm/vllm/v1/spec_decode/llm_base_proposer.py, vllm/vllm/envs.py, usr-local-bin/serve-glm53-flash-nvfp4-dflash2.sh, usr-local-libexec/serve-glm53-flash-tp3-r17-dense.sh, usr-local-bin/serve-glm52-v16.sh, usr-local-bin/serve-glm52-hybrid-v19.sh, usr-local-bin/serve-glm53-flash-lmcache.sh, usr-local-bin/serve-ds4-flash.sh, usr-local-bin/serve-glm52.sh, usr-local-bin/serve-kimi-k3-qsrt.sh, vllm/serve-glm53.sh

## Batch 13 - Q25

```text
GLM-5.3 SOURCE-TRACE BATCH 13 RESULT
Questions selected:             1
Questions answered:             1
With >=1 valid source path:     1
All cited paths valid:          1
Elapsed:                        155.78 sec
API prompt tokens:              950284
Output tokens:                  25131
Reasoning chars:                71898
Finish reason:                  stop
```

### Q25

TP3 R17 LAUNCH POLICY, RECONSTRUCTED END-TO-END

1. Dispatcher gate (`usr-local-bin/serve-glm53-flash-tp3-r17.sh`): TP=4 delegates to the qualified TP4 launcher; TP=3 enters the R17 TP3 policy; other TP sizes are rejected. TP3 also rejects LMCache, DCP != 1, multimodal mode != weights, positional model overrides, unknown speculation modes, split-page cache variables, and locked CLI overrides. The locked options include model/revision/spec config, TP/PP/DCP, EP toggles, custom all-reduce toggle, graph controls, scheduler/context/memory limits, KV dtype and cache-memory overrides, additional/compilation/transfer config, and offload controls. `lock_env`/`lock_env_from_parent` enforce exact environment values.

2. Locked values: target model/revision and DFlash model/revision; TP=3/DCP=1; MM encoder weights mode; cache interleave=4; 1,048,576 context; 8 sequences; 8192 batched/scheduled tokens; prefill interval 8; GPU utilization .91; FULL graphs capped at 16 with sizes 1,2,4,8,16; dense hybrid cache; KV fp8; prefix caching on; B12X attention and linear; target MoE auto; MTP attention B12X and MTP MoE Humming; B12X target NVFP4 W4A4; B12X KDA decode; FlashKDA prefill; B12X PCIe all-reduce and one-shot threshold 84KB; runtime-proof gate; isolated JIT/cache fingerprint. Final exec forcibly appends `--enable-expert-parallel --mm-encoder-tp-mode weights`.

3. Capture wrapper: validates/injects the locked exact graph sizes and invokes the dense R17 libexec launcher.

4. Base launcher: validates TP/DCP/interleave/scheduler settings, enforces dense-cache contract, constructs either MTP or DFlash speculative config, and launches vLLM with TP3, PP1, DCP1, 1M context, seq8, batch8192, chunked prefill, BF16 compute, FP8 KV, modelopt_mixed, block256, instanttensor, B12X attention/linear, target MoE auto, B12X KDA decode + FlashKDA prefill additional config, FULL graphs, and prefix caching. Custom all-reduce is left enabled because B12X PCIe all-reduce is mandatory.

5. Deeper vLLM interpretation: VllmConfig applies GLM TP3 physical geometry before generic validation. `apply_glm53_tp3_target_geometry` requires expert parallelism and rewrites logical 64-head geometry into physical 72 MLA / 66 KDA heads, pads shared-expert/MTP/vocab/vision geometry, and recomputes model config. SpeculativeConfig couples target and draft topology, keeping MTP in target EP topology but DFlash dense/outside EP. Model code consumes the padded geometry in embedding/lm-head/shared experts/MTP projections/KDA loaders. `log_glm53_r17_tp3_runtime_proof` is called after model load and verifies the live B12X PCIe one-shot collective, EP size 3, B12X KDA decode, FlashKDA prefill, and weights-mode multimodal configuration; any mismatch aborts startup.

6. Launcher policy vs vLLM interpretation: shell code enforces the fixed profile, locked CLI/env, dense cache, graph list, cache fingerprint, and mandatory EP/MM flags. Runtime vLLM materializes physical geometry, constructs the draft topology/tensors, selects/binds kernels and collectives, and re-verifies the actual live runtime proof.

`B12X_PCIE_ALLREDUCE_ALGORITHM` is locked/exported by the launcher, but no included corpus file shows its runtime consumer: INSUFFICIENT CORPUS EVIDENCE for its deeper interpretation.

Sources: usr-local-bin/serve-glm53-flash-tp3-r17.sh, usr-local-bin/serve-glm53-flash-tp3-r17-capture.sh, usr-local-libexec/serve-glm53-flash-tp3-r17-dense.sh, usr-local-bin/serve-glm53-flash-lmcache.sh, vllm/vllm/config/vllm.py, vllm/vllm/config/speculative.py, vllm/vllm/transformers_utils/configs/glm53_tp3.py, vllm/vllm/v1/worker/utils.py, vllm/vllm/v1/worker/gpu/model_runner.py, vllm/vllm/v1/worker/gpu_model_runner.py, vllm/vllm/models/glm5next/nvidia/model.py, vllm/vllm/models/glm5next/nvidia/mtp.py, vllm/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py, vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py, vllm/vllm/envs.py

</details>
