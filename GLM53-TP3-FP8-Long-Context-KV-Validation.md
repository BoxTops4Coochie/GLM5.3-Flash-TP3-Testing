# GLM-5.3 Flash TP3 — FP8 Long-Context KV Cache Validation

## Executive Summary

**Test 1 - Distributed Exact Retrieval:** Ten unique records were distributed throughout contexts ranging from roughly 122K to 953K API prompt tokens. FP8 KV returned every requested code exactly, finishing **50/50 overall** and **10/10 at 952,751 tokens**.

**Test 2 - Distributed Multi-Hop Reasoning:** Ten chains per context required linking customer → account → service tier → multiplier plus a separate base allocation, followed by arithmetic. FP8 KV finished **30/30 overall**, including **10/10 at 963,574 API prompt tokens**.

**Test 3 - Adversarial Temporal / Revision Reasoning:** At 966,393 API prompt tokens, each answer required resolving superseded records, rejecting obsolete/draft decoys, traversing a longer graph, applying the active calculation-order policy, and performing arithmetic. The successful run completed naturally and scored **10/10**, using **15,760 output tokens**.

**Test 4 - Real Linux Kernel Documentation Exam:** A deterministic **950,000-token** corpus built from **395 upstream Linux documentation files** was followed by 23 technical questions. GLM answered **23/23**, cited valid in-corpus source paths for **23/23**, and completed naturally with `finish_reason=stop`; semantic correctness can be graded separately against the cited source text.

## Combined Results

| Test | Largest API prompt | Result |
|---|---:|---:|
| Test 1 - Distributed Exact Retrieval | 952,751 | 10/10 at largest context |
| Test 2 - Distributed Multi-Hop Reasoning | 963,574 | 10/10 at largest context |
| Test 3 - Adversarial Temporal / Revision Reasoning | 966,393 | 10/10 |
| Test 4 - Real Linux Kernel Documentation Exam | 950,462 | 23/23 answered, 23/23 valid source paths |

Synthetic exact checks:

```text
Distributed retrieval:      50/50
Multi-hop reasoning:        30/30
Adversarial temporal:       10/10

Synthetic total:            90/90
```

The Linux documentation exam is listed separately because the current **23/23** score reflects answer completion and valid in-corpus source attribution rather than a completed source-by-source semantic grade.

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
