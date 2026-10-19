#!/usr/bin/env python3
"""Single-stream benchmark following the protocol in meldavy/dsv41f-1017-sm120-tp3.

That README reports prefill and decode for DeepSeek-V4.1-Flash-1017 at TP3 with
thinking off, one stream, and these cells:
  - prefill at ~10k and ~32k prompt tokens (synthetic filler), tok/s = prompt / TTFT
  - decode after those prompts, 512 output tokens
  - decode on four realistic 600-token outputs (Python module + tests, Rust parser,
    TCP explainer, refactor), measured warm
Its benchmark scripts are not published, so the prompts here are written to the
same description. Each cell runs once to warm up, then RUNS measured times; the
output reports min-max like the README.

Usage: meldavy_bench.py OUT.json [RUNS]
"""
import json
import random
import sys
import time
import urllib.request

URL = "http://localhost:15015/v1/chat/completions"
MODEL = "DeepSeek-V4.1-Flash"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
WORDS = (
    "the archive of the northern valley records harvests storms bridges and the slow "
    "work of rebuilding roads after each winter while merchants trade wool grain and "
    "iron along the river"
).split()
REALISTIC = {
    "python_module_tests": "Write a Python module implementing an LRU cache with TTL expiry, "
    "including type hints, docstrings, and a pytest test suite covering eviction and expiry.",
    "rust_parser": "Write a Rust parser for a small arithmetic expression language with "
    "integers, + - * /, parentheses and variables, using a recursive-descent design and an AST.",
    "tcp_explainer": "Explain how TCP establishes a connection, controls congestion and "
    "recovers from packet loss, with concrete examples of the three-way handshake and slow start.",
    "refactor": "Refactor this Python into clean, idiomatic code with type hints and explain "
    "each change:\n\ndef proc(d):\n    r = []\n    for k in d.keys():\n        if d[k] != None:\n"
    "            if type(d[k]) == int:\n                r.append(str(k) + '=' + str(d[k] * 2))\n"
    "            else:\n                r.append(str(k) + '=' + str(d[k]))\n    return ','.join(r)",
}


def filler(tokens, seed):
    rng = random.Random(seed)
    # A unique leading tag keeps every prompt out of the server's prefix cache,
    # so TTFT always measures a full prefill.
    tag = f"[{time.time_ns()}-{seed}] "
    # ~0.97 words per token for this vocabulary (measured on the DS4.1 tokenizer).
    return tag + " ".join(rng.choice(WORDS) for _ in range(int(tokens * 0.967)))


def stream(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            choices = data.get("choices") or []
            if first is None and choices and (
                choices[0]["delta"].get("content") or choices[0]["delta"].get("reasoning_content")
            ):
                first = time.perf_counter() - t0
            usage = data.get("usage") or usage
    total = time.perf_counter() - t0
    out = usage["completion_tokens"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": out,
        "ttft_s": round(first, 3),
        "prefill_tok_s": round(usage["prompt_tokens"] / first),
        "decode_tok_s": round((out - 1) / (total - first), 1),
    }


def cell(name, prompt_fn, max_tokens, results):
    stream(prompt_fn(0), max_tokens)  # warm-up
    runs = [stream(prompt_fn(i + 1), max_tokens) for i in range(RUNS)]
    results[name] = runs
    pre = [r["prefill_tok_s"] for r in runs]
    dec = [r["decode_tok_s"] for r in runs]
    print(f"{name:22s} prompt {runs[0]['prompt_tokens']:>6} tok | prefill {min(pre):,}-{max(pre):,} tok/s"
          f" | decode {min(dec)}-{max(dec)} tok/s", flush=True)


def main():
    results = {}
    for ctx in (10_000, 32_000):
        cell(f"filler_{ctx // 1000}k", lambda s, c=ctx: filler(c, s) + "\n\nContinue this text.", 512, results)
    for name, prompt in REALISTIC.items():
        cell(name, lambda s, p=prompt: f"[{time.time_ns()}] " + p, 600, results)
    json.dump(results, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
