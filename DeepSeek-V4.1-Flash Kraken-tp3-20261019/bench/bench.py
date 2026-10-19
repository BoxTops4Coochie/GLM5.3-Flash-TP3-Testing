#!/usr/bin/env python3
"""Fixed DS4.1 TP3 test set, identical for every image/weights run.

Writes one JSON result to argv[1]. Checks: known-answer correctness, two
1,500-token decode runs, a ~32K-token needle retrieval (TTFT + recall), and
four concurrent requests.
"""
import json
import os
import random
import sys
import threading
import time
import urllib.request

URL = "http://localhost:15015/v1/chat/completions"
MODEL = "DeepSeek-V4.1-Flash"
# BENCH_GREEDY=1: temperature 0, so every run of a prompt yields the same text
# and MTP acceptance; speed differences then reflect the configuration only.
GREEDY = os.environ.get("BENCH_GREEDY", "0") == "1"
DECODE_RUNS = 3 if GREEDY else 2


def spec_counters():
    """(verification steps, accepted draft tokens) from the server's metrics."""
    try:
        text = urllib.request.urlopen(URL.replace("/v1/chat/completions", "/metrics"), timeout=10).read().decode()
    except OSError:
        return None
    vals = {}
    for line in text.splitlines():
        for key in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(key + "{"):
                vals[key] = float(line.rsplit(" ", 1)[1])
    if len(vals) != 2:
        return None
    return vals["vllm:spec_decode_num_drafts_total"], vals["vllm:spec_decode_num_accepted_tokens_total"]


def chat(prompt, max_tokens, effort="low"):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    if GREEDY:
        body["temperature"] = 0.0
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    text = reasoning = ""
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                r_piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                c_piece = delta.get("content") or ""
                if (r_piece or c_piece) and first is None:
                    first = time.time()
                reasoning += r_piece
                text += c_piece
    t1 = time.time()
    n = usage.get("completion_tokens", 0)
    decode = (n - 1) / (t1 - first) if first and n > 1 and t1 > first else None
    return {
        "ttft_s": round(first - t0, 3) if first else None,
        "completion_tokens": n,
        "prompt_tokens": usage.get("prompt_tokens"),
        "decode_tok_s": round(decode, 1) if decode else None,
        "answer": text.strip(),
        "reasoning_chars": len(reasoning),
    }


KNOWN = [
    ("What is 17*23? Reply with just the number.", ["391"]),
    ("What is the capital of Australia? One word.", ["canberra"]),
    ("What is the chemical symbol for gold? Reply with just the symbol.", ["au"]),
    ("If a train leaves at 14:40 and the trip takes 2 hours 35 minutes, when does it arrive? Reply as HH:MM.", ["17:15"]),
    ("Spell the word 'necessary' backwards. Reply with only the result.", ["yrassecen"]),
]

DECODE_PROMPT = (
    "Write a detailed 600-word explanation of how TCP congestion control works, covering slow start, "
    "congestion avoidance, fast retransmit and fast recovery."
)


def needle_prompt(target_words=24000, seed=7):
    rng = random.Random(seed)
    words = ("river stone lantern copper meadow signal harbor quiet amber window falcon garden "
             "orbit cedar velvet engine marble canyon ribbon summit glacier pocket thunder violet").split()
    filler = [" ".join(rng.choice(words) for _ in range(12)) + "." for _ in range(target_words // 12)]
    pos = int(len(filler) * 0.63)
    filler.insert(pos, "The secret vault code for the Halvorsen archive is 48213-ZETA.")
    return ("Read the following document carefully.\n\n" + "\n".join(filler) +
            "\n\nQuestion: What is the secret vault code for the Halvorsen archive? Reply with only the code.")


def main():
    out = {"known": [], "decode": [], "needle": None, "concurrent": None}
    for prompt, accept in KNOWN:
        r = chat(prompt, 2048)
        r["pass"] = any(a in r["answer"].lower().replace(" ", "") for a in accept)
        r["prompt"] = prompt
        out["known"].append(r)
        print("known", r["pass"], repr(r["answer"][:60]), flush=True)
    for _ in range(DECODE_RUNS):
        before = spec_counters()
        r = chat(DECODE_PROMPT, 1500)
        after = spec_counters()
        r.pop("answer")
        # Steps/s is the engine speed independent of which drafts were accepted;
        # tok/s = steps/s x tokens per step.
        if before and after and after[0] > before[0] and r["decode_tok_s"]:
            steps = after[0] - before[0]
            seconds = (r["completion_tokens"] - 1) / r["decode_tok_s"]
            r["steps_per_s"] = round(steps / seconds, 2)
            r["tokens_per_step"] = round(r["completion_tokens"] / steps, 3)
        out["decode"].append(r)
        print("decode", r, flush=True)
    r = chat(needle_prompt(), 2048)
    r["pass"] = "48213-zeta" in r["answer"].lower()
    out["needle"] = r
    print("needle", r["pass"], r["ttft_s"], r["prompt_tokens"], repr(r["answer"][:60]), flush=True)
    results = [None] * 4

    def worker(i):
        try:
            results[i] = chat(DECODE_PROMPT, 800)
            results[i].pop("answer")
        except Exception as e:  # recorded, not raised: one failure must not hide the others
            results[i] = {"error": repr(e)}

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    toks = sum(r.get("completion_tokens", 0) for r in results)
    out["concurrent"] = {"requests": results, "wall_s": round(wall, 1),
                         "aggregate_tok_s": round(toks / wall, 1),
                         "errors": sum("error" in r for r in results)}
    print("concurrent", out["concurrent"]["aggregate_tok_s"], "agg tok/s, errors", out["concurrent"]["errors"], flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
