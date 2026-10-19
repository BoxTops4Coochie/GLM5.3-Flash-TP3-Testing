#!/usr/bin/env python3
"""Time-to-first-token of a short request that arrives during another request's long prefill.

A ~420K-token prompt starts; DELAY seconds later a ~135-token request arrives. Reports both
TTFTs (streaming, first content or reasoning token). Prompts carry unique tags, so the prefix
cache never serves them. Usage: prefill_fairness.py OUT.json [RUNS] [DELAY_S]
"""
import json, random, sys, threading, time, urllib.request
URL = "http://127.0.0.1:15015/v1/chat/completions"
WORDS = "alpha river stone cloud maple ember quartz harbor violet summit lantern meadow copper falcon willow glacier".split()
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
DELAY = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
LONG_WORDS = 373_000  # ~1.125 tokens per word -> ~420K tokens


def ttft(prompt, max_tokens, out, key):
    body = json.dumps({"model": "DeepSeek-V4.1-Flash", "stream": True, "max_tokens": max_tokens,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    t0 = time.perf_counter(); first = None; usage = None
    with urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            ch = d.get("choices") or []
            delta = ch[0]["delta"] if ch else {}
            if first is None and any(delta.get(k) for k in ("content", "reasoning", "reasoning_content")):
                first = time.perf_counter() - t0
            usage = d.get("usage") or usage
    out[key] = {"ttft_s": round(first, 2), "prompt_tokens": usage["prompt_tokens"]}


results = []
for run in range(RUNS):
    rng = random.Random(run)
    tag = f"[{time.time_ns()}] "
    long_prompt = tag + " ".join(rng.choice(WORDS) for _ in range(LONG_WORDS)) + "\nSummarize in one line."
    short_prompt = tag + "Explain in two sentences why the sky looks blue during the day and red at sunset, " \
        "covering Rayleigh scattering, the path length of sunlight through the atmosphere, and why " \
        "violet light is not what we mostly perceive even though it scatters more strongly than blue."
    out = {}
    t_long = threading.Thread(target=ttft, args=(long_prompt, 16, out, "long"))
    t_long.start(); time.sleep(DELAY)
    ttft(short_prompt, 16, out, "short")
    t_long.join()
    results.append(out)
    print(f"run {run}: short {out['short']['prompt_tokens']} tok TTFT {out['short']['ttft_s']} s | "
          f"long {out['long']['prompt_tokens']} tok TTFT {out['long']['ttft_s']} s", flush=True)
    time.sleep(5)
json.dump(results, open(sys.argv[1], "w"), indent=1)
