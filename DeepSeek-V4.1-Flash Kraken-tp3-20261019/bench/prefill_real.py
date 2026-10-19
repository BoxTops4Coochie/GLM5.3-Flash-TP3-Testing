#!/usr/bin/env python3
"""Time-to-first-token on real source-code prompts (not synthetic filler).

Usage: prefill_real.py OUT.json ROOT   ROOT: a directory of .py files (the release numbers
used a FlashInfer checkout at 5b065abe: ROOT=flashinfer/flashinfer). Each run uses a different slice of the
corpus, so no prompt repeats within a run.
"""
import glob, json, sys, time, urllib.request
URL = "http://localhost:15015/v1/chat/completions"
root = sys.argv[2]
text = "\n".join(open(f, errors="ignore").read() for f in sorted(glob.glob(root + "/**/*.py", recursive=True)))
def ttft(prompt):
    body = json.dumps({"model": "DeepSeek-V4.1-Flash", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 8, "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if first is None and d.get("choices") and d["choices"][0]["delta"].get("content") is not None:
                first = time.perf_counter() - t0
            if first is None and d.get("choices") and d["choices"][0]["delta"].get("reasoning_content") is not None:
                first = time.perf_counter() - t0
            usage = d.get("usage") or usage
    return first, usage["prompt_tokens"]
out = []
offset = 0
for chars in (400_000, 1_000_000, 400_000):
    p = text[offset:offset + chars] + "\n\nSummarize what this code does in five bullet points."
    offset += chars
    t, n = ttft(p)
    out.append({"chars": chars, "prompt_tokens": n, "ttft_s": round(t, 2), "tok_s": round(n / t)})
    print(out[-1], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
