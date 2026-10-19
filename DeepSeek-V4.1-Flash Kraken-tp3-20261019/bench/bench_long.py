#!/usr/bin/env python3
"""One near-1M-token needle request with GPU memory sampled every 0.5 s.

Usage: bench_long.py OUT.json [target_words]. ~1.19 tokens per filler word,
so the default 840,000 words is ~1.0M prompt tokens.
"""
import json
import subprocess
import sys
import threading
import time

import bench

samples = []
stop = threading.Event()


def sample_gpus():
    while not stop.is_set():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.split()
        samples.append([time.time()] + [int(x) for x in out])
        time.sleep(0.5)


def main():
    words = int(sys.argv[2]) if len(sys.argv) > 2 else 840000
    prompt = bench.needle_prompt(target_words=words)
    sampler = threading.Thread(target=sample_gpus, daemon=True)
    sampler.start()
    t0 = time.time()
    try:
        r = bench.chat(prompt, 512)
        r["pass"] = "48213-zeta" in r["answer"].lower()
    except Exception as e:  # keep the memory trace even when the request fails
        r = {"error": repr(e)}
    r["wall_s"] = round(time.time() - t0, 1)
    stop.set()
    sampler.join()
    peaks = [max(s[i] for s in samples) for i in range(1, len(samples[0]))]
    r["peak_gpu_mib"] = peaks
    r["prefill_tok_s"] = (round(r["prompt_tokens"] / r["ttft_s"]) if r.get("ttft_s") and r.get("prompt_tokens") else None)
    print(json.dumps({k: v for k, v in r.items() if k != "answer"}), "answer:", repr(r.get("answer", "")[:60]), flush=True)
    json.dump({"result": r, "samples": samples}, open(sys.argv[1], "w"))


if __name__ == "__main__":
    main()
