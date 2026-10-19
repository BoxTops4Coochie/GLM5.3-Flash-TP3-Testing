"""Simultaneous requests filling the KV pool: LONG x ~891K-token prompts (LONG_DECODE tokens each)
plus SHORT x ~14K-token prompts (SHORT_DECODE each), ignore_eos, distinct prompts. GPU memory every
200 ms, vLLM KV usage / running / waiting / preemptions every 5 s."""

import json, random, subprocess, threading, time, urllib.request, re, concurrent.futures, pathlib
D = pathlib.Path(__file__).resolve().parent
URL = 'http://127.0.0.1:15015'
WORDS = 'alpha river stone cloud maple ember quartz harbor violet summit lantern meadow copper falcon willow glacier'.split()
stop = False

def sample_mem():
    p = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used', '--format=csv,noheader,nounits',
                          '--loop-ms=200'], stdout=open(D / 'stress-mem.csv', 'w'))
    while not stop: time.sleep(1)
    p.terminate()

def sample_kv():
    with open(D / 'stress-kv.log', 'w') as f:
        while not stop:
            try:
                m = urllib.request.urlopen(URL + '/metrics', timeout=10).read().decode()
                vals = {k: re.search(rf'^{k}\{{[^}}]*\}} ([0-9.e+-]+)', m, re.M) for k in
                        ('vllm:kv_cache_usage_perc', 'vllm:num_requests_running', 'vllm:num_requests_waiting',
                         'vllm:num_preemptions_total')}
                f.write(json.dumps({'t': time.time(), **{k.split(':')[1]: (float(v.group(1)) if v else None) for k, v in vals.items()}}) + '\n'); f.flush()
            except Exception as e:
                f.write(json.dumps({'t': time.time(), 'err': repr(e)}) + '\n')
            time.sleep(5)

def req(i, nwords, max_tokens):
    rnd = random.Random(1000 + i)
    text = ' '.join(rnd.choice(WORDS) for _ in range(nwords))
    body = dict(model='DeepSeek-V4.1-Flash', max_tokens=max_tokens, ignore_eos=True, temperature=1, top_p=0.95,
                messages=[dict(role='user', content=f'Request {i}. ' + text + '\nHow many times does the word "falcon" appear? Guess briefly.')])
    t = time.time()
    r = urllib.request.Request(URL + '/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(r, timeout=7200) as resp: d = json.load(resp)
        return dict(i=i, ok=True, secs=round(time.time() - t), usage=d['usage'], finish=d['choices'][0]['finish_reason'],
                    text=(d['choices'][0]['message'].get('content') or '')[:200])
    except urllib.error.HTTPError as e:
        return dict(i=i, ok=False, secs=round(time.time() - t), err=e.read().decode()[:500])
    except Exception as e:
        return dict(i=i, ok=False, secs=round(time.time() - t), err=repr(e))

threads = [threading.Thread(target=sample_mem, daemon=True), threading.Thread(target=sample_kv, daemon=True)]
for t in threads: t.start()
import os
LONG, SHORT = int(os.environ.get('LONG', '2')), int(os.environ.get('SHORT', '2'))
LONG_WORDS, SHORT_WORDS = int(os.environ.get('LONG_WORDS', '945000')), int(os.environ.get('SHORT_WORDS', '14800'))
LONG_DECODE, SHORT_DECODE = int(os.environ.get('LONG_DECODE', '24000')), int(os.environ.get('SHORT_DECODE', '16000'))
jobs = [(i, LONG_WORDS, LONG_DECODE) for i in range(LONG)] + [(LONG + i, SHORT_WORDS, SHORT_DECODE) for i in range(SHORT)]
with concurrent.futures.ThreadPoolExecutor(len(jobs)) as ex:
    res = list(ex.map(lambda j: req(*j), jobs))
stop = True; time.sleep(2)
json.dump(res, open(D / 'stress-results.json', 'w'), indent=1)
peak = {}
for line in open(D / 'stress-mem.csv'):
    p = [x.strip() for x in line.split(',')]
    if len(p) == 3 and p[1].isdigit(): peak[int(p[1])] = max(peak.get(int(p[1]), 0), int(p[2]))
kv = [json.loads(l) for l in open(D / 'stress-kv.log')]
ok = [r for r in res if r['ok'] and r['usage']['completion_tokens'] == (LONG_DECODE if r['i'] < LONG else SHORT_DECODE)]
print('requests ok', len(ok), '/', len(res))
for r in res:
    if r not in ok: print('FAIL', r)
print('prompt tokens', sum(r['usage']['prompt_tokens'] for r in res if r['ok']), 'secs', max(r['secs'] for r in res))
print('peak KV %', round(100 * max((x.get('kv_cache_usage_perc') or 0) for x in kv), 1),
      'max running', max((x.get('num_requests_running') or 0) for x in kv),
      'max waiting', max((x.get('num_requests_waiting') or 0) for x in kv),
      'preemptions', max((x.get('num_preemptions_total') or 0) for x in kv))
print('min free MiB', {g: 97887 - v for g, v in sorted(peak.items())})
