"""Run the synthetic vision set against the server on :15015. Usage: run_vision.py LABEL"""
import base64, concurrent.futures, json, pathlib, re, sys, time, urllib.request
D = pathlib.Path(__file__).resolve().parent; S = D / (sys.argv[2] if len(sys.argv) > 2 else 'set'); label = sys.argv[1]
items = json.load(open(S / 'answers.json'))
def norm(s):
    s = (s or '').strip().lower()
    s = re.sub(r'[*`"\'.]', '', s); s = re.sub(r'\s+', ' ', s)
    return s.replace(' - ', '-').strip()
def ask(item, temp):
    u = 'data:image/png;base64,' + base64.b64encode((S / item['file']).read_bytes()).decode()
    body = dict(model='DeepSeek-V4.1-Flash', max_tokens=16000, temperature=temp, top_p=0.95 if temp else 1.0, seed=7,
                messages=[dict(role='user', content=[dict(type='image_url', image_url=dict(url=u)), dict(type='text', text=item['question'])])])
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:15015/v1/chat/completions',
            json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=1200))
        out = r['choices'][0]['message'].get('content') or ''; usage = r['usage']['completion_tokens']; fin = r['choices'][0]['finish_reason']
    except Exception as e:
        out, usage, fin = f'ERROR {e!r}'[:200], 0, 'error'
    got = norm(out.splitlines()[-1] if out.strip() else '')
    ok = got == norm(item['answer']) or norm(out) == norm(item['answer'])
    return dict(id=item['id'], kind=item['kind'], temp=temp, answer=item['answer'], output=out[-200:], ok=ok, tokens=usage, finish=fin, secs=round(time.time() - t, 1))
res = []
for temp in (0, 1):
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        res += list(ex.map(lambda it: ask(it, temp), items))
json.dump(res, open(D / f'results-{label}-{S.name}.json', 'w'), indent=1)
kinds = sorted({r['kind'] for r in res})
print(f'== {label} [{S.name}]')
for temp in (0, 1):
    rr = [r for r in res if r['temp'] == temp]
    print(f'T={temp}: total {sum(r["ok"] for r in rr)}/{len(rr)} | ' + ' '.join(f'{k} {sum(r["ok"] for r in rr if r["kind"]==k)}/{sum(1 for r in rr if r["kind"]==k)}' for k in kinds))
for r in res:
    if not r['ok']: print('  MISS', r['temp'], r['id'], 'expected', repr(r['answer']), 'got', repr(r['output'][-60:]), r['finish'])
