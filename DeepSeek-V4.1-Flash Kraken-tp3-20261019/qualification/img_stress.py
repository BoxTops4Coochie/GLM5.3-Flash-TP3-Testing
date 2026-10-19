"""NREQ (default 4) concurrent requests x 10 photo-like 6000x4000 JPEGs (distinct per request); GPU memory sampled
every 200 ms. Reports per-request outcome and the minimum free memory per GPU."""
import base64, concurrent.futures, io, json, pathlib, random, subprocess, sys, time, urllib.request
import numpy as np
from PIL import Image, ImageFilter
D = pathlib.Path(__file__).resolve().parent
NREQ, NIMG = int(sys.argv[1]) if len(sys.argv) > 1 else 4, int(sys.argv[2]) if len(sys.argv) > 2 else 10
def photo(seed):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((6000, 4000), Image.BICUBIC)
    noise = rng.integers(-40, 40, (4000, 6000, 3), dtype=np.int16)
    arr = np.clip(np.asarray(img, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    b = io.BytesIO(); Image.fromarray(arr).save(b, 'JPEG', quality=88); return b.getvalue()
print('generating', NREQ * NIMG, 'images', flush=True)
imgs = [[photo(1000 * r + i) for i in range(NIMG)] for r in range(NREQ)]
print('avg JPEG MB', round(sum(len(x) for r in imgs for x in r) / (NREQ * NIMG) / 1e6, 1), flush=True)
mem = subprocess.Popen(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits', '--loop-ms=200'],
                       stdout=open(D / 'img-mem.csv', 'w'))
def req(r):
    content = [dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + base64.b64encode(x).decode())) for x in imgs[r]]
    content.append(dict(type='text', text=f'Request {r}: describe the dominant colors in each of these {NIMG} images in one short line each.'))
    body = dict(model='DeepSeek-V4.1-Flash', max_tokens=3000, temperature=1, top_p=0.95, messages=[dict(role='user', content=content)])
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request('http://localhost:15015/v1/chat/completions',
            json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=3600))
        return dict(r=r, ok=True, secs=round(time.time() - t), usage=d['usage'])
    except urllib.error.HTTPError as e:
        return dict(r=r, ok=False, secs=round(time.time() - t), err=e.read().decode()[:300])
    except Exception as e:
        return dict(r=r, ok=False, secs=round(time.time() - t), err=repr(e)[:300])
with concurrent.futures.ThreadPoolExecutor(NREQ) as ex:
    res = list(ex.map(req, range(NREQ)))
time.sleep(2); mem.terminate()
peak = {}
for line in open(D / 'img-mem.csv'):
    p = [x.strip() for x in line.split(',')]
    if len(p) == 2 and p[0].isdigit(): peak[int(p[0])] = max(peak.get(int(p[0]), 0), int(p[1]))
json.dump(res, open(D / 'img-results.json', 'w'), indent=1)
for x in res: print(x['r'], x['ok'], x['secs'], (x.get('usage') or {}).get('prompt_tokens'), x.get('err', ''))
print('min free MiB', {g: 97887 - v for g, v in peak.items()})
