"""Consolidate per-variation 0/32K/128K prefill + decode + retrieval evidence.

Sources, all from qualification-20260919 (8 configurations):
  *-decode.log      C1/C8 decode at contexts 0 / 32K / 128K
  *-prefill.json    standalone cold prefill at 8K / 32K / 64K / 128K
  *-retrieval.log   exact-match retrieval at 128K and 900K
  *-smoke.log       8/8 exact arithmetic, natural EOS

Long-generation integrity comes from the 20260920 trials and is merged in by
run label, not read here.
"""
import ast, json, pathlib, re

_ROOT = pathlib.Path(__file__).resolve().parents[1].parent
# Prefer the flashkda re-measurement; fall back to the superseded b12x-prefill
# run only for arms it has not reached yet.
FK = _ROOT / 'qualification-20260920/flashkda-remeasure'
B12X = _ROOT / 'qualification-20260919'


class _Sources:
    """Resolve each artefact to the newest run that produced it."""

    def __truediv__(self, name):
        cand = FK / name
        return cand if cand.exists() else B12X / name


Q = _Sources()
CONFIGS = [f'{c}-{m}-dcp{d}' for c in ('default', 'uncensored')
           for m in ('mtp', 'dflash2') for d in (1, 3)]
DECODE = re.compile(r'\bC (\d+) (\[\(.*\)\])\s*$')


def decode(label):
    """{concurrency: {context: (tok/s, steps/s, accept_len)}}"""
    out = {}
    for line in (Q / f'{label}-decode.log').read_text().splitlines():
        m = DECODE.search(line.strip())
        if not m:
            continue
        out[int(m.group(1))] = {row[0]: tuple(row[1:]) for row in ast.literal_eval(m.group(2))}
    return out


def prefill(label):
    d = json.loads((Q / f'{label}-prefill.json').read_text())['prefill']
    return {int(k): (v['tok_per_sec'], v['ttft_seconds'], v['prompt_tokens']) for k, v in d.items()}


def retrieval(label):
    # The re-measurement sweep's retrieval stage aborted on an output-directory
    # collision, so seven arms were recovered by a separate retrieval pass that
    # writes `-retrieval2.log`. The topped-up eighth arm has the plain name.
    for name in (f'{label}-retrieval2.log', f'{label}-retrieval.log'):
        p = FK / name
        if p.exists():
            return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    text = (B12X / f'{label}-retrieval.log').read_text()
    return [l.strip() for l in text.splitlines() if l.strip()]


def smoke(label):
    return (Q / f'{label}-smoke.log').read_text().strip().splitlines()[0]


rows = {}
for label in CONFIGS:
    try:
        rows[label] = dict(decode=decode(label), prefill=prefill(label),
                           retrieval=retrieval(label), smoke=smoke(label))
    except FileNotFoundError as exc:
        rows[label] = dict(missing=str(exc))

out = pathlib.Path(__file__).resolve().parent / 'variation-stats.json'
out.write_text(json.dumps(rows, indent=2))

for label in CONFIGS:
    r = rows[label]
    if 'missing' in r:
        print(f'{label}: MISSING {r["missing"]}'); continue
    c1 = r['decode'].get(1, {}); c8 = r['decode'].get(8, {})
    pf = r['prefill']
    print(f'\n## {label}')
    print(f'  smoke: {r["smoke"]}')
    print('  decode tok/s (steps/s, accept)  ctx:      0        32K       128K')
    for c, cells in (('C1', c1), ('C8', c8)):
        if not cells:
            continue
        s = '  '.join(f'{cells[k][0]:7.1f} ({cells[k][1]:5.1f}, {cells[k][2]:.2f})'
                      for k in (0, 32768, 131072) if k in cells)
        print(f'  {c}  {s}')
    print('  prefill tok/s: ' + '  '.join(f'{k//1024}K={pf[k][0]:.0f}' for k in sorted(pf)))
    for line in r['retrieval'][-4:]:
        print(f'  retrieval: {line}')
print(f'\nWrote {out}')
