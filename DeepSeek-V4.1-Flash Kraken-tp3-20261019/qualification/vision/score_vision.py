#!/usr/bin/env python3
"""Re-score run_vision.py results two ways: strict (run_vision's rule: the last line equals the
answer) and extracted (also accept a final 'the answer is X' sentence). Usage: score_vision.py RESULTS.json..."""
import json, re, sys
def norm(s):
    s = (s or '').strip().lower(); s = re.sub(r'[*`"\'.]', '', s); s = re.sub(r'\s+', ' ', s)
    return s.replace(' - ', '-').strip()
def extracted(out):
    m = re.findall(r'answer is[:\s]*\**([^*\n]+?)\**\s*\.?\s*$', out.strip(), re.I | re.M)
    return norm(m[-1]) if m else None
for f in sys.argv[1:]:
    res = json.load(open(f)); kinds = sorted({r['kind'] for r in res})
    print('==', f)
    for temp in (0, 1):
        rr = [r for r in res if r['temp'] == temp]
        ext = [r for r in rr if r['ok'] or extracted(r['output']) == norm(r['answer'])]
        print(f'  T={temp}: strict {sum(r["ok"] for r in rr)}/{len(rr)} | extracted {len(ext)}/{len(rr)} | ' +
              ' '.join(f'{k} {sum(1 for r in ext if r["kind"]==k)}/{sum(1 for r in rr if r["kind"]==k)}' for k in kinds))
        for r in rr:
            if r not in ext: print('    MISS', r['id'], 'expected', repr(r['answer']), 'got', repr(r['output'][-70:]), r['finish'])
