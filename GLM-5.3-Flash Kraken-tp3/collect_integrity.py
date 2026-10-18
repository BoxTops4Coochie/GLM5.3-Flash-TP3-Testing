"""Build the long-generation integrity tally straight from the trial outputs.

`long_context.py` refuses to overwrite an existing output directory, so a
restarted sweep silently skips seeds it already ran and their rows never reach
the runner's own json. The output directories are the source of truth, so scan
those instead of trusting any single runner's bookkeeping.

Emits integrity.json: one row per (arm, seed), with the same EMPTY / REPEAT /
THIN verdict the runners apply.
"""
import json, pathlib, re

D = pathlib.Path(__file__).resolve().parent
OUT = D.parent / 'dflash-matched/outputs'

# arm label -> (checkpoint, mode, dcp, prefill backend)
ARMS = {
    'default-mtp3': ('default', 'mtp', 1, 'b12x'),
    'default-dflash2': ('default', 'dflash2', 1, 'b12x'),
    'uncensored-mtp3': ('uncensored', 'mtp', 1, 'b12x'),
    'uncensored-dflash2-flashkda': ('uncensored', 'dflash2', 1, 'flashkda'),
    'uncensored-mtp3-flashkda': ('uncensored', 'mtp', 1, 'flashkda'),
    'default-mtp3-dcp1-flashkda': ('default', 'mtp', 1, 'flashkda'),
    'default-dflash2-dcp1-flashkda': ('default', 'dflash2', 1, 'flashkda'),
    'default-mtp3-dcp3-flashkda': ('default', 'mtp', 3, 'flashkda'),
    'default-dflash2-dcp3-flashkda': ('default', 'dflash2', 3, 'flashkda'),
    'uncensored-mtp3-dcp3-flashkda': ('uncensored', 'mtp', 3, 'flashkda'),
    'uncensored-dflash2-dcp3-flashkda': ('uncensored', 'dflash2', 3, 'flashkda'),
}
SEED = re.compile(r'^(?P<arm>.+)-(?P<seed>2\d\d)$')

rows = []
for d in sorted(OUT.iterdir()):
    m = SEED.match(d.name)
    if not m or m.group('arm') not in ARMS:
        continue
    f = d / 'result.json'
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    ckpt, mode, dcp, prefill = ARMS[m.group('arm')]
    q = r.get('quality', {}).get('content', {})
    qr = r.get('quality', {}).get('reasoning', {}) or {}
    c = r.get('content', '')
    # Runaway reasoning is its own mode: the model loops inside the thinking
    # channel and runs to the token cap, so `content` is empty for a reason
    # entirely unlike a clean end of turn. Scoring only the answer channel
    # conflated the two. finish_reason 'length' means truncated, not finished.
    finish = (r.get('finish_reasons') or [None])[0]
    reasoning_repeat = qr.get('repeat_8gram', 0.0)
    truncated = finish == 'length'
    # <|observation|> means the model issued a tool call and is waiting for its
    # result. The 826K prompt is an agentic coding transcript, so the model
    # sometimes continues the pattern and writes its answer into a `write` tool
    # call instead of the answer channel. long_context.py supplies no `tools`,
    # so those tokens parse into neither content nor tool_calls and the response
    # looks empty. That is a property of the test prompt, not of generation
    # integrity, so it is scored separately rather than as a failure.
    # <|user|> (154827) is the normal end of turn.
    tool_call = 154829 in (r.get('stop_reasons') or [])
    empty = (not c.strip()) and not tool_call
    verdict = dict(
        truncated=truncated,
        reasoning_loop=reasoning_repeat > 0.05 and truncated,
        empty=empty and not truncated,
        repeat=q.get('repeat_8gram', 0.0) > 0.05,
        # An empty answer is already its own verdict; do not also call it thin.
        thin=(not tool_call) and not empty and len(c) < 2000
        and r['usage']['completion_tokens'] > 1500)
    rows.append(dict(
        arm=m.group('arm'), checkpoint=ckpt, mode=mode, dcp=dcp, prefill=prefill,
        tool_call=tool_call, finish=finish,
        reasoning_repeat_8gram=reasoning_repeat,
        reasoning_chars=len(r.get('reasoning', '') or ''),
        seed=int(m.group('seed')), content_chars=len(c),
        completion=r['usage']['completion_tokens'],
        reasoning=r['usage'].get('completion_tokens_details', {}).get('reasoning_tokens'),
        repeat_8gram=q.get('repeat_8gram', 0.0),
        ttr=q.get('worst_window_ttr', 0.0),
        fail=any(verdict.values()), **verdict))

rows.sort(key=lambda r: (r['checkpoint'], r['mode'], r['dcp'], r['prefill'], r['seed']))
(D / 'integrity.json').write_text(json.dumps(rows, indent=2))

groups = {}
for r in rows:
    groups.setdefault((r['checkpoint'], r['mode'], r['dcp'], r['prefill']), []).append(r)
print(f'{"checkpoint":11s} {"mode":8s} dcp {"prefill":9s}  pass  chars range        empties toolcalls')
for k, g in sorted(groups.items()):
    fails = sum(x['fail'] for x in g)
    lo, hi = min(x['content_chars'] for x in g), max(x['content_chars'] for x in g)
    e = sum(x['empty'] for x in g); tc = sum(x['tool_call'] for x in g)
    ok = [x for x in g if not x['fail'] and not x['tool_call']]
    lo, hi = (min(x['content_chars'] for x in ok), max(x['content_chars'] for x in ok)) if ok else (0, 0)
    print(f'{k[0]:11s} {k[1]:8s}  {k[2]}  {k[3]:9s}  {len(g)-fails}/{len(g)}  '
          f'{lo:>7,}-{hi:>7,}  {e}      {tc}')
print(f'\n{len(rows)} trials -> {D / "integrity.json"}')
