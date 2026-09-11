"""Natural-EOS long-context retrieval with exact known answers at three positions."""
import hashlib
import argparse
import json
import re
from pathlib import Path
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/long-retrieval'
OUT.mkdir(exist_ok=True)
URL = 'http://127.0.0.1:15015'
unit = 'This ordinary archived entry contains no requested marker.\n'
expected = {'A': 'ALPHA-R34-7K2M9Q', 'B': 'BETA-R34-8P6V3X', 'C': 'GAMMA-R34-4T9N2Z'}


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def messages(n):
    text = ('Read this archive and retrieve its three marker values.\n'
            'Marker A = ' + expected['A'] + '\n' + unit * (n//2)
            + 'Marker B = ' + expected['B'] + '\n' + unit * (n-n//2)
            + 'Marker C = ' + expected['C'] + '\n'
            'Return only a JSON object with keys A, B, C and their exact marker values.')
    return [{'role': 'user', 'content': text}]


parser = argparse.ArgumentParser()
parser.add_argument('--targets', nargs='+', type=int, default=[128000, 900000])
args = parser.parse_args()
for target in args.targets:
    subprocess.run([str(HERE.parent / 'r27-port/.venv/bin/python'), str(HERE / 'wait-idle.py')], check=True)
    n = target // 10
    attempts = []
    for _ in range(8):
        msgs = messages(n)
        tokenized = post('/tokenize', dict(model='GLM-5.3-Flash-TP3', messages=msgs,
            add_generation_prompt=True, chat_template_kwargs={'reasoning_effort':'high', 'clear_thinking':False}))
        count = tokenized['count']
        attempts.append({'records': n, 'tokens': count})
        if abs(count-target) < 100:
            break
        n = max(1, round(n * target/count))
    assert abs(count-target) < 100, attempts
    body = dict(model='GLM-5.3-Flash-TP3', messages=msgs, temperature=1, top_p=.95,
        seed=target, min_tokens=0, ignore_eos=False, max_completion_tokens=2048,
        repetition_penalty=1, tool_choice='none', cache_salt=f'r34-exact-retrieval-{target}',
        chat_template_kwargs={'reasoning_effort':'high', 'clear_thinking':False})
    request_bytes = json.dumps(body).encode()
    out = OUT / str(target)
    out.mkdir(exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(dict(target=target,
        request_sha256=hashlib.sha256(request_bytes).hexdigest(), attempts=attempts,
        settings={k:v for k,v in body.items() if k!='messages'}, expected=expected,
        runtime=json.loads(subprocess.check_output(['docker','inspect','glm53-r34-tp3']))[0]), indent=2))
    print('START',target,'actual prompt',count,flush=True)
    response = post('/v1/chat/completions', body)
    (out/'response.json').write_text(json.dumps(response,indent=2))
    choice = response['choices'][0]
    text = choice['message'].get('content') or ''
    strict_json = True
    try:
        actual = json.loads(text)
    except ValueError:
        strict_json = False
        fenced = re.fullmatch(r'```(?:json)?\s*\n(.*)\n```', text.strip(), re.DOTALL)
        try:
            actual = json.loads(fenced.group(1)) if fenced else None
        except ValueError:
            actual = None
    passed = (actual == expected and choice['finish_reason']=='stop'
              and not choice['message'].get('tool_calls')
              and response['usage']['prompt_tokens']==count)
    receipt = dict(passed=passed,strict_json=strict_json,actual=actual,expected=expected,usage=response['usage'],
                   finish_reason=choice['finish_reason'])
    (out/'result.json').write_text(json.dumps(receipt,indent=2))
    print('DONE',target,json.dumps(receipt),flush=True)
    assert passed, receipt
