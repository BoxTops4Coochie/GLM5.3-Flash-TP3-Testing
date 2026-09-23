#!/usr/bin/env python3
"""Translate the R34-style environment into the pinned Kraken native launcher."""
import json
import os
import re
import sys

CHECKPOINTS = {
    'default': ('local-inference-lab/GLM-5.3-Flash-NVFP4', '175ae8ce3b5af842b0d0140dbeb43e9cfc557c49'),
    'uncensored': ('orcarouter/GLM-5.3-Flash-Uncensored-NVFP4', 'ec0adf4f49c9570807cc11a5f650538c1893ae54'),
}


def build_args(env):
    def value(key, default):
        return env.get(key) or default

    def positive(key, default):
        text = value(key, str(default))
        if not re.fullmatch(r'[1-9][0-9]*', text):
            raise ValueError(f'{key} must be a positive integer')
        return int(text)

    checkpoint = value('CHECKPOINT', 'default')
    if checkpoint not in CHECKPOINTS:
        raise ValueError('CHECKPOINT must be default or uncensored')
    model, revision = CHECKPOINTS[checkpoint]
    for key, expected in [('MODEL', model), ('MODEL_REVISION', revision)]:
        if env.get(key) and env[key] != expected:
            raise ValueError(f'{key} disagrees with CHECKPOINT={checkpoint}: expected {expected}')
    mode = value('MODE', 'dflash2')
    mode = {'dflash': 'dflash2', 'mtp0': 'off'}.get(mode, mode)
    if mode not in ('mtp', 'dflash2', 'off'):
        raise ValueError('MODE must be mtp, dflash2 (or dflash), or mtp0 (or off)')
    dcp = value('DCP', '1')
    if dcp not in ('1', '3'):
        raise ValueError('DCP must be 1 or 3')
    reasoning = value('REASONING_EFFORT', 'max')
    clear = value('CLEAR_THINKING', 'true').lower()
    if reasoning not in ('low', 'high', 'max') or clear not in ('true', 'false'):
        raise ValueError('REASONING_EFFORT must be low/high/max; CLEAR_THINKING must be true/false')
    maximum = positive('MAX_CUDAGRAPH_CAPTURE_SIZE', 64 if mode == 'dflash2' else 32)
    sizes = value('CUDAGRAPH_CAPTURE_SIZES', '')
    if not sizes:
        sizes = [n for n in (1, 2, 4, 8, 16, 32, 40, 48, 64, 96, 128, 192, 256) if n < maximum]
        # Exact intermediate batches: measured at default/DCP1/graph32/8 slots.
        # Other checkpoints, DCP3, custom maxima and explicit ladders retain their policy.
        if checkpoint == 'default' and dcp == '1' and maximum == 32 and positive('MAX_NUM_SEQS', 8) == 8:
            if mode == 'off':
                sizes = [1, 2, 3, 4, 5, 6, 7, 8, 16]
            elif mode == 'mtp':
                sizes = [1, 2, 4, 8, 12, 16, 20, 24, 28]
        # Preserve the measured default ladder (DFlash graph64 excludes40/48).
        if maximum == 64:
            sizes = [1, 2, 4, 8, 16, 32]
        sizes.append(maximum)
    else:
        if not all(re.fullmatch(r'[1-9][0-9]*', s) for s in sizes.split()):
            raise ValueError('CUDAGRAPH_CAPTURE_SIZES must be a space-separated list of positive integers')
        sizes = [int(s) for s in sizes.split()]
        if sizes != sorted(set(sizes)) or sizes[-1] != maximum:
            raise ValueError('Capture sizes must be increasing, unique, and end at the graph maximum')
    args = ['--profile', 'glm53-flash', '--hardware', 'rtx-pro-6000-pcie',
            '--model', model, '--revision', revision, '--mode', mode,
            '--tensor-parallel-size', '3', '--decode-context-parallel-size', dcp,
            '--enable-expert-parallel', '--moe-backend', 'marlin' if checkpoint == 'uncensored' else 'auto',
            '--max-num-seqs', str(positive('MAX_NUM_SEQS', 8)),
            '--max-model-len', str(positive('MAX_MODEL_LEN', 1048576)),
            '--gpu-memory-utilization', value('GPU_MEMORY_UTILIZATION', '0.95'),
            '--max-num-batched-tokens', str(positive('MAX_NUM_BATCHED_TOKENS', 4096)),
            '--max-cudagraph-capture-size', str(maximum),
            '--cache-mode', 'vram', '--enable-flashinfer-autotune',
            '--served-model-name', value('SERVED_MODEL_NAME', 'GLM-5.3-Flash-TP3'),
            '--default-chat-template-kwargs', json.dumps({'reasoning_effort': reasoning, 'clear_thinking': clear == 'true'}),
            '--override-generation-config', '{"temperature":1.0,"top_p":0.95}',
            '--mm-encoder-tp-mode', 'weights']
    args += ['--cudagraph-capture-sizes', *map(str, sizes)]
    if checkpoint == 'uncensored':
        args += ['--quantization', 'compressed-tensors', '--load-format', 'auto']
    if mode == 'mtp':
        spec = dict(method='mtp', num_speculative_tokens=3, revision=revision,
                    draft_sample_method='probabilistic', rejection_sample_method='standard',
                    moe_backend='triton' if checkpoint == 'uncensored' else 'marlin', attention_backend='B12X')
        # Acceptance-length adaptation: average accepted draft lengths over N
        # verification steps and trim the speculative-token count, with
        # num_speculative_tokens as the upper bound. Empty leaves it disabled,
        # which is the qualified configuration. Under evaluation at C8, where
        # acceptance is ~2.5 of a possible 4; it measured 0.0% at C1, as
        # expected, since the controller never trims there.
        window = value('MTP_ADAPTIVE_WINDOW', '')
        if window:
            if not re.fullmatch(r'[1-9][0-9]*', window):
                raise ValueError('MTP_ADAPTIVE_WINDOW must be a positive integer')
            spec['adaptive_speculative_tokens_window'] = int(window)
        args += ['--speculative-config', json.dumps(spec)]
    elif mode == 'dflash2':
        draft = value('DFLASH_MODEL', 'local-inference-lab/GLM-5.3-Flash-DFlash2')
        draft_revision = value('DFLASH_MODEL_REVISION', '713226ab03bc38afdf955c7450436c2f7176f6f8' if draft == 'local-inference-lab/GLM-5.3-Flash-DFlash2' else '')
        if not re.fullmatch('[0-9a-f]{40}', draft_revision):
            raise ValueError('A custom DFLASH_MODEL requires its immutable DFLASH_MODEL_REVISION')
        args += ['--draft-tokens', '7', '--draft-model', draft, '--draft-revision', draft_revision]
    return args


def main():
    # Preserve historical command-list overrides used by the recorded benchmarks.
    extra = sys.argv[1:]
    if '--profile' in extra:
        args = extra
    else:
        if any(x != '--print-config' for x in extra):
            raise ValueError('Configure through environment variables; only --print-config is accepted')
        args = build_args(os.environ) + extra
    # These belong to this wrapper; native Kraken explicitly rejects MODE.
    for key in ("CHECKPOINT", "MODE", "MODEL", "MODEL_REVISION", "DFLASH_MODEL",
                "DFLASH_MODEL_REVISION", "DCP", "MAX_NUM_BATCHED_TOKENS",
                "MAX_CUDAGRAPH_CAPTURE_SIZE", "CUDAGRAPH_CAPTURE_SIZES",
                "MAX_NUM_SEQS", "MAX_MODEL_LEN", "GPU_MEMORY_UTILIZATION",
                "REASONING_EFFORT", "CLEAR_THINKING", "SERVED_MODEL_NAME"):
        os.environ.pop(key, None)
    # Blank optional environment entries mean use the native defaults.
    for key in list(os.environ):
        if os.environ[key] == '':
            del os.environ[key]
    os.execv('/usr/local/bin/lil-entrypoint', ['/usr/local/bin/lil-entrypoint', *args])


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        sys.exit(str(exc))
