#!/usr/bin/env python3
"""Translate the R34-style environment into the pinned Kraken native launcher."""
import json
import os
import re
import sys

# Explicit KV budget for the one combination it was sized and stress-tested in
# (default/MTP3/DCP1/MOE_TP=1, 8 slots, batch 4096, graph 32): 1779 blocks,
# 3,319,246 tokens; three ~1M contexts resident with 1.5 GB/GPU headroom at peak.
KV_RECLAIM_BYTES = 24707662848
# DCP3/MOE_TP=1 (experimental): 1774 blocks (blocks are 13,888,512 B/GPU),
# 8,417,076 tokens; 8x~895K stress with 6 resident (63.5% KV), 0 preemptions,
# peak 96,364 MiB/GPU (1.5 GB free).
KV_RECLAIM_BYTES_BY_DCP = {'1': KV_RECLAIM_BYTES, '3': 1774 * 13888512}
# With LMCACHE=l1 (DCP1 only): 1720 blocks, 3,180,865 tokens; LMCache's GPU-side
# transfer buffers need ~0.8 GB more headroom (1,779 blocks hangs in autotune).
KV_RECLAIM_BYTES_L1 = {'1': 1720 * 13888512}
# Uncensored checkpoint, MOE_TP=1 (Marlin TP experts, experimental), no LMCache:
# DCP1 1721 blocks = 3,211,030 tokens (3x~895K resident, 86.8% KV, peak 1.85 GB free);
# DCP3 1770 blocks = 8,398,097 tokens, 8.01x 1M (8x895K stress, 5 resident, 0
# preemptions, peak 1.39 GB free; 1686 blocks also passed).
KV_RECLAIM_BYTES_UNCENSORED = {'1': 1721 * 13888512, '3': 1770 * 13888512}

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
    moe_tp = value('MOE_TP', '0')
    if moe_tp not in ('0', '1'):
        raise ValueError('MOE_TP must be 0 or 1')
    # MOE_TP=1 is qualified at DCP1; DCP3 is an experimental combination.
    # MOE_TP=1 is qualified for default/MTP/DCP1; DCP3 and the uncensored checkpoint
    # (Marlin TP experts) are experimental.
    if moe_tp == '1' and (mode != 'mtp' or dcp not in ('1', '3')):
        raise ValueError('MOE_TP=1 is supported only with MODE=mtp and DCP=1 (DCP=3 experimental)')
    for retired in ('EPLB', 'KDA_NO_COPY'):
        if value(retired, '0').lower() not in ('0', 'false'):
            raise ValueError(f'{retired} was retired after slower repeated tests; remove it')
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
    # LMCACHE: off = GPU prefix cache only (vram); l1 = LMCache host-RAM tier
    # (LMCACHE_L1_GB, default 96; must fit host /dev/shm). L2 (disk) not tested at TP3.
    lmcache = value('LMCACHE', 'off').lower()
    if lmcache not in ('off', 'l1'):
        raise ValueError('LMCACHE must be off or l1 (the L2 disk tier is not tested at TP3)')
    cache_args = ['--cache-mode', 'vram']
    if lmcache == 'l1':
        l1 = value('LMCACHE_L1_GB', '96')
        if not re.fullmatch(r'[1-9][0-9]*', l1):
            raise ValueError('LMCACHE_L1_GB must be a positive integer')
        cache_args = ['--cache-mode', 'lmcache', '--cache-l1-gib', l1, '--no-cache-l2-enabled']
    args = ['--profile', 'glm53-flash', '--hardware', 'rtx-pro-6000-pcie',
            '--model', model, '--revision', revision, '--mode', mode,
            '--tensor-parallel-size', '3', '--decode-context-parallel-size', dcp,
            '--enable-expert-parallel', '--moe-backend', 'marlin' if checkpoint == 'uncensored' else 'auto',
            '--max-num-seqs', str(positive('MAX_NUM_SEQS', 8)),
            '--max-model-len', str(positive('MAX_MODEL_LEN', 1048576)),
            '--gpu-memory-utilization', value('GPU_MEMORY_UTILIZATION', '0.95'),
            '--max-num-batched-tokens', str(positive('MAX_NUM_BATCHED_TOKENS', 4096)),
            '--max-cudagraph-capture-size', str(maximum),
            *cache_args, '--enable-flashinfer-autotune',
            '--served-model-name', value('SERVED_MODEL_NAME', 'GLM-5.3-Flash-TP3'),
            '--default-chat-template-kwargs', json.dumps({'reasoning_effort': reasoning, 'clear_thinking': clear == 'true'}),
            '--override-generation-config', '{"temperature":1.0,"top_p":0.95}',
            '--mm-encoder-tp-mode', 'weights']
    if moe_tp == '1':
        args.remove('--enable-expert-parallel')
        # MOE_TP_BACKEND: TP-expert MoE kernels. Defaults: flashinfer_cutlass
        # (default checkpoint), marlin (uncensored). b12x is experimental.
        backend = value('MOE_TP_BACKEND', 'marlin' if checkpoint == 'uncensored' else 'flashinfer_cutlass')
        if backend not in ('flashinfer_cutlass', 'marlin', 'b12x'):
            raise ValueError('MOE_TP_BACKEND must be flashinfer_cutlass, marlin or b12x')
        args[args.index('--moe-backend') + 1] = backend
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
    # KV_RECLAIM: auto (default) enables the LM-head BF16 release, the shared
    # TP/EP PyNCCL communicator and the explicit KV budget only for the tested
    # combination; every other setting keeps vLLM's automatic KV sizing. 0 = off.
    reclaim = value('KV_RECLAIM', 'auto').lower()
    if reclaim not in ('auto', '0'):
        raise ValueError('KV_RECLAIM must be auto or 0')
    if checkpoint == 'uncensored':
        budgets = {} if lmcache == 'l1' else KV_RECLAIM_BYTES_UNCENSORED
    else:
        budgets = KV_RECLAIM_BYTES_L1 if lmcache == 'l1' else KV_RECLAIM_BYTES_BY_DCP
    tested = (mode == 'mtp' and dcp in budgets and moe_tp == '1'
              and positive('MAX_NUM_SEQS', 8) == 8 and positive('MAX_NUM_BATCHED_TOKENS', 4096) == 4096
              and maximum == 32 and not env.get('CUDAGRAPH_CAPTURE_SIZES')
              and positive('MAX_MODEL_LEN', 1048576) == 1048576
              and value('GPU_MEMORY_UTILIZATION', '0.95') == '0.95')
    kv_bytes = value('KV_CACHE_MEMORY_BYTES', '')
    if kv_bytes and not re.fullmatch(r'[1-9][0-9]*', kv_bytes):
        raise ValueError('KV_CACHE_MEMORY_BYTES must be a positive integer')
    if reclaim == 'auto' and tested:
        args += ['--kv-cache-memory-bytes', kv_bytes or str(budgets[dcp])]
    elif kv_bytes:
        raise ValueError('KV_CACHE_MEMORY_BYTES is only accepted with KV_RECLAIM=auto in the tested '
                         'default/MTP3/DCP1 or DCP3 with MOE_TP=1 configuration')
    env['_GLM53_KV_RECLAIM_ACTIVE'] = '1' if reclaim == 'auto' and tested else '0'
    return args


# lil's LMCache L1 arena (host /dev/shm via ipc: host) is not unlinked when the
# container stops, and lil refuses to start over an existing arena. Hold an
# exclusive lock beside it for the container's lifetime; whoever gets the lock
# knows no live server owns the arena and removes the stale one.
LMCACHE_ARENA = '/dev/shm/lmcache_l1_pool_lmcache-glm53-flash-8000-18000'


def _claim_lmcache_arena():
    import fcntl
    fd = os.open(LMCACHE_ARENA + '.lock', os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError(f'Another running server holds the LMCache arena {LMCACHE_ARENA}')
    os.set_inheritable(fd, True)  # kept open across exec for the container's lifetime
    if os.path.exists(LMCACHE_ARENA):
        os.unlink(LMCACHE_ARENA)
        print(f'GLM-5.3 LMCache: removed stale L1 arena {LMCACHE_ARENA}', flush=True)


def main():
    # Preserve historical command-list overrides used by the recorded benchmarks.
    extra = sys.argv[1:]
    if '--profile' in extra:
        args = extra
    else:
        if any(x != '--print-config' for x in extra):
            raise ValueError('Configure through environment variables; only --print-config is accepted')
        args = build_args(os.environ) + extra
    moe_tp = os.environ.get('MOE_TP', '0') or '0'
    if moe_tp not in ('0', '1'):
        raise ValueError('MOE_TP must be 0 or 1')
    if moe_tp == '1':
        if not os.path.isfile('/opt/lil/glm53-tp-experts.json'):
            raise ValueError('MOE_TP=1 requires glm53-kraken-tp3:20260925 or a compatible derived image')
        if '--enable-expert-parallel' in args or args[args.index('--moe-backend') + 1] not in ('flashinfer_cutlass', 'marlin', 'b12x'):
            raise ValueError('MOE_TP=1 requires expert parallelism off and flashinfer_cutlass, marlin or b12x')
    if '--enable-eplb' in args:
        raise ValueError('EPLB was retired after slower repeated tests')
    os.environ['VLLM_GLM53_TP3_MOE_TP'] = '2112' if moe_tp == '1' else '0'
    os.environ['VLLM_GLM53_KDA_NO_COPY'] = '0'
    active = os.environ.pop('_GLM53_KV_RECLAIM_ACTIVE', '0') == '1'
    os.environ['VLLM_GLM53_FP8_LM_HEAD_DROP_BF16'] = '1' if active else '0'
    os.environ['VLLM_SHARE_PYNCCL_COMMS'] = '1' if active else '0'
    print(f'GLM-5.3 KV reclaim {"active" if active else "inactive (automatic KV sizing)"}', flush=True)
    # These belong to this wrapper; native Kraken explicitly rejects MODE.
    for key in ("CHECKPOINT", "MODE", "MODEL", "MODEL_REVISION", "DFLASH_MODEL",
                "DFLASH_MODEL_REVISION", "DCP", "MAX_NUM_BATCHED_TOKENS",
                "MAX_CUDAGRAPH_CAPTURE_SIZE", "CUDAGRAPH_CAPTURE_SIZES",
                "MAX_NUM_SEQS", "MAX_MODEL_LEN", "GPU_MEMORY_UTILIZATION",
                "REASONING_EFFORT", "CLEAR_THINKING", "SERVED_MODEL_NAME",
                "MOE_TP", "EPLB", "EPLB_WINDOW_SIZE", "EPLB_STEP_INTERVAL", "KDA_NO_COPY",
                "KV_RECLAIM", "KV_CACHE_MEMORY_BYTES", "LMCACHE", "LMCACHE_L1_GB", "MOE_TP_BACKEND"):
        os.environ.pop(key, None)
    # Blank optional environment entries mean use the native defaults.
    for key in list(os.environ):
        if os.environ[key] == '':
            del os.environ[key]
    if '--cache-mode' in args and args[args.index('--cache-mode') + 1] == 'lmcache':
        _claim_lmcache_arena()
    os.execv('/usr/local/bin/lil-entrypoint', ['/usr/local/bin/lil-entrypoint', *args])


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        sys.exit(str(exc))
