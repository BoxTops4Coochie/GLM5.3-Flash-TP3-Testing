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
# 20261007 image (upstream 20261006 base): budgets re-checked on this base only.
# The 20260927 budgets (1779 blocks) run out of memory in startup autotuning here,
# so the original checkpoint uses 1700 blocks (3,222,912 tokens, launch + speed tested)
# and every other non-CSF combination uses automatic KV sizing.
KV_RECLAIM_BYTES_BY_DCP = {'1': 1700 * 13888512}
# With LMCACHE=l1 (DCP1 only): 1720 blocks, 3,180,865 tokens; LMCache's GPU-side
# transfer buffers need ~0.8 GB more headroom (1,779 blocks hangs in autotune).
KV_RECLAIM_BYTES_L1 = {}  # not re-measured on this base: automatic sizing
# Uncensored checkpoint, MOE_TP=1 (Marlin TP experts, experimental), no LMCache:
# DCP1 1721 blocks = 3,211,030 tokens (3x~895K resident, 86.8% KV, peak 1.85 GB free);
# DCP3 1770 blocks = 8,398,097 tokens, 8.01x 1M (8x895K stress, 5 resident, 0
# preemptions, peak 1.39 GB free; 1686 blocks also passed).
# CSF-QAD (W4A4 + resident expanded scales) DCP1: 2215 blocks = 4,199,951 tokens
# (4.01x 1M); 4 x ~891K prompts + 120K decode each, 0 preemptions, peak KV 89%,
# min free 1.80 GB/GPU (rebase-20261006/stress-2215res, 2026-10-06). 2170 blocks
# (3.92x) also passed with 2.42 GB free.
KV_RECLAIM_BYTES_CSF = {'1': 2215 * 13888512}
KV_RECLAIM_BYTES_UNCENSORED = {}  # not re-measured on this base: automatic sizing

CHECKPOINTS = {
    'default': ('local-inference-lab/GLM-5.3-Flash-NVFP4', '175ae8ce3b5af842b0d0140dbeb43e9cfc557c49'),
    'uncensored': ('orcarouter/GLM-5.3-Flash-Uncensored-NVFP4', 'ec0adf4f49c9570807cc11a5f650538c1893ae54'),
    # Same QAD routed experts as 'default', stored as FP4-CSF (compressed expert
    # scales, read compressed by B12X) with MXFP8 attention/shared experts and
    # NVFP4 MTP experts. Needs a Kraken image with the NVFP4-CSF reader and
    # MOE_TP=1 (the experts pad 2048 -> 2112 inside the CSF reader). Experimental.
    'csf-qad': ('local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD', 'fd660d51d1fc3caae26a4bf31b7451475bbb9bdc'),
    # Same QAD decoder (byte-identical) + quantized vision tower (MXFP8 attention,
    # NVFP4 MLP), standard ModelOpt layout with CSF-encoded expert scales. Vision
    # runs in data mode (replicated) since its quantized widths do not pad at TP3.
    'csf-qad-2': ('local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD', 'dec48abd33efa73c3bb7c95b74eee10cad34f9be'),
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
        raise ValueError('CHECKPOINT must be default, uncensored, csf-qad or csf-qad-2')
    csf = checkpoint.startswith('csf-qad')
    model, revision = CHECKPOINTS[checkpoint]
    for key, expected in [('MODEL', model), ('MODEL_REVISION', revision)]:
        if env.get(key) and env[key] != expected:
            raise ValueError(f'{key} disagrees with CHECKPOINT={checkpoint}: expected {expected}')
    mode = value('MODE', 'dflash2')
    mode = {'dflash': 'dflash2', 'mtp0': 'off'}.get(mode, mode)
    if mode not in ('mtp', 'dflash2', 'off'):
        raise ValueError('MODE must be mtp, dflash2 (or dflash), or mtp0 (or off)')
    # DRAFT_TOKENS: speculative tokens per step. Empty = the mode default (MTP 3,
    # DFlash2 7). 1-7: MTP reuses its single draft layer; DFlash2's draft block
    # is 8 positions. Only MTP 3 / DFlash2 7 are qualified, and only MTP 3 keeps
    # the measured graph ladder and the KV reclaim budget (others: automatic KV).
    draft_default = {'mtp': 3, 'dflash2': 7}.get(mode)
    draft_tokens = draft_default
    if value('DRAFT_TOKENS', ''):
        if mode == 'off':
            raise ValueError('DRAFT_TOKENS needs MODE=mtp or dflash2')
        draft_tokens = positive('DRAFT_TOKENS', draft_default)
        if not 1 <= draft_tokens <= 7:
            raise ValueError('DRAFT_TOKENS must be 1-7')
    dcp = value('DCP', '1')
    if dcp not in ('1', '3'):
        raise ValueError('DCP must be 1 or 3')
    moe_tp = value('MOE_TP', '0')
    if moe_tp not in ('0', '1'):
        raise ValueError('MOE_TP must be 0 or 1')
    # MOE_TP=1 is qualified at DCP1; DCP3 is an experimental combination.
    # MOE_TP=1 is qualified for default/MTP/DCP1; DCP3 and the uncensored checkpoint
    # (Marlin TP experts) are experimental.
    # csf-qad also allows DFlash2 (experimental): its experts need MOE_TP=1.
    moe_tp_modes = ('mtp', 'dflash2') if csf else ('mtp',)
    if moe_tp == '1' and (mode not in moe_tp_modes or dcp not in ('1', '3')):
        raise ValueError('MOE_TP=1 is supported only with MODE=mtp (csf-qad: mtp or dflash2) and DCP=1 or 3')
    if csf and (mode not in ('mtp', 'dflash2') or moe_tp != '1'):
        raise ValueError('CHECKPOINT=csf-qad needs MODE=mtp or dflash2 and MOE_TP=1 (FP4-CSF experts run on B12X without EP)')
    if csf:
        # CSF_ACTIVATIONS: a4 (default; W4A4 like the original checkpoint on
        # FlashInfer, ~10% faster decode) or a16 (upstream's W4A16, more exact).
        # CSF_RESIDENT_SCALES: 1 (default) expands expert scales once per layer
        # (+3.7% C1, ~2.4 GiB/GPU); 0 keeps them compressed (per-call decode).
        acts = value('CSF_ACTIVATIONS', 'a4').lower()
        resident = value('CSF_RESIDENT_SCALES', '1')
        if acts not in ('a4', 'a16') or resident not in ('0', '1'):
            raise ValueError('CSF_ACTIVATIONS must be a4 or a16; CSF_RESIDENT_SCALES 0 or 1')
        env['VLLM_B12X_MOE_FP4_FORCE_A16'] = '1' if acts == 'a16' else '0'
        env['VLLM_GLM53_CSF_RESIDENT_SCALES'] = resident
    for retired in ('EPLB', 'KDA_NO_COPY'):
        if value(retired, '0').lower() not in ('0', 'false'):
            raise ValueError(f'{retired} was retired after slower repeated tests; remove it')
    reasoning = value('REASONING_EFFORT', 'max')
    clear = value('CLEAR_THINKING', 'true').lower()
    if reasoning not in ('low', 'high', 'max') or clear not in ('true', 'false'):
        raise ValueError('REASONING_EFFORT must be low/high/max; CLEAR_THINKING must be true/false')
    # Default graph maximum covers 8 slots x (draft tokens + 1) verify positions.
    default_maximum = 64 if mode == 'dflash2' else 32
    if mode == 'mtp' and draft_tokens > 3:
        default_maximum = positive('MAX_NUM_SEQS', 8) * (draft_tokens + 1)
    maximum = positive('MAX_CUDAGRAPH_CAPTURE_SIZE', default_maximum)
    sizes = value('CUDAGRAPH_CAPTURE_SIZES', '')
    if not sizes:
        sizes = [n for n in (1, 2, 4, 8, 16, 32, 40, 48, 64, 96, 128, 192, 256) if n < maximum]
        # Exact intermediate batches: measured at default/DCP1/graph32/8 slots.
        # Other checkpoints, DCP3, custom maxima and explicit ladders retain their policy.
        # csf-qad*: the same measured ladder as the default checkpoint (C1/C8 sizes unchanged).
        if (checkpoint == 'default' or csf) and dcp == '1' and maximum == 32 and positive('MAX_NUM_SEQS', 8) == 8:
            if mode == 'off':
                sizes = [1, 2, 3, 4, 5, 6, 7, 8, 16]
            elif mode == 'mtp' and draft_tokens == 3:
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
            '--mm-encoder-tp-mode', 'data' if checkpoint == 'csf-qad-2' else 'weights']
    if moe_tp == '1':
        args.remove('--enable-expert-parallel')
        # MOE_TP_BACKEND: TP-expert MoE kernels. Defaults: flashinfer_cutlass
        # (default checkpoint), marlin (uncensored). b12x is experimental.
        backend = value('MOE_TP_BACKEND', 'marlin' if checkpoint == 'uncensored' else 'b12x' if csf else 'flashinfer_cutlass')
        if backend not in ('flashinfer_cutlass', 'marlin', 'b12x'):
            raise ValueError('MOE_TP_BACKEND must be flashinfer_cutlass, marlin or b12x')
        if csf and backend != 'b12x':
            raise ValueError('CHECKPOINT=csf-qad decodes its compressed experts on b12x only')
        args[args.index('--moe-backend') + 1] = backend
    args += ['--cudagraph-capture-sizes', *map(str, sizes)]
    if checkpoint == 'uncensored':
        args += ['--quantization', 'compressed-tensors', '--load-format', 'auto']
    if mode == 'mtp':
        spec = dict(method='mtp', num_speculative_tokens=draft_tokens, revision=revision,
                    draft_sample_method='probabilistic', rejection_sample_method='standard',
                    moe_backend=value('MTP_MOE_BACKEND', 'triton' if checkpoint == 'uncensored' else 'marlin'),
                    attention_backend='B12X')
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
        args += ['--draft-tokens', str(draft_tokens), '--draft-model', draft, '--draft-revision', draft_revision]
    # KV_RECLAIM: auto (default) enables the LM-head BF16 release, the shared
    # TP/EP PyNCCL communicator and the explicit KV budget only for the tested
    # combination; every other setting keeps vLLM's automatic KV sizing. 0 = off.
    reclaim = value('KV_RECLAIM', 'auto').lower()
    if reclaim not in ('auto', '0'):
        raise ValueError('KV_RECLAIM must be auto or 0')
    if checkpoint == 'csf-qad':
        # The CSF budget assumes W4A4 + resident scales; W4A16 or compressed
        # scales leave more room but fall back to the same (safe) budget.
        budgets = {} if lmcache == 'l1' else KV_RECLAIM_BYTES_CSF
    elif checkpoint == 'csf-qad-2':
        # Same model memory as csf-qad (60.93 GiB/GPU with resident scales).
        budgets = {} if lmcache == 'l1' else KV_RECLAIM_BYTES_CSF
    elif checkpoint == 'uncensored':
        budgets = {} if lmcache == 'l1' else KV_RECLAIM_BYTES_UNCENSORED
    else:
        budgets = KV_RECLAIM_BYTES_L1 if lmcache == 'l1' else KV_RECLAIM_BYTES_BY_DCP
    tested = (mode == 'mtp' and draft_tokens == 3 and dcp in budgets and moe_tp == '1'
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
                "MAX_CUDAGRAPH_CAPTURE_SIZE", "CUDAGRAPH_CAPTURE_SIZES", "DRAFT_TOKENS",
                "MAX_NUM_SEQS", "MAX_MODEL_LEN", "GPU_MEMORY_UTILIZATION",
                "REASONING_EFFORT", "CLEAR_THINKING", "SERVED_MODEL_NAME",
                "MOE_TP", "EPLB", "EPLB_WINDOW_SIZE", "EPLB_STEP_INTERVAL", "KDA_NO_COPY",
                "KV_RECLAIM", "KV_CACHE_MEMORY_BYTES", "LMCACHE", "LMCACHE_L1_GB", "MOE_TP_BACKEND",
                "CSF_ACTIVATIONS", "CSF_RESIDENT_SCALES", "MTP_MOE_BACKEND"):
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
