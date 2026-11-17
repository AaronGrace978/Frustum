"""Token-level generation control flow, independent of any model or tensor library.

Kept separate so the prefix-reuse logic can be tested without a GPU.
"""


def chat_prompt(history, prompt, system=None):
    """DeepSeek-V4.1 prompt for non-thinking chat.

    Checked against the checkpoint's own encoding/encoding.py
    (encode_messages(..., thinking_mode='chat')): identical strings for single
    turns, multi-turn history and a leading system message. selftest.py repeats
    that comparison whenever the official encoder is present.
    history: list of (user, assistant) pairs.
    """
    text = '<｜begin▁of▁sentence｜>'
    if system:
        text += f'<｜System｜>{system}'
    for user, reply in history:
        text += f'<｜User｜>{user}<｜Assistant｜></think>{reply}<｜end▁of▁sentence｜>'
    return text + f'<｜User｜>{prompt}<｜Assistant｜></think>'


def reusable_prefix(fed, ids):
    """Length of the already-processed token prefix that the new prompt extends, else 0."""
    if fed and len(fed) < len(ids) and ids[:len(fed)] == fed:
        return len(fed)
    return 0


def drive(fed, ids, max_new_tokens, eos, pick, run, reuse=True, on_token=None, append_chunk=1):
    """Produce up to max_new_tokens after prompt `ids`.

    fed: list of token ids the engine state currently covers. Mutated to stay
         in step with the engine; cleared if anything fails part-way.
    run(tokens, phase, want_logits) -> (logits or None, timing dict). The engine
         must start from scratch when phase == 'prefill' and append otherwise.
    pick(logits) -> token id.
    append_chunk: new prompt tokens fed per engine step when extending existing state (v0.9).
         1 is the old one-token-per-step behaviour; larger values cut the number of passes
         over the layers, and so the number of per-layer expert-read stalls.
    Returns (output_ids, timings, reused_prompt_tokens, stopped_at_eos).
    """
    if append_chunk < 1:
        raise ValueError('append_chunk must be at least 1')
    reused = reusable_prefix(fed, ids) if reuse else 0
    if not reused:
        fed.clear()
    output, timings = [], []

    def step(tokens, phase, want_logits):
        logits, timing = run(tokens, phase, want_logits)
        fed.extend(tokens)
        timings.append(timing)
        return logits

    try:
        pending = ids[reused:]
        if reused:
            # With state present the engine advances append_chunk positions per step.
            for begin in range(0, len(pending), append_chunk):
                piece = pending[begin:begin + append_chunk]
                logits = step(piece, 'reuse_feed', begin + len(piece) == len(pending))
        else:
            logits = step(pending, 'prefill', True)
        stopped = False
        while True:
            token = pick(logits)
            timings[-1]['token_id'] = token
            if token == eos:
                stopped = True
                break
            output.append(token)
            if on_token is not None:
                on_token(output, timings[-1])
            if len(output) >= max_new_tokens:
                break
            logits = step([token], 'decode', True)
    except BaseException:
        fed.clear()  # per-layer state may be half-advanced; rebuild next time
        raise
    return output, timings, reused, stopped
