"""Token-level generation control flow, independent of any model or tensor library.

Kept separate so the prefix-reuse logic can be tested without a GPU.
"""


def reusable_prefix(fed, ids):
    """Length of the already-processed token prefix that the new prompt extends, else 0."""
    if fed and len(fed) < len(ids) and ids[:len(fed)] == fed:
        return len(fed)
    return 0


def drive(fed, ids, max_new_tokens, eos, pick, run, reuse=True, on_token=None):
    """Produce up to max_new_tokens after prompt `ids`.

    fed: list of token ids the engine state currently covers. Mutated to stay
         in step with the engine; cleared if anything fails part-way.
    run(tokens, phase, want_logits) -> (logits or None, timing dict). The engine
         must start from scratch when phase == 'prefill' and append otherwise.
    pick(logits) -> token id.
    Returns (output_ids, timings, reused_prompt_tokens, stopped_at_eos).
    """
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
            # With state present the engine advances one position per step.
            for index, token in enumerate(pending):
                logits = step([token], 'reuse_feed', index == len(pending) - 1)
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
