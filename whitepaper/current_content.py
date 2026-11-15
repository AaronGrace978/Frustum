"""Condensed whitepaper content, backed by the saved run records."""
import json
from statistics import mean


def current_pages(root):
    def load(name):
        return json.loads((root.parent / name).read_text(encoding='utf-8'))

    deep = load('chat_smoke_results.json')
    phi = load('phi4_chat_results.json')
    glm = load('glm_chat_results.json')
    block = load('layers0_3_results.json')
    experiment = load('glm_placement_results.json')
    repeat = load('glm_placement_reverse_results.json')
    split_runs = [next(run for run in record['runs'] if run['mode'] == 'split')
                  for record in (experiment, repeat)]
    first_runs = {run['mode']: run for run in experiment['runs']}
    repeated_runs = {run['mode']: run for run in repeat['runs']}

    def first(run, phase):
        return run[phase]['time_to_first_token_seconds']

    def span(phase):
        values = sorted(first(run, phase) for run in split_runs)
        return f'{values[0]:.2f} to {values[1]:.2f} s'

    cached = split_runs[0]['followup_reuse']['reused_prompt_tokens']
    reply = split_runs[0]['followup_reuse']['reply']
    mib = experiment['gpu_cache_budget_bytes'] / 2**30
    rows = [['Policy', 'Warm first token, first / reverse run']]
    for name, label in [('baseline', 'Unified GPU cache'), ('split', 'Split GPU cache'), ('tiered', 'Split GPU + host RAM')]:
        second = repeated_runs.get(name)
        second_time = f'{first(second, "warm"):.2f} s' if second else 'not repeated'
        rows.append([label, f'{first(first_runs[name], "warm"):.2f} s / {second_time}'])
    tier_reads = first_runs['tiered']['warm']['file_bytes'] / 1e9
    split_reads = first_runs['split']['warm']['file_bytes'] / 1e9
    return [
        {
            'title': 'Frustum_loader',
            'subtitle': 'Local model inference on a 16 GB GPU',
            'body': [
                ('p', 'Consolidated engineering whitepaper | Version 0.4.0 | November 14, 2026'),
                ('p', 'Project concept and direction: Aaron Grace. Implementation and analysis assisted by ChatGPT. Experimental engineering report.'),
                ('contributor', 'Contributor: Dino Buddy. The project also received assistance from DinoClaw, Aaron Grace’s original agent runtime.'),
                ('h', 'What the prototype demonstrates'),
                ('p', f'Frustum_loader ran three distinct checkpoints on one RTX 5060 Ti. In GLM-4.7-Flash, reusing {cached} cached prompt tokens brought a warm follow-up to its first token in {span("followup_reuse")}. Rebuilding the same conversation took {span("followup_rebuild")}. Both paths produced the same follow-up token IDs.'),
                ('table', [
                    ['Checkpoint', 'Recorded result on the same GPU'],
                    ['DeepSeek V4.1 Flash', f'All 40 layers ran from a roughly 510 GB checkpoint. First smoke test generated {len(deep["token_ids"])} tokens in {deep["seconds"]:.0f} s.'],
                    ['Phi-4 14B Q4_K_S', f'Packed weights resident. Load {phi["load_seconds"]:.2f} s, first token after load {phi["first_token_seconds"]:.2f} s, then {mean(phi["decode_seconds"]):.2f} s per step.'],
                    ['GLM-4.7-Flash Q4_K_M', f'On-demand routed experts. Original first token {glm["first_token_seconds"]:.2f} s, complete reply {glm["first_token_seconds"] + sum(glm["decode_seconds"]):.2f} s, including the end step.'],
                ]),
                ('h', 'Original GLM prompt and reply'),
                ('transcript', glm['prompt']),
                ('transcript', glm['reply']),
                ('p', 'These are separate smoke tests with different model settings and prompt lengths. The within-model GLM comparison appears on the next page.'),
            ],
        },
        {
            'title': 'GLM placement and warm chat',
            'body': [
                ('h', 'How placement works'),
                ('p', 'The reader serves bounded weight ranges from the local checkpoint. GLM loads only selected routed expert slices. The split policy sets aside GPU space for attention, routers, shared experts, and other always-used weights. A separate cache holds routed slices. The optional host tier retains evicted candidates in system RAM.'),
                ('p', 'ActivatePrime keeps one model worker warm. GLM reuses its attention state only when the formatted conversation starts with the exact cached token sequence. A prefix mismatch triggers a rebuild. The current app defaults to split placement.'),
                ('h', 'Measured comparison'),
                ('p', f'The policies shared a {mib:.0f} GiB GPU weight budget. Each run started with fresh application caches. The Windows file cache was uncontrolled. The warm prompt repeats the initial Dino Buddies prompt with weights already cached.'),
                ('table', rows),
                ('table', [
                    ['Split GPU follow-up', 'First / reverse run'],
                    ['Full prompt rebuild', ' / '.join(f'{first(r, "followup_rebuild"):.2f} s' for r in split_runs)],
                    [f'Reuse {cached} prompt tokens', ' / '.join(f'{first(r, "followup_reuse"):.2f} s' for r in split_runs)],
                ]),
                ('transcript', f'Follow-up prompt: {experiment["followup"]}\nModel reply: {reply}'),
                ('p', f'The RAM tier cut warm logical file reads from {split_reads:.2f} GB to {tier_reads:.2f} GB in the first comparison, while increasing first-token latency. Split placement used about {max(r["peak_reserved_bytes"] for r in split_runs)/2**30:.2f} GiB peak PyTorch-reserved GPU memory. Logical reads can be served by the OS cache.'),
            ],
        },
        {
            'title': 'Validation and evidence',
            'body': [
                ('h', 'Checks and scope'),
                ('p', f'The DeepSeek Layer 0 repeatability audit passed. Layers 0-3 passed the tested checks with {block["full_prefill_vs_prefix_decode_relative_L2"]:.2e} relative L2 difference between prefill and cached decode. Phi-4 passed sampled dequantization and finite-logit checks. All tested GLM placement modes reproduced the saved Dino reply token for token.'),
                ('p', 'GLM follow-up parity was checked against a full rebuild on the tested conversation. Cold first-token timings varied widely with run order and cache state. Full-model numerical parity, other GPUs, and broader multi-turn workloads remain unverified.'),
                ('h', 'Reproduce the GLM comparison'),
                ('code', '.\\run_glm_placement_test.ps1 --evidence placement-a.json\n.\\run_glm_placement_test.ps1 --order "split,baseline" --evidence placement-b.json'),
                ('p', 'The launcher accepts FRUSTUM_PYTHON to select an existing CUDA Python environment. Use the same GLM GGUF checkpoint and tokenizer for both orders.'),
                ('h', 'Embedded evidence'),
                ('table', [
                    ['Records', 'What they contain'],
                    ['GLM placement and reverse-order JSONs', 'Per-turn timings, cache counters, reply token IDs, and GPU memory.'],
                    ['DeepSeek, Phi-4, and GLM run JSONs', 'Original prompts, generated replies, token IDs, and measurements.'],
                    ['Layer and attention checks, logs, cache profiles', 'Earlier checkpoint validation and performance experiments.'],
                ]),
                ('p', 'Open the PDF Attachments pane to retrieve the original files. This edition is a local evidence copy. Some older terminal logs contain machine-specific paths.'),
            ],
        },
    ]
