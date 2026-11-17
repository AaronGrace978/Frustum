"""Check qwen_next_chat.QwenNext against the transformers reference on a tiny random Qwen3-Next.

Builds a small Qwen3NextForCausalLM with random weights (every parameter perturbed, so the
(1 + weight) norms and gated norms are exercised), saves it as a two-shard safetensors
checkpoint with an index, loads it through Frustum's Catalog and tiered expert cache (with a GPU
tier too small for all experts plus a RAM tier, so eviction and host hits happen), and compares
logits position by position:
  prefill of 5 tokens, then 3 single-token decode steps, then one 4-token append,
against one full-sequence forward of the reference.

Needs transformers >= 4.57 (Qwen3-Next). Point QWEN_PARITY_TRANSFORMERS at a folder that holds it
(e.g. a pip --target install) if it is not installed in this environment:

    python qwen_next_parity.py
    python qwen_next_parity.py --dtype bf16
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

extra = os.environ.get('QWEN_PARITY_TRANSFORMERS')
if extra:
    sys.path.insert(0, extra)

import torch  # noqa: E402

from qwen_next_chat import QwenNext, prompt_suffix  # noqa: E402


def tiny_config(Config):
    return Config(vocab_size=160, hidden_size=64, intermediate_size=96, num_hidden_layers=8,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=32, partial_rotary_factor=0.25,
                  rope_theta=10000.0, linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
                  linear_num_key_heads=2, linear_num_value_heads=4, num_experts=8, num_experts_per_tok=3,
                  moe_intermediate_size=32, shared_expert_intermediate_size=32, norm_topk_prob=True,
                  full_attention_interval=4, rms_norm_eps=1e-6, tie_word_embeddings=False,
                  max_position_embeddings=256)


def save_checkpoint(model, config, folder, dtype):
    from safetensors.torch import save_file
    state = {k: v.detach().to(dtype).contiguous() for k, v in model.state_dict().items()}
    names = sorted(state)
    shards = {'model-00001-of-00002.safetensors': names[::2], 'model-00002-of-00002.safetensors': names[1::2]}
    weight_map = {}
    for filename, keys in shards.items():
        save_file({k: state[k] for k in keys}, str(folder / filename), metadata={'format': 'pt'})
        weight_map.update({k: filename for k in keys})
    total = sum(v.numel() * v.element_size() for v in state.values())
    (folder / 'model.safetensors.index.json').write_text(json.dumps(dict(metadata=dict(total_size=total),
                                                                         weight_map=weight_map)))
    cfg = config.to_dict()
    cfg['model_type'] = 'qwen3_next'
    (folder / 'config.json').write_text(json.dumps(cfg))
    return state


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dtype', choices=('f32', 'bf16'), default='f32')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
    dtype = dict(f32=torch.float32, bf16=torch.bfloat16)[args.dtype]
    torch.manual_seed(args.seed)
    config = tiny_config(Qwen3NextConfig)
    config._attn_implementation = 'eager'
    ref = Qwen3NextForCausalLM(config).eval()
    for name, param in ref.named_parameters():
        if name.endswith('A_log') or name.endswith('dt_bias'):
            param.add_(torch.randn_like(param) * 0.1)
        elif name.endswith('linear_attn.norm.weight'):
            param.copy_(1 + torch.randn_like(param) * 0.1)
        elif name.endswith('norm.weight'):
            param.copy_(torch.randn_like(param) * 0.1)
        else:
            param.copy_(torch.randn_like(param) * 0.15)
    ref = ref.to(dtype)
    ids = torch.randint(0, config.vocab_size, (12,)).tolist()
    expected = ref(torch.tensor([ids])).logits[0].float()
    # The same (rounded) weights evaluated in float32: what both bf16 engines are approximating.
    exact = ref.float()(torch.tensor([ids])).logits[0] if dtype != torch.float32 else expected
    ref = ref.to(dtype)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        state = save_checkpoint(ref, config, folder, dtype)
        dense = sum(v.numel() * v.element_size() for k, v in state.items()
                    if '.mlp.experts.' not in k and k != 'model.embed_tokens.weight')
        one = 3 * config.hidden_size * config.moe_intermediate_size * torch.finfo(dtype).bits // 8
        # 7 experts fit on the GPU (of 64), 40 in RAM: evictions, host hits and drive reads all happen.
        engine = QwenNext(folder, dense + 7 * one, 40 * one, device=device)
        try:
            got = [engine.forward(ids[:5], all_positions=True)]
            for i in range(5, 8):
                got.append(engine.forward([ids[i]], all_positions=True))
            suffix, reused = prompt_suffix(engine, ids)
            assert reused == 8 and suffix == ids[8:], (reused, suffix)
            got.append(engine.forward(suffix, all_positions=True))
            got = torch.cat(got).cpu()
            stats = engine.metrics()
        finally:
            engine.close()
        # A different conversation must start over.
        assert prompt_suffix(engine, [ids[0] + 1] + ids[1:])[1] == 0 and engine.tokens == []

    scale = exact.abs().max().item()
    worst = (got - expected).abs().max().item() / scale
    agree = (got.argmax(-1) == expected.argmax(-1)).float().mean().item()
    report = dict(dtype=args.dtype, positions=len(ids), max_rel_diff_vs_reference=worst, argmax_agreement=agree,
                  gpu_hits=stats['gpu_hits'], host_hits=stats['host_hits'], disk_reads=stats['disk_reads'],
                  evictions=stats['evictions'])
    if dtype == torch.float32:
        ok = worst <= 2e-4 and agree == 1
    else:
        # bf16 rounding differs between the two engines; require this engine to be no further from
        # the float32 evaluation of the same weights than the reference's own bf16 path is (+20%).
        mine = (got - exact).abs().max().item() / scale
        theirs = (expected - exact).abs().max().item() / scale
        report.update(engine_err_vs_f32=mine, reference_err_vs_f32=theirs)
        ok = mine <= 1.2 * theirs + 1e-3
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit('PARITY FAILED')
    print('PARITY OK')


if __name__ == '__main__':
    main()
