# Frustum_loader v0.10: less GPU and Python overhead per token, research notes, roadmap

Base: `a3b2a56` (v0.9). Measured 2026-11-16 on the target machine (RTX 5060 Ti, 32 GB RAM, T7 at
10 Gbps). Every change produces bit-identical numbers; reply token ids are unchanged.

## Changes

- `fused_kernels.act_quant` + `Fused.activation`: `expert_correctness.activation` in one CUDA launch
  instead of about thirteen (0.274 -> 0.036 ms per call). Self-test on edge cases (all magnitudes,
  zero groups, FP8 ties and subnormals, float32 subnormals, clamping, NaN groups, both dtypes), then
  the first 64 real calls compared bit for bit; any difference switches only this kernel off.
  0 mismatches in 400 random stress tensors. `FRUSTUM_FUSED_ACT=0` turns it off.
- `streamed_moe`: w1 and w3 share one quantized input (`quantize`), and all CUDA call sites in
  `streamed_moe` and `layers0_3_block` use the fused quantizer.
- `expert_cache`: the CPU-side NaN-scale check (`inspect`) now also runs on the default blocking
  path, memoised per expert, so a newly uploaded expert needs no GPU synchronisation (there were
  761 forced synchronisations per decode token).
- `bench_e2e.py`: absolute output paths (the `--baseline-dir` mode wrote evidence relative to the
  other checkout and failed).

## Result

Same prompt, fresh processes, same session: a clean v0.9 checkout 4.1 s/token; v0.10 4.02 and
3.92 s/token (two runs). About 2-4% faster, token ids identical. Most expert work was already hidden
behind drive reads. Single-run numbers: the planned 3-run A/B stopped on the path bug above.

## Profiles (`v10_logs/`)

- One decode step: 9,629 copies (783 ms), 5,778 small divisions, 761 forced synchronisations.
  One expert call 1.25 ms, of which the fused matrix kernel is 0.11 ms and activation 0.27 ms x 3.
- Between layers, while the drive is idle (`between_layer_profile.txt`): attention 0.42 s/token,
  hyper-connection mixes 0.27 s (80 calls, 20-iteration Sinkhorn as ~85 tiny kernels), rope 0.10 s,
  re-decoding FP8 `wo_a` to FP32 every token 0.10 s.

## DSpark: analysed, paused (`v10_logs/expert_union.txt`)

Distinct experts per layer for 1/2/3/4/6 consecutive decode tokens: 6.0 / 9.9 / 12.9 / 15.6 / 20.3.
Verifying one token plus five drafts needs 3.4x the experts of one token, about 11-12 s per pass on
this drive, so about 3 accepted tokens per pass are needed to break even. Estimated 12-22% faster with
2-3 drafts and good acceptance, nothing with poor acceptance; needs draft layers, two heads and
attention-state rollback. If resumed: drafter in shadow mode first, continue only above ~60%
first-draft acceptance.

## Next (roadmap; whitepaper Section 20)

1. CUDA graphs for the hyper-connection mixes, direct FP8-to-BF16 `wo_a` decode, cached small
   weights (all exact).
2. Investigate the prefill/decode drift found in v0.9.
3. Model router in ActivatePrime: GLM or Phi-4 by default, DeepSeek on request.
4. Persist expert-use history and pre-warm the RAM tier at start-up.
5. DSpark drafter in shadow mode.
6. Opt-in fast mode (cache-aware routing, fewer experts per token), with quality measured.

ActivatePrime 0.1.4's worker enables unbuffered reads (with the larger RAM tier) and batched
follow-ups, which its own code path did not get before.
