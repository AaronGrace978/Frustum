# Frusium_loader — current prototype

GPU: NVIDIA RTX 5060 Ti 16 GB. Validated with Python 3.13 and torch 2.11.0+cu128.
Working folder: G:\Models\DeepSeek-V4.1-Flash\Frusium_loader

## Run

From this folder in PowerShell:

    .\run_moe_test.ps1
    .\run_sparse_rows_test.ps1
    .\run_block_primitives_test.ps1

The MoE test defaults to layers 0,1,2,3, with two independently generated BF16 inputs per layer. A smaller run:

    .\run_moe_test.ps1 --layers 0 --tokens 1

Existing tests:

    .\run_expert_test.ps1
    .\run_benchmark.ps1 --cache-mib 256

The launchers select the isolated CUDA interpreter installed under the Codex workspace on C:. They are machine-specific; if moved to another PC, install a CUDA-enabled PyTorch build and NumPy in a virtual environment and invoke the Python scripts with that interpreter. CUDA environment and model weights are not bundled.

## What is implemented

- Header-only safetensors catalog and bounded mmap range reader (frusium_loader.py).
- Byte-budgeted GPU LRU and two pinned 8 MiB host staging buffers, copy stream, compute stream, and CUDA event lifetime guards (benchmark_layers.py).
- Real MXFP4 expert: three projections, FP8 activation quantization, BF16 rounding, and model-specific SwiGLU clamps (expert_correctness.py).
- Text router using actual BF16 router weights, sqrt(softplus) scores, correction bias for selection, six experts, normalized/scaled unbiased mixture weights (streamed_moe.py).
- Mixture weights applied inside experts before final projection, as in the supplied model.
- FP8 shared expert with per-32x32 block scales; combined routed + shared output.
- Exact next-selected-expert prefetch after routing. This does not predict experts of a future transformer layer.
- Bounded sparse row lookup for token embeddings and Engram FP8/E8M0 tables; repeated rows served from GPU LRU (sparse_rows.py).
- RMSNorm, mHC coefficient generation/Sinkhorn normalization, residual collapse/expansion (block_primitives.py).

## Validation results

- All four MoE sublayers passed independent NumPy FP64 router/linear references, with independent arithmetic FP4 and FP8 decoders. Two inputs selected 12 distinct experts per layer, plus the shared expert. Final BF16 outputs matched exactly on these test inputs.
- Four-layer MoE run: 1,059.861504 MB logical payload reads, 39 evictions, peak allocated 0.4143 GiB, peak reserved 0.5254 GiB. CPU-validation-inclusive runtime was 91.5 seconds. It is NOT a token-generation speed measurement.
- Sparse rows: first/interior/last rows and duplicates tested in the 384,006,168-row Engram table and token embedding table. Lookup payload: 31,512 bytes; separate reference reads: 42,016 bytes. Warm lookups read zero additional bytes.
- Residual primitives passed using real layer-0 parameters versus a NumPy FP64 calculation. The residual matrix uses [source,destination] orientation, matching the supplied source.
- Earlier single-expert and six loader tests passed.

Results are in moe_correctness_results.json, sparse_rows_results.json, block_primitives_results.json, and expert_correctness_results.json.

## Accuracy and memory limits

This is a dequantize-and-compute correctness baseline, not fused FP4 Tensor Core inference. FP8 activation conversion and SwiGLU use PyTorch on both reference and GPU paths; the CPU reference is therefore independent for weight decode, routing arithmetic and matrix products, not every numerical operation. Original TileLang kernel parity has not been established. Test agreement is not a proof over arbitrary inputs.

The 12 GiB setting caps the PyTorch allocator, not all Windows processes or all driver allocations. Free-memory checks retain 2 GiB headroom for weight-cache allocations. These controls cannot guarantee Windows never uses shared GPU memory. Small row/primitive tests also guard available memory.

Cache hit rate includes prefetched entries. Logical file-read counts are not physical SSD reads because OS caching may serve requests. CPU reference validation keeps one expert matrix in CPU RAM at a time; the full model is never loaded.

The checksum benchmark still exists for transfer experiments and is explicitly labeled synthetic. The MoE runner performs real expert computation. Its four tests are INDEPENDENT sublayers, not four complete transformer layers chained together.

## Remaining full-model work (updated after layer-0 milestone)

1. Integrate RoPE, sliding-window attention, compressed KV construction, sparse/hierarchical index selection, source-layer sharing and causal encoder-decoder scheduling.
2. Wire token-derived n-gram hashing and Engram gates/projections to the tested row reader.
3. Layer-0 SWA block assembly and prefill/decode schedule consistency now pass; establish original-kernel parity and extend integration to subsequent blocks.
4. Stream the output head, integrate tokenizer/prompt encoding, and validate a 40-layer text-generation loop.
5. Profile actual routing traces and optimize kernels/prefetch; vision and DSpark come after correct text decoding.

The first bounded text-generation smoke test is recorded below. Full-model numerical parity remains open.

Format reference: https://docs.nvidia.com/nemo/automodel/nightly/nemo-automodel/nemo_automodel/components/quantization/mxfp4
Model behavior source: local inference/model.py and inference/kernel.py, read for the implementation; the full downloaded model module is not executed.

## Complete layer-0 milestone

    .\run_layer0_test.ps1

layer0_block.py integrates actual token embeddings, RMSNorm, mHC, low-rank FP8 Q/KV projections, RoPE, sliding-window attention with learned sink, persistent KV ring, BF16 grouped output projection (dequantized from checkpoint FP8 as convert.py prescribes), routed/shared MoE, and residual updates. Layer 0 requires no compressed attention or Engram insertion.

PASS: 3-token prefill versus 2-token prefill plus one cached decode, with zero observed output difference for token embedding IDs 100,200,300. The MoE portion also passed its CPU reference on these attention-derived inputs. Peak allocated: 0.91855 GiB; peak reserved: 1.24805 GiB. Logical reads: 508.495832 MB. Runtime: 32.88 s including CPU validation, not token speed.

Attention primitives passed a small NumPy oracle, causality, ring wraparound/position validation, and RoPE reversal checks. The full block compares schedules of the same implementation: it is not an independent full-block oracle or original TileLang parity. Full 128-position-window wrap was tested at the ring primitive level, not by running 129 real block tokens. Attention reduces the selected window directly, so its reduction order can differ from TileLang online blocks. Later compressed-KV attention, full Engram integration and 40-layer generation remain unfinished.

The seven-page whitepaper in whitepaper/ is frozen at version 0.1 before this milestone; layer0_block_results.json records the subsequent result.

## Sequential validation and local chat

### GLM placement and warm-chat benchmark

`benchmark_glm_placement.py --evidence glm_placement_repeat.json` compares a unified GPU LRU, separate GPU budgets for always-used and routed expert weights, and the split GPU cache with a bounded host RAM tier. Run it with the same CUDA Python environment as the chat tests. The total GPU weight budget stays fixed across policies. The benchmark checks the saved Dino Buddies answer token for token and compares a cached follow-up with a full re-prefill of the same conversation.

An empty application cache does not imply an empty Windows file cache. File bytes and read time here come from Frusium's bounded reader. Upload timing is host wall time for blocking copies and can include CUDA synchronization. The RAM budget counts retained payload bytes, not overall host memory usage.

The first comparison is saved in `glm_placement_results.json`. All three modes reproduced the saved Dino reply. In the split mode, a warm first token took 9.80 s and a follow-up with 34 cached prompt tokens took 5.33 s to its first token, versus 11.72 s with a full rebuild. The RAM tier reduced logical file reads, but its warm latency was higher in this run. See the per-turn counters for the other policies.

A reverse-order repeat in `glm_placement_reverse_results.json` measured 10.27 s warm for split and 11.20 s for baseline. The split follow-up took 5.28 s with prefix reuse and 12.42 s with a rebuild. First-run timings varied widely between orders. ActivatePrime's source defaults to split placement and keeps the host tier optional.

### Phi-4 smoke test

Run `.\run_phi4_test.ps1 --max-new-tokens 24` with Microsoft's `phi-4-Q4_K_S.gguf` and Phi-4 `tokenizer.json` in `G:\Models\Phi-4`. This backend uses Frusium_loader's bounded reader, keeps the packed weights on the GPU, and runs the Phi-4 forward pass and KV cache in PyTorch. It uses the `gguf` package for metadata and reference dequantization. The saved `phi4_chat_results.json` records the first 40-layer smoke test. Sampled quantization checks and finite logits passed. Full-model numerical parity remains open.

### Profile cached decode

Run `.\run_chat.ps1 --prompt "Say hello." --max-new-tokens 3 --output-head resident --profile-decode --evidence decode_profile_repeat.json` to capture two cached decode steps across all 40 layers. The profile records synchronized per-layer wall time, logical bytes by reader, and routed expert IDs. Synchronizing each layer can perturb timing. Logical bytes include reads served by the OS cache.

The saved `decode_profile_results.json` matched the first three token IDs of the original run. Prefill took 721.3 s. Cached decode took 279.9 s and 68.9 s. Each decode requested about 5.93 GB of expert weights, 5.62 GB of dense weights, and 0.16 GB of router weights. The resident head requested no additional bytes during decode and took roughly 33 ms per step.

Run `python decode_cache_replay.py decode_profile_results.json` to estimate demand-order LRU reuse from the recorded routes and checkpoint metadata. On the second recorded decode step, a 6 GiB expert cache retains an estimated 159 of 280 expert requests and cuts expert reads from 5.93 GB to 2.27 GB. This is a replay estimate. Measure actual latency and memory in a live run with `--expert-cache-mib 6144` before adopting it.

The 40-layer smoke test returned “Hello! 👋 How can” for “Say hello.” at the six-token limit. It took 1,117.56 seconds and peaked at 3.057 GiB of PyTorch-reserved memory. See chat_smoke_results.json for token IDs and memory measurements.

Run `.\run_layers0_3_test.ps1 --output layers0_3_repeat.json` to repeat the four-block check without overwriting the saved report. The recorded run matched the saved Layer 0 output, passed the four MoE references, and reached 3.039 GiB peak PyTorch-reserved memory. Whitepaper v0.3 documents the run.

Start interactive text chat with `.\run_chat.ps1`, or run one prompt with `.\run_chat.ps1 --prompt "Say hello." --max-new-tokens 6`. Use `/quit` to exit. The chat runner uses greedy decoding and a bounded context. It re-prefills the conversation on each turn and streams the output head in chunks on each generated token.
