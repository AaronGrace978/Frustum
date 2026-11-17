# Frustum_loader

Streaming inference for checkpoints far larger than GPU memory. The reference target is DeepSeek-V4.1-Flash (about 510 GB on disk) on one 16 GB GPU, read from an external SSD.

Current code: v0.10 (bit-identical fused activation kernel and fewer GPU synchronisations: about 2-4% faster decode; research notes and roadmap in the whitepaper). Measured on an RTX 5060 Ti with the checkpoint on a USB SSD: DeepSeek decode **5.69 → 4.06 s/token** (v0.8, unbuffered reads), and follow-up turns start replying up to **51% sooner** (v0.9, batched follow-up prompts). See below. Live v0.5.1 benchmarks from a first test session are further down. The numbers under "History and recorded evidence" were measured with v0.4. Run the verification and benchmark steps below to get current numbers for your machine.

**Whitepaper:** [`whitepaper/Frustum_loader_Whitepaper_v0.11.0.pdf`](whitepaper/Frustum_loader_Whitepaper_v0.11.0.pdf) documents the design, the v0.5.1 live benchmarks, v0.7's I/O/roofline tooling and the v0.8 and v0.9 measured results. Details and evidence: [`CHANGES_v0.10.md`](CHANGES_v0.10.md) and `v10_logs/`; [`CHANGES_v0.9.md`](CHANGES_v0.9.md) and `v09_logs/`; [`CHANGES_v0.8.md`](CHANGES_v0.8.md) and `v08_logs/`; [`CHANGES_v0.7.md`](CHANGES_v0.7.md) and `v07_logs/`. Rebuild the current whitepaper with `python whitepaper/build_whitepaper.py`. Earlier editions are in `whitepaper/archive/`.

## Quick start

    .\run_verify.ps1                      # tests, drive speed, v0.4 parity, timed chat
    .\run_selftest_v07.ps1                 # v0.7 I/O, cache, spill, and roofline tests
    .\run_chat.ps1                        # interactive chat
    .\run_chat.ps1 --prompt "Say hello." --max-new-tokens 6

The launchers look for Python in this order: `FRUSTUM_PYTHON`, `.venv\Scripts\python.exe` in this folder, then the original Codex environment. To set up a fresh machine:

    py -3.13 -m venv .venv
    .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
    .venv\Scripts\pip install -r requirements.txt

The model folder defaults to the parent of this folder. Set `FRUSTUM_MODEL` or pass `--model` to point elsewhere. Weights and the CUDA environment are not bundled.

## Qwen3-Next-80B-A3B

**First live run (ActivatePrime 0.1.5, 163 GB checkpoint on the T7, RTX 5060 Ti, 32 GB RAM):** 88 tokens in 132.3 s (0.79 tokens/s) on the first turn, first token after 32 s including loading; a 23-token follow-up reused 100 cached tokens, started replying after 26 s and produced 96 tokens in 195.4 s (0.56 tokens/s). Placement: dense 3.8 GiB on the GPU, experts 6.9 GiB GPU + 14.1 GiB RAM. Single app runs; event log in [`v11_logs/qwen_next_first_live_run.txt`](v11_logs/qwen_next_first_live_run.txt), whitepaper Section 20.


[`qwen_next_chat.py`](qwen_next_chat.py) runs [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) (163 GB BF16, 48 layers, 512 experts with 10 active per token) on the same tiered expert cache as DeepSeek and GLM. About 3.7 GiB of dense weights stay on the GPU. The 6 MiB routed experts stream through the GPU tier, a host-RAM tier (`FRUSTUM_QWEN_HOST_GIB`, default 85% of free RAM above 6 GiB, up to 16 GiB) and the drive. The token embedding is read one row per token. It is a plain-PyTorch implementation of both layer types: gated full attention (every fourth layer) and Gated DeltaNet linear attention with a fixed-size recurrent state (the other 36). Follow-up turns that extend the previous token ids only process the new tokens. ActivatePrime 0.1.5 exposes it as "Qwen3-Next 80B".

[`qwen_next_parity.py`](qwen_next_parity.py) checks it against the transformers 4.57 reference on a tiny random two-shard checkpoint (prefill, single-token decode and a multi-token append, with evictions and RAM-tier hits). Float32 agrees to 5e-6 of the logit scale, with the same argmax at every position. In bf16 it is as close to the float32 result as the reference's own bf16 path, or closer.

    python qwen_next_chat.py --prompt "Say hello." --max-new-tokens 8
    set QWEN_PARITY_TRANSFORMERS=<folder with transformers 4.57>  &  python qwen_next_parity.py

## What v0.10 changes

Exact (bit-identical) cuts to per-token GPU and Python overhead: a fused activation kernel (0.274 to 0.036 ms per call, self-checked against PyTorch and self-disabling on any difference; `FRUSTUM_FUSED_ACT=0` turns it off), one shared quantization for w1 and w3, and CPU-side scale checks that remove 761 GPU synchronisations per token. Decode is about 2-4% faster with identical tokens: most of that work was already hidden behind drive reads. The whitepaper's new Section 20 records the research from this round: the DSpark analysis (paused: verifying drafts needs 3.4x the experts), game-engine streaming ideas, native FP4 on the 5060 Ti, hardware outlook and a roadmap. See [`CHANGES_v0.10.md`](CHANGES_v0.10.md).

## What v0.9 changes

Follow-up turns used to feed their new prompt tokens through all 40 layers one token at a time, paying the per-layer expert-read stall for every token. Attention can now extend the conversation state by many tokens in one pass, so a follow-up prompt goes through in one pass (or a few, for long ones). Full details: [`CHANGES_v0.9.md`](CHANGES_v0.9.md).

| Follow-up, full 40-layer model | v0.8 (one token per pass) | v0.9 (`--append-chunk 32`, default) |
|---|---|---|
| "Say one word." (8 new tokens) | 35.8 s to first token | **22.1 s (−38%)** |
| A 29-token question | 123.7 s | **60.1 s (−51%)** |
| Reply tokens | | Identical to v0.8 on every turn |

- Each appended token gets exactly the sliding window, compressed groups and attention arithmetic a one-token decode step would give it. Unit tests check the window rows and compressor state bit for bit against decode; on the real model, layer 0 agrees with decode on 13 of 16 positions bit for bit (prefill differs from decode 17× more).
- `--append-chunk N` / `FRUSTUM_APPEND_CHUNK` sets tokens per pass; `1` restores v0.8. `--followup TEXT` (repeatable) scripts a multi-turn run in one process.
- `long_context_check.py --append-chunk 8` compares batched append with decode beyond the 128-token window. Note: that check's own tolerance already fails at 4 layers on unchanged v0.8 code (prefill vs decode differ by ~4.6%); see the changes file.

    python frustum_chat.py --prompt "Do you like Dino Buddies?" --followup "Say one word." --append-chunk 1   # v0.8 behaviour
    python frustum_chat.py --prompt "Do you like Dino Buddies?" --followup "Say one word."                    # v0.9 default
    python long_context_check.py --tokens 160 --tail 24 --layers 4 --append-chunk 8

## What v0.8 changes

v0.8 is the first release measured end to end on the target machine (RTX 5060 Ti 16 GB, 32 GB RAM, DeepSeek V4.1 Flash on a Samsung T7 at a 10 Gbps USB port). Every number here is a `bench_e2e.py` median over 3 or more fresh-process runs, with token ids identical across configurations. Full tables: [`CHANGES_v0.8.md`](CHANGES_v0.8.md).

- **Unbuffered reads, on by default on Windows.** `--reader unbuffered` (or `FRUSTUM_READER`) reads the checkpoint with `FILE_FLAG_NO_BUFFERING`. Buffered reads made Windows read 27% more than requested and keep a second copy of every expert in its file cache, which with 32 GB of RAM never produced a warm-cache benefit. Unbuffered reads fetch exactly the requested bytes (863 vs 580 MB/s effective) and let the automatic host RAM tier take 85% of spare RAM instead of 50%. **Decode 5.69 → 4.06 s/token (−29%), time to first token 63.0 → 58.1 s.** `--reader direct` restores v0.7.
- **Where the rest goes.** `tools/profile_decode.py` shows each decode token spends about 2.8 s waiting for expert reads and about 2 s on GPU work (expert calls, attention, hyper-connection mixes), alternating layer by layer: the drive idles while the GPU works and vice versa.
- **Measured and left off.** Pinned staging with copy-stream uploads (`FRUSTUM_ASYNC_UPLOAD=1`) measured 4.33 s/token, and gated next-layer prefetch 4.42 s/token, both slower than the 4.07 s/token default. The next-layer router guess is right 72% of the time, which doubles read/compute overlap but wastes about 0.37 GB/token on wrong guesses. Both stay available for A/B runs.
- **Check the drive link first.** The same T7 measured about 42 MB/s on one port and 909 MB/s on a 10 Gbps port. Four parallel reads reach the drive's ceiling; more do not help.

    .\run_bench.ps1 --configs direct patch             # v0.7 buffered reader vs v0.8 default
    .\run_bench.ps1 --configs patch async_upload gated  # the opt-in experiments
    python tools\profile_decode.py --prompt "Say hello." --max-new-tokens 24

## What v0.7 changes

v0.7 is a measurement-and-I/O release. It does not change model weights, routing decisions, attention, or the reference path. Its purpose is to show whether a decode step is actually limited by the checkpoint drive or the GPU before spending effort on the wrong optimization. The new paths were first exercised in a no-GPU sandbox; the v0.8 section above has the first measurements on the real checkpoint (v0.7 defaults: 5.69 s/token).

- **End-to-end I/O accounting.** `io_ledger.py` records logical expert bytes, first-time versus repeated reads, queue depth, drive-busy time, blocking, overlap, and speculative-read waste. `os_cache.py` also collects best-effort physical device read counters, so buffered Windows reads are not mistaken for physical disk traffic.
- **Roofline and reproducible benchmarks.** `bench_e2e.py`, `run_bench.ps1`, and `roofline.py` run cold/warm configurations at least three times and report median/p95 latency, logical/physical bytes, effective throughput, drive utilisation, and whether the run appears storage- or compute-bound. Routing traces can now be replayed with `decode_cache_replay.py`.
- **Optional verified spill tier.** `--spill-dir` copies previously read expert bytes to a faster local drive. Every entry is tied to the checkpoint identity and protected by CRC-32; a mismatch, truncation, or changed checkpoint falls back to the original weights. It is disabled unless a spill directory is explicitly supplied.
- **Safer expert-cache behaviour.** A cache now protects all experts needed by the current layer from evicting one another, coalesces nearby tensors into fewer positioned reads where the shard layout permits, and can shrink the host cache if free RAM falls below `--ram-floor-gib` (3 GiB by default).
- **Prediction is measurable before it reads.** Next-layer prediction still never changes the experts selected by the real router. It now supports `shadow` (score only), `gated` (the default when prediction is enabled; read only after measured cold-guess precision reaches 0.6), and `always` (the v0.6-style policy). Guesses are issued only after the current layer's required reads, so they cannot queue ahead of selected experts. Prediction remains off by default.

Start with the checks, then take a baseline before enabling optional features:

    .\run_selftest.ps1 ; .\run_selftest_v07.ps1
    .\run_verify.ps1
    .\run_bench.ps1 --configs baseline patch shadow
    python roofline.py bench_runs\<timestamp>\patch_cold_0.json bench_runs\<timestamp>\patch_warm_0.json

If the roofline says first-time storage reads dominate, try the spill tier on a faster internal SSD: `.\run_bench.ps1 --configs patch spill --spill-dir D:\frustum_spill`. If it says repeated reads dominate, tune `--host-cache-gib` and replay a recorded trace. Keep the resulting token IDs: they must match across configurations.

## What v0.6 changes

v0.5 moved the bottleneck. With the drive on a USB 3.x link, a large share of each token is now spent on the GPU, and most of that was not arithmetic: every backend decoded each packed weight matrix into a full FP32 copy on every token, then multiplied one row of activations by it.

**Fused decode kernels** (`fused_kernels.py`). For decode steps (up to 8 rows of activations), one CUDA kernel reads the packed bytes once, decodes them in registers and accumulates the dot products. No FP32 copy of the matrix is ever written. Formats: GGUF Q4_K, Q5_K, Q6_K and Q8_0 (Phi-4, GLM), FP8 E4M3 with 32x32 E8M0 blocks (DeepSeek dense and shared experts), packed FP4 E2M1 with per-32 E8M0 scales (DeepSeek routed experts). The source is compiled at load time with NVRTC, which already ships in the PyTorch CUDA wheel, so nothing new is installed.
- Decoded weights are bit-identical to the PyTorch reference decoders (same operations, same order, FMA off; FP16 rounding where the reference rounds). Only the summation order differs from cuBLAS.
- Before anything uses it, the first call for each matrix shape is also run on the reference path and compared. A mismatch, compile failure or launch error switches the fused path off for the process and prints why. Prefill keeps the reference path.
- `FRUSTUM_FUSED=0` or `frustum_chat.py --no-fused` turns it off.
- Checked without a GPU by `tools/fused_host_check.py`, which compiles the same CUDA source with g++ (32 threads per warp so the warp shuffles behave as on hardware). Every column of every format is bit-exact against NumPy ports of the reference decoders, and those ports match the official `gguf` dequantizer bit for bit.

**Next-layer expert prediction** (DeepSeek, `streamed_moe.py`). During a decode step, layer N's MoE input is also run through layer N+1's router, and reads for its top experts start in the background straight away, behind layer N's own reads. When layer N+1 routes for real, reads that were guessed right are already done or under way; wrong guesses are cancelled if they have not started. The real router still decides which experts run, so a wrong guess costs drive time, never accuracy. Recall (the share of routed experts that were guessed) is reported in `--evidence` and `/stats`. `FRUSTUM_PREDICT_NEXT=N` or `--predict-next N` turns it on with N guesses per layer (6 matches the router).

**Prediction is off by default.** The first live DeepSeek turn with it on took about 7 minutes against about 3 minutes on v0.5.1. DeepSeek is drive-bound, and wrong guesses start reading at once, so they hold up the real reads. It stays off until `run_speed_check.ps1 --deepseek` shows its recall and its effect. Fused kernels are on by default, including in ActivatePrime; `--legacy` turns them off.

### Measure it

    .\run_fused_check.ps1                 # kernels vs reference on random weights, with timings (seconds, no models)
    .\run_speed_check.ps1                 # Phi-4 and GLM: reference then v0.6, s/token and token match
    .\run_speed_check.ps1 --deepseek      # adds DeepSeek with prediction recall (long)

`speed_check_results.json` records both runs. Token ids can differ after many tokens because the summation order changed and a near-tie can flip; the first tokens should match.

## What v0.5 changed

The v0.4 decode profile showed where a token's 58 seconds went: 2.3 GB of routed experts arriving from the drive at a flat 37 MB/s, with 5.6 GB of dense weights and 0.16 GB of router weights re-read on every token as well. Model arithmetic was not the bottleneck, so v0.5 changes where bytes come from and leaves the arithmetic alone.

| Area | v0.4 | v0.5 |
|---|---|---|
| Disk reads | New 8 MiB mmap window per chunk, shard reopened per tensor | Persistent handles, large positioned reads straight into the destination (`RangeReader`) |
| Dense weights, routers | 256 MiB / 32 MiB LRU, re-read every token | Resident on the GPU when they fit |
| Output head | Streamed by default: 1.3 GB read per token | Resident by default |
| Routed experts | One GPU LRU, one read at a time | GPU tier, optional host-RAM tier, 4 reader threads (`expert_cache.py`) |
| Eviction | LRU | Frequency-ranked when the GPU tier is smaller than one token's working set, LRU otherwise (`cache_policy.py`) |
| Dense FP8 decode | Three full-size FP32 temporaries | One, same products |
| Follow-up turns | Whole conversation re-prefilled | Processed prefix reused when the token ids match exactly |
| Output | Printed at the end, greedy only | Streamed; optional temperature, top-k, top-p |
| Launchers | Hard-coded interpreter path | `FRUSTUM_PYTHON` / local `.venv` / original path |

`--legacy` on `run_chat.ps1` restores the v0.4 placement and the mmap reader for a same-day A/B run.

## v0.5.1 live benchmarks

Hardware: NVIDIA RTX 5060 Ti 16 GB, Samsung T7 SSD on USB 3.x (USB-C), Windows.

### Drive link matters

The T7 over a USB 2.0-speed link delivered about 50 MB/s and dominated every timing. Over USB-C on a USB 3.x port, the same drive reads at roughly 500 MB/s — a 10× difference that cascades through every model.

### GLM-4.7-Flash (30 B MoE, ~12 GB)

| Metric | USB 2.0 link | USB 3.x (USB-C) |
|---|---|---|
| Cold load + 9-token reply | 262.1 s | 23 s |
| Effective cold read speed | ~50 MB/s | ~520 MB/s |

### Phi-4 (14 B, GPU-resident GGUF)

Cold load + 36-token reply: 53.8 s (~1.5 s per token). Generation speed matches the v0.4 baseline (1.4 s/token saved). The model fits on the GPU, so decode speed is limited by per-token dequantization in PyTorch, not the drive.

### DeepSeek V4.1 Flash (~510 GB, 40 layers, 384 routed experts per layer)

| Turn | Tokens | Wall time | Per token |
|---|---|---|---|
| First (cold) | 24 | 192.2 s | ~8.0 s |
| Second (warm cache) | 24 | 227.1 s | ~9.5 s |

Previous estimate with v0.4 at USB 2.0 speed: ~58 s per token. The USB-C link and the v0.5 tiered expert cache together deliver a **7× speedup**. The second turn is slightly slower because the growing conversation increases prefill work (prefix reuse still skips the already-fed tokens). With the LFU cache retaining frequently-used experts (~57% repeat rate between consecutive tokens), subsequent turns in a longer conversation should see further improvement as the working set stabilises.

These are first-session numbers from ActivatePrime 0.1.1 running Frustum v0.5.1. Formal verify and parity runs have not been done yet.

Two things from the earlier plan were deliberately not built:

- Caching decoded (FP32) weights. Nothing in the profile says decode time matters yet, and decoded weights take 4 to 8 times the VRAM of packed ones. Measure first with `--profile-decode`.
- Guessing the next layer's experts. With a host tier holding recent experts, the candidates a guess could add are the ones that are neither recent nor frequent, and a wrong guess spends drive time. The experts of the current layer are known exactly after routing and are read in parallel.

## Check your drive link first

Every cold layer in the v0.4 profile read at 37 to 38 MB/s. A T7 on a USB 3.x port should manage several hundred. A flat 37 is what a USB 2.0 link delivers, though the old mmap path could also be responsible. This tells you which:

    .\run_io_bench.ps1

It reads fresh tensors through the old path and the new one and prints MB/s for each. If large direct reads are still under about 60 MB/s, the limit is the port, cable or hub, and no software change will move it. If direct reads are fast and mmap is slow, the new reader has already fixed it.

## Placement

At startup the chat runner prints its plan, for example dense weights resident, output head resident, shared experts pinned, and the size and policy of the routed-expert cache. Relevant options:

| Option | Default | Meaning |
|---|---|---|
| `--gpu-cap-gib` | 12 | PyTorch allocator cap |
| `--reserve-gib` | 2 | Part of the cap left for activations and temporaries |
| `--dense`, `--output-head` | auto | `resident`, `streamed`, or let the plan decide |
| `--expert-cache-mib` | plan | GPU budget for packed experts |
| `--host-cache-gib` | auto | RAM tier behind the GPU cache; `0` disables. Auto uses half of free RAM above 6 GiB, at most 16 GiB |
| `--io-threads` | 4 | Background readers for experts |
| `--cache-policy` | auto | `lfu` or `lru` for the GPU expert tier |
| `--no-reuse` | off | Re-prefill every turn |
| `--temperature`, `--top-k`, `--top-p`, `--seed` | greedy | Sampling |

On a 16 GB card with the default cap, the always-used set takes about 8 GiB, which leaves roughly 2 GiB for routed experts: about 110 of the up to 240 a token touches. The host tier is what absorbs the rest, so a machine with more free RAM will read less from the drive. If you hit a CUDA out-of-memory error, raise `--reserve-gib` or pass `--dense streamed`.

In chat: `/reset` clears the conversation, `/stats` prints cache counters, `/quit` exits. Each reply ends with a line showing seconds per token, bytes read from disk and where expert requests were served from.

## Verifying

`.\run_verify.ps1` runs four steps and writes everything to `verify_runs\<timestamp>\`:

1. `selftest.py`: 28 unit tests on a synthetic checkpoint, including the chat template against DeepSeek's own encoder. No GPU or weights.
2. `bench-io`: the drive check above.
3. `layer0_block.py`: the layer-0 output hash, routing and stage fingerprints must equal `repro_baseline.json`, recorded with v0.4. This is the proof that v0.5 did not change arithmetic.
4. `frustum_chat.py --prompt "Say hello."` for six tokens: token ids must equal the saved v0.4 run, and the timings are listed beside it.

`run_repro_test.ps1` will report `same_source: false` against the old baseline because the sources changed. The other checks in it should still pass.

Status of the code as committed: the reader, cache policy, expert cache, placement planner, generation control flow and chat template pass their unit tests on CPU. The GPU path has not been executed since v0.4. Step 3 and step 4 are that test.

Each change is its own commit, so a failure can be localised:

    git log --oneline
    git revert <commit>        # undo one change and re-run verify

Two switches do the same without touching history: `FRUSTUM_ATTENTION=loop` restores the original attention loop for prefill, and `--legacy` restores the v0.4 placement and reader.

## What v0.5.1 added

| Change | Effect on arithmetic |
|---|---|
| Chat template checked against `encoding/encoding.py` (identical in chat mode); `--system` | None |
| FP4 expert decode through a byte-pair table, block scales applied in place | Bit-identical; `decoder_test` asserts it on the GPU at start-up |
| Batched sparse attention for prefill; decode keeps the original loop | Prefill may differ in the last bit |
| GLM backend on the shared tiered cache with read-ahead | None |
| `long_context_check.py` / `run_long_context.ps1` | New test: prefill vs decode schedules past the 128-token window |

`layer0_block.py` deliberately stays a separate implementation. It duplicates code from the main engine, but that is what makes the layer-0 hash an independent cross-check instead of the engine agreeing with itself.

## Is the output right?

Agreement so far is this implementation against itself and against NumPy re-derivations of single components. Whether all 40 layers match the published model is open. `reference_parity.py` measures it against a provider serving the same checkpoint, or against another Frustum run:

    .\run_chat.ps1 --prompt "Say hello." --max-new-tokens 12 --top-logprobs 10 --evidence ours.json
    .\run_parity.ps1 ours.json --api-base https://host/v1 --api-model NAME --endpoint completions

See the script header for caveats about chat templates and provider quantization.

## Baseline for the GGUF backends

`phi4_chat.py` is unchanged apart from inheriting the faster reader; `glm_chat.py` now also uses the shared tiered cache. Both models fit on a 16 GB card, where llama.cpp is the standard to compare against, and their saved decode speed here is about 1.4 s per token. That cost is per-use dequantization in PyTorch, which the loader changes do not touch. `baseline_llamacpp.py` runs `llama-bench` on the same file and prints both figures.

## Modules

| File | Role |
|---|---|
| `frustum_loader.py` | Header-only catalog, `RangeReader`, `FrustumLoader` byte cache, `bench-io` |
| `cache_policy.py` | Frequency ranking, tier bookkeeping, replay simulator (no torch) |
| `expert_cache.py` | GPU + host tiers and threaded reads for routed experts |
| `placement.py` | What stays resident, from checkpoint metadata |
| `generation.py` | Chat template and prefill / reuse / decode control flow (no torch) |
| `streamed_moe.py`, `layers0_3_block.py`, `layer0_block.py`, `block_primitives.py`, `sparse_rows.py`, `expert_correctness.py` | Model arithmetic and its tests |
| `fused_kernels.py`, `fused_check.py`, `speed_check.py`, `tools/fused_host_check.py` | v0.6 fused decode kernels, their GPU check and benchmark, the A/B speed check, the no-GPU source check |
| `frustum_chat.py` | Chat runner |
| `decode_cache_replay.py` | Replay recorded routing through cache policies and tier sizes |
| `verify_upgrade.py`, `selftest.py`, `long_context_check.py`, `reference_parity.py`, `baseline_llamacpp.py` | Verification |
| `benchmark_layers.py` | Original synthetic transfer benchmark, kept with its v0 cache |

---

# History and recorded evidence (v0.1 to v0.4)

Everything below was written for, and measured with, the earlier code. Sentences about launchers being machine-specific, streamed output heads and re-prefilling on every turn describe v0.4.

## v0.4 prototype notes

GPU: NVIDIA RTX 5060 Ti 16 GB. Validated with Python 3.13 and torch 2.11.0+cu128.
Working folder: G:\Models\DeepSeek-V4.1-Flash\Frustum_loader

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

- Header-only safetensors catalog and bounded mmap range reader (frustum_loader.py).
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

An empty application cache does not imply an empty Windows file cache. File bytes and read time here come from Frustum's bounded reader. Upload timing is host wall time for blocking copies and can include CUDA synchronization. The RAM budget counts retained payload bytes, not overall host memory usage.

The first comparison is saved in `glm_placement_results.json`. All three modes reproduced the saved Dino reply. In the split mode, a warm first token took 9.80 s and a follow-up with 34 cached prompt tokens took 5.33 s to its first token, versus 11.72 s with a full rebuild. The RAM tier reduced logical file reads, but its warm latency was higher in this run. See the per-turn counters for the other policies.

A reverse-order repeat in `glm_placement_reverse_results.json` measured 10.27 s warm for split and 11.20 s for baseline. The split follow-up took 5.28 s with prefix reuse and 12.42 s with a rebuild. First-run timings varied widely between orders. ActivatePrime's source defaults to split placement and keeps the host tier optional.

### Phi-4 smoke test

Run `.\run_phi4_test.ps1 --max-new-tokens 24` with Microsoft's `phi-4-Q4_K_S.gguf` and Phi-4 `tokenizer.json` in `G:\Models\Phi-4`. This backend uses Frustum_loader's bounded reader, keeps the packed weights on the GPU, and runs the Phi-4 forward pass and KV cache in PyTorch. It uses the `gguf` package for metadata and reference dequantization. The saved `phi4_chat_results.json` records the first 40-layer smoke test. Sampled quantization checks and finite logits passed. Full-model numerical parity remains open.

### Profile cached decode

Run `.\run_chat.ps1 --prompt "Say hello." --max-new-tokens 3 --output-head resident --profile-decode --evidence decode_profile_repeat.json` to capture two cached decode steps across all 40 layers. The profile records synchronized per-layer wall time, logical bytes by reader, and routed expert IDs. Synchronizing each layer can perturb timing. Logical bytes include reads served by the OS cache.

The saved `decode_profile_results.json` matched the first three token IDs of the original run. Prefill took 721.3 s. Cached decode took 279.9 s and 68.9 s. Each decode requested about 5.93 GB of expert weights, 5.62 GB of dense weights, and 0.16 GB of router weights. The resident head requested no additional bytes during decode and took roughly 33 ms per step.

Run `python decode_cache_replay.py decode_profile_results.json` to estimate demand-order LRU reuse from the recorded routes and checkpoint metadata. On the second recorded decode step, a 6 GiB expert cache retains an estimated 159 of 280 expert requests and cuts expert reads from 5.93 GB to 2.27 GB. This is a replay estimate. Measure actual latency and memory in a live run with `--expert-cache-mib 6144` before adopting it.

The 40-layer smoke test returned “Hello! 👋 How can” for “Say hello.” at the six-token limit. It took 1,117.56 seconds and peaked at 3.057 GiB of PyTorch-reserved memory. See chat_smoke_results.json for token IDs and memory measurements.

Run `.\run_layers0_3_test.ps1 --output layers0_3_repeat.json` to repeat the four-block check without overwriting the saved report. The recorded run matched the saved Layer 0 output, passed the four MoE references, and reached 3.039 GiB peak PyTorch-reserved memory. Whitepaper v0.3 documents the run.

Start interactive text chat with `.\run_chat.ps1`, or run one prompt with `.\run_chat.ps1 --prompt "Say hello." --max-new-tokens 6`. Use `/quit` to exit. The chat runner uses greedy decoding and a bounded context. It re-prefills the conversation on each turn and streams the output head in chunks on each generated token.

---

## Contributors

- **Aaron Grace** — Architecture, original engine (v0.1–v0.4), layer-0 block, MoE pipeline, sparse attention, GGUF backends, ActivatePrime
- **OpenAI** — Codex environment and tooling used during development
- **Claude (Anthropic)** — v0.5 to v0.10 engineering: fused activation kernel, batched follow-up prompts, unbuffered reader and v0.8 measurements, I/O ledger and roofline tooling, fused decode kernels, next-layer expert prediction, I/O rewrite, tiered expert cache, placement planner, generation control flow, cache policy, batched attention, expert decode, GLM cache migration, verification harness, ActivatePrime integration
