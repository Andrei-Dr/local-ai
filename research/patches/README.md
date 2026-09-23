# llama.cpp patches — what they are and what they do

These patches make a 35-billion-parameter Mixture-of-Experts model (Qwen3.6-35B-A3B, ~13 GB quantized) run fast on a PC whose
GPU has only 4 GB of memory (GTX 1650 SUPER, no tensor cores) and a 6-core CPU with 16 GB of RAM. Every change was checked for accuracy before it shipped. Measured now (LEGACY = our first served build, medians from the run
ledger):

<!-- BEGIN GENERATED: headline (bench/docgen.py; edit the source, not this block) -->
prompt reading ~511 tokens/s at 9.3k tokens (LEGACY 47, 10.8x); writing 55-70 tokens/s (64-70 on short prompts, 55 after a 9.3k-token prompt)
<!-- END GENERATED: headline -->

This page explains the patches in plain language. `SERIES.md` is the lab ledger: the status of each patch, the switch that turns
it on, the measurements behind it, and the exact command we serve with.

## The problem, in one paragraph

A Mixture-of-Experts model has 256 small "expert" networks per layer and uses only 8 of them for each token. All the experts
together (~11.7 GB) do not fit on the GPU, so they live in system RAM and the CPU computes them; everything else (attention, the
shared layers, ~1.5 GB) lives on the GPU. Each token therefore bounces between CPU and GPU 40 times (once per layer), and the
speed is set by three narrow pipes: system RAM bandwidth (~38 GB/s, shared by the CPU and by every copy to the GPU), the PCIe
bus (~12 GB/s), and GPU memory (192 GB/s, 4 GB). Almost every patch here either keeps useful data on the GPU, overlaps work that
used to wait, or picks GPU code that suits a card without tensor cores.

## Folder map

| path | what it is | status |
|---|---|---|
| `mainline-series/0001-0030` | **the current patch series**, one linear chain on upstream llama.cpp (`git am` in order) | what we build and serve |
| `SERIES.md` | status, switches, evidence and the serving command for every patch in `mainline-series/` | read with this page |
| `experimental/` | parked or superseded work: pre-gated expert prefetch, its diagnostics, a cache warm-up variant, and the first cuts of patches that later landed as 0021 and 0024 | not served |
| `box-series/` | the earlier series on the PrismML fork of llama.cpp, now used only for the ternary Bonsai model that mainline cannot load | history |
| `pr27861.diff` | upstream llama.cpp PR #27861, the original expert-cache proposal our cache started from | reference |
| `pr28549.diff`, `pr28549-ported.diff` | upstream PR #28549 (separate graph-result buffers, so speculative draft steps can reuse CUDA graphs) and our port to the fork | merged upstream |
| `pr28739.diff` | upstream PR #28739 (the scheduler skips empty expert-id tensors) | merged upstream |
| `pr26079.diff` | upstream PR #26079 (mat-vec vs matrix-kernel switch points; tuned upstream for newer GPUs only) | reference |
| `moecache-p2/p3/p4.diff` | early stages of our cache on the fork: gated admission (p2), 1-4 token batches (p3), CPU/GPU overlap barrier (p4) | folded into 0001 |
| `mainline-moecache-e613ef2.diff` | the whole cache squashed onto upstream commit e613ef2, before it was split into the series | superseded by `mainline-series/` |
| `bundles/` | git bundles of the fork branches (restorable history) | archive |

## The series, by what it does

Generated from `series.toml` (one entry per patch file; a patch without an entry fails the check).

<!-- BEGIN GENERATED: patch-guide (bench/docgen.py; edit the source, not this block) -->
### 1. A GPU cache for the experts (0001-0007)
The core idea. The GPU keeps a least-recently-used cache of expert weights ("slots", ~20-26 per layer). An expert in the
cache is computed on the GPU; a miss is computed on the CPU from system RAM while its upload happens in the background, so a miss
never waits for the bus. About half of all expert uses hit the cache.

- **0001** The cache itself: slots on the GPU, background uploads, admission only after repeated misses (so one-off experts do not evict hot ones), 1-4 token batches for speculative decoding, and CPU and GPU running at the same time. Switch: ``--moe-expert-cache N``. Status: SERVED.
- **0002** Fills the cache from the prompt's routing, so the first answer tokens start warm. Switch: ``--moe-expert-cache-warm``. Status: SERVED.
- **0003** Lets short n-gram drafts roll back cheaply (speculative decoding bookkeeping). Status: SERVED.
- **0004** Fix: warms the cache only from single-sequence batches. Status: SERVED.
- **0005** Cache-aware routing: nudges the router toward experts already cached. Changes the output, so it stays off. Switch: ``--moe-expert-cache-bias`, off`. Status: SERVED.
- **0006** Fix: cache-aware routing scales only plain, non-negative router probabilities. Status: SERVED.
- **0007** Makes the cache deterministic. A test tool: with it on, a patched build must produce byte-identical text to the reference, which is how exact changes are proven. Switch: `env, off`. Status: SERVED.

### 2. Keep the CPU and the GPU busy at the same time (0008-0020)
Each token alternates between CPU work (expert misses) and GPU work 40 times. These patches remove waits so the two
overlap; 0008-0010 together gave part of a +5.4% decode gain with byte-identical output.

- **0008** Computes the always-on shared expert on the GPU while the CPU works on the routed experts. Status: SERVED.
- **0009** A faster GPU kernel for a small copy in the model's recurrent (delta-net) layers. Status: SERVED.
- **0010** Starts CPU-to-GPU copies on the GPU's stream without stalling the CPU. Status: SERVED.
- **0017** For big prompt batches, uploads whole expert tensors without first reading the router's choices back to the CPU. Switch: `ON; `GGML_SCHED_MOE_READBACK=1` = old`. Status: STABLE.
- **0018** Can skip a needless wait before copies the GPU already orders correctly (opt-in; 0020 makes it the default). Switch: `see 0020`. Status: STABLE (on via 0020).
- **0020** Makes 0018 the default: host overhead per layer drops from ~89 to ~63 microseconds. Switch: `ON`. Status: STABLE.

### 3. Fast prompt reading: prefill mode (0011-0030)
Reading a prompt processes thousands of tokens at once, where the per-token cache no longer helps. With 0014, prefill
mode took a 9,279-token prompt from 47 to 404 tokens/s; the upload overlap (0024) adds +19-23% with identical results.

- **0011** Prefill mode: for a big prompt, release the cache slots, use that GPU memory to process the prompt in large chunks, then bring the cache back for writing. Switch: `CLI, off unless `-ubp``. Status: STABLE.
- **0012** Fix: returns the GPU's temporary memory pools before the cache slots come back. Switch: `with 0011`. Status: STABLE.
- **0013** Fix: suspending the cache detaches its tensors so resuming re-allocates them. Switch: `with 0011`. Status: STABLE.
- **0024** Overlaps uploading the next layer's experts with computing the current one, on a second GPU stream. Switch: ``GGML_SCHED_MOE_PREFETCH=1` for serving (unset = off)`. Status: STABLE.
- **0025** Diagnostic: mode 2 plans the overlap without the second stream (isolates the layout change). Switch: `env, off`. Status: STABLE.
- **0026** Diagnostic: logs the planned splits and filters them by routed-token count. Switch: `env, off`. Status: STABLE.
- **0027** Diagnostic: plans only splits whose first node name matches a substring. Switch: `env, off`. Status: STABLE.
- **0028** Diagnostic: turns the mode on but skips the planning loop (to bisect its cost). Switch: `env, off`. Status: STABLE.
- **0029** Fix: the overlap planner tests the operation first and caches the GPU verdict, removing a ~7% cost on every token. Switch: `with 0024`. Status: STABLE.
- **0030** Fix: plans only graphs big enough for whole-tensor uploads, so a planned buffer no longer blocks the cache from coming back. Switch: `with 0024`. Status: STABLE.

### 4. GPU code for a card without tensor cores (0014-0021)
GTX 16xx cards report the same generation as RTX 20xx cards but lack their tensor cores, so upstream picks kernels that
are slow here. Writing after a 9k-token prompt got +6-7% from 0021 alone.

- **0014** Matrix multiply uses the dp4a integer kernels on Turing cards without tensor cores; the enabler for fast prompt reading. Switch: `CMake `-DGGML_CUDA_MMQ_NO_MMA=ON``. Status: STABLE.
- **0015** Attention never selects the tensor-core kernel. Measured worse than 0019 for writing; stays off. Switch: ``GGML_CUDA_FA_NO_MMA=1`, off`. Status: STABLE (off).
- **0016** Sizes the expert matrix tiles by tokens per expert. Neutral at the serving config; stays off. Switch: ``GGML_CUDA_MMQ_MOE_EXPERT_COLS=1`, off`. Status: STABLE (off).
- **0019** Attention uses the tile kernel for batches of 32+ tokens (prompt reading). Switch: ``GGML_CUDA_FA_TILE_MIN_BATCH=32` for serving`. Status: STABLE.
- **0021** Attention also uses the tile kernel once the context is long (writing after a long prompt). Switch: ``GGML_CUDA_FA_MMA_MAX_KV=4096` for serving (unset = off)`. Status: STABLE.

### 5. Cheaper speculative decoding (0022-0023)
The model has a small built-in draft head (MTP) that guesses the next tokens; the full model checks the guesses in one
pass. Cutting the draft's vocabulary made writing 5-10% faster with the output unchanged.

- **0022** The draft head scores only the ~49k most common tokens instead of all 248k. The full model still checks every guess against the full vocabulary, so the output is unchanged; only the draft gets cheaper. Switch: ``LLAMA_MTP_VOCAB_FILE=/ai/models/mtp-Qwen3.6-35B-A3B-vocab49k.bin` for serving (unset = full head)`. Status: STABLE.
- **0023** Fix: the cut-down draft head skips graphs built over weights that are not allocated yet (startup memory-fit pass). Switch: `with 0022`. Status: STABLE.
<!-- END GENERATED: patch-guide -->

## How accuracy is protected
Every patch is either **exact** (same output bytes as before, proven with 0007's deterministic mode) or **non-inferior** (the
next-token probabilities stay as close to a near-lossless reference model as before, measured as KL divergence), and it must be
faster by more than the run-to-run noise. Changes that alter the output by design (0005) stay off.

## Build and apply

See [QUICKSTART.md](../../QUICKSTART.md): `stable/build.sh` applies the series and builds llama-server.

## Glossary
- **decode / prefill**: writing tokens one step at a time / reading the prompt in large batches.
- **expert, router**: the small sub-networks of a MoE layer, and the part that picks 8 of them per token.
- **KV cache**: the stored attention state of earlier tokens; it grows with context length.
- **MTP / speculative decoding**: a cheap draft guesses several tokens, the full model verifies them in one pass.
- **KL divergence (KLD)**: how far a model's next-token probabilities are from a reference; 0 = identical.
- **dp4a / MMA**: integer dot-product instructions every recent NVIDIA GPU has / tensor-core instructions the GTX 16xx lacks.
