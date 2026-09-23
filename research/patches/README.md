# llama.cpp patches — what they are and what they do

These patches make a 35-billion-parameter Mixture-of-Experts model (Qwen3.6-35B-A3B, ~13 GB quantized) run fast on a PC whose
GPU has only 4 GB of memory (GTX 1650 SUPER, no tensor cores) and a 6-core CPU with 16 GB of RAM. Stock llama.cpp runs it, but
slowly: about 48 tokens/s reading a long prompt. With these patches it reads prompts at about 500 tokens/s and writes at
55-70 tokens/s, and every change was checked for accuracy before it shipped.

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

### 1. A GPU cache for the experts (0001-0007)
The core idea. The GPU keeps a least-recently-used cache of expert weights ("slots", ~21-26 per layer). An expert that is in
the cache is computed on the GPU; a miss is computed on the CPU from system RAM while the upload happens in the background, so a
miss never waits for the bus. About half of all expert uses hit the cache.
- **0001** the cache itself: slots, background uploads, admission only after repeated misses (so one-off experts do not evict
  hot ones), 1-4 token batches (needed for speculative decoding), and CPU/GPU running at the same time. Switch: `--moe-expert-cache N`.
- **0002 / 0004** fill the cache from the prompt's routing so the first answer tokens start warm (`--moe-expert-cache-warm`).
- **0003** lets short n-gram drafts roll back cheaply (speculative decoding bookkeeping).
- **0005 / 0006** "cache-aware routing": nudge the router toward cached experts. Changes the output, so it is **off**.
- **0007** `LLAMA_MOE_CACHE_SYNC=1` makes the cache deterministic. A test tool: with it on, a patched build must produce
  byte-identical text to the reference, which is how "exact" changes are proven.

### 2. Keep the CPU and the GPU busy at the same time (0008-0010, 0017, 0018, 0020)
- **0008** computes the always-on "shared expert" on the GPU while the CPU works on the routed experts. **0009** a faster GPU
  kernel for a small copy in the model's recurrent layers. **0010** starts CPU-to-GPU copies without stalling the CPU.
  Together: part of a +5.4% decode gain, output byte-identical.
- **0017** for big prompt batches, upload whole expert tensors without first reading the router's choices back to the CPU.
- **0018 / 0020** drop a needless wait before copies that the GPU already orders correctly (host overhead per layer 89 -> 63 us).

### 3. Fast prompt reading: "prefill mode" (0011-0013, 0024-0030)
Reading a prompt processes thousands of tokens at once, where the per-token cache no longer helps.
- **0011** prefill mode (`--ubatch-prefill N`, served as `-ubp 2048`): for a big prompt, release the cache slots, use that GPU
  memory to process the prompt in large chunks, then bring the cache back for writing. **0012 / 0013** make the hand-back reliable.
  With 0014 below: 9,279-token prompt 47 -> 404 tokens/s.
- **0024** overlaps uploading the next layer's experts with computing the current one on a second GPU stream
  (`GGML_SCHED_MOE_PREFETCH=1`): prompt reading +19-23%, results identical. **0029 / 0030** fix two costs it first had
  (planning overhead on every token; a buffer that blocked the cache from coming back).
- **0025-0028** diagnostic switches used to find those two costs; they do nothing unless set.

### 4. GPU code for a card without tensor cores (0014-0016, 0019, 0021)
The GTX 16xx cards report the same generation as RTX 20xx cards but lack their tensor cores, so upstream picks kernels that are
slow here.
- **0014** matrix multiply uses the dp4a integer path (build flag `-DGGML_CUDA_MMQ_NO_MMA=ON`); the enabler for fast prefill.
- **0019** attention uses the "tile" kernel for batches of 32+ tokens (`GGML_CUDA_FA_TILE_MIN_BATCH=32`) and **0021** also for
  long contexts (`GGML_CUDA_FA_MMA_MAX_KV=4096`): writing after a 9k-token prompt +6-7%.
- **0015** (never use the tensor-core attention kernel) and **0016** (size expert matrix tiles per expert) are opt-in
  alternatives that measured worse or neutral; they stay off.

### 5. Cheaper speculative decoding (0022-0023)
The model has a small built-in "draft" head (MTP) that guesses the next tokens; the full model then checks the guesses in one pass.
- **0022** the draft head scores only the ~49k most common tokens instead of all 248k (`LLAMA_MTP_VOCAB_FILE=...`). The full
  model still checks every guess against the full vocabulary, so the output is unchanged; only the draft gets cheaper.
  Writing speed +5-10%. **0023** fixes a crash during the startup memory-fit pass.

## How accuracy is protected
Every patch is either **exact** (same output bytes as before, proven with 0007's deterministic mode) or **non-inferior** (the
next-token probabilities stay as close to a near-lossless reference model as before, measured as KL divergence), and it must be
faster by more than the run-to-run noise. Changes that alter the output by design (0005) stay off.

## Build and apply
```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout e613ef2                       # the upstream base of the series ("hexagon: enable I32 GET_ROWS (#29116)")
git am /path/to/research/patches/mainline-series/*.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_MMQ_NO_MMA=ON
cmake --build build -j
```
Run it with the serving command in `SERIES.md` (environment switches + `llama-server` flags).

## Glossary
- **decode / prefill**: writing tokens one step at a time / reading the prompt in large batches.
- **expert, router**: the small sub-networks of a MoE layer, and the part that picks 8 of them per token.
- **KV cache**: the stored attention state of earlier tokens; it grows with context length.
- **MTP / speculative decoding**: a cheap draft guesses several tokens, the full model verifies them in one pass.
- **KL divergence (KLD)**: how far a model's next-token probabilities are from a reference; 0 = identical.
- **dp4a / MMA**: integer dot-product instructions every recent NVIDIA GPU has / tensor-core instructions the GTX 16xx lacks.
