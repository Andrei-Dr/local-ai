# KV Cache Tiering: Why It Does Not Work Like MoE Expert Caching (and What Does)

## 1. First Principles: Dense Access vs. Sparse Access

Standard multi-head attention computes `softmax(QK^T)V` over **every** past key for **every** new token. There is no locality to exploit — the query must touch the full KV set before it knows which entries matter. Contrast with MoE routing: a gate picks 8 of 256 experts (3% fan-out) per token, and the choice is known *before* the read.

**Bandwidth cost per generated token across PCIe 3.0 x16 (~12 GB/s pinned):**

| Context length | KV bytes read (20 KB/tok) | PCIe transfer time | Fraction of decode budget at 10 tok/s |
|---|---|---|---|
| 4K | 80 MiB | 6.7 ms | 6.7% |
| 8K | 160 MiB | 13 ms | 13% |
| 32K | 640 MiB | 53 ms | 53% — decode-bound |
| 128K | 2.5 GiB | 213 ms | pipeline-stalled |

At 32K context the KV fetch alone consumes half a decode step on your PCIe 3.0 bus. MoE expert loads are ~40 MiB each and overlap with compute on other layers; KV is needed all-at-once, in the critical path of the single attention operation that produces the next token's logits. This is the fundamental asymmetry: **MoE is sparse and predictable; dense attention is dense and unpredictable.**

Hybrid architectures (Qwen3.6-35B-A3B) change the math: only 10 of 40 layers use full attention (2 KV heads x 256 dims = 1 KB/token/layer → ~10 KB/token for the attention layers alone, plus ~20 KB/token overhead for per-head norms/biases → the stated 20 KB/token). The 30 Gated-DeltaNet layers carry a fixed 62.8 MiB recurrent state regardless of context length. This hybrid structure makes the KV cache *smaller* than in a pure-attention model but does *not* change the dense-access pattern within the attention layers that do exist.

## 2. Published Systems That Treat KV as Evictable / Tiered / Retrieved

### (a) Eviction by importance or recency

| System | Mechanism | Quality cost | Lossless? | Ref |
|---|---|---|---|---|
| **H2O** | Cumulative attention score ("heavy hitters") + recent window | <1 ppl on OPT/LLaMA at 20% budget | No | arXiv:2306.14048 (NeurIPS 2023) |
| **StreamingLLM** | Keep initial "attention sink" tokens + sliding window | Streaming-only; loses mid-context | No | arXiv:2309.17453 (ICLR 2024) |
| **Scissorhands** | Persistence-of-importance hypothesis; historical accumulation | Comparable to H2O | No | arXiv:2305.17118 (NeurIPS 2023) |
| **SnapKV** | Observation-window attention pooling over last 64 tokens | <1% drop on LongBench | No | arXiv:2404.14469 (NeurIPS 2024) |
| **PyramidKV** | Layer-varying budget (more cache in early layers, less in late) | Matches SnapKV with fewer total KVs | No | arXiv:2406.02069 (2024) |

All eviction methods permanently discard tokens. For tasks requiring precise retrieval from mid-context (needle-in-haystack), accuracy degrades. Not applicable to hybrid models where the attention layers already see only a small KV set.

### (b) Keep everything, fetch only what the query needs

| System | Mechanism | Quality cost | Lossless? | Ref |
|---|---|---|---|---|
| **Quest** | Block-level min/max key summaries; top-K block selection per query | Negligible at 128K with 2K budget | Yes (full KV retained) | arXiv:2406.10774 (2024) |
| **InfLLM** | Block-level CPU-GPU memory orchestration with representative vectors | Comparable to full attention at 1M tokens | Yes | arXiv:2402.04617 (NeurIPS 2024) |
| **InfiniGen** | Speculative prefetch: approximate next-layer query via SVD to pre-select KV from CPU | Up to 3x speedup over naive offload, negligible accuracy loss | Yes | OSDI 2024 (Lee et al.) |
| **ShadowKV** | Low-rank key cache on GPU + value cache offloaded to CPU; sparse reconstruction | Lossless on RULER/NIAH benchmarks | Effectively yes | arXiv:2410.21465 (ICML 2025 Spotlight) |
| **RetrievalAttention** | ANNS index on CPU for KV vectors; attention-aware search to handle query-key OOD | Lossless; lower latency than flat/IVF | Yes | arXiv:2409.10516 (NeurIPS 2024) |
| **MagicPIG** | LSH sampling on CPU; hash tables on CPU, sampling replaces top-K | Better accuracy than Quest on retrieval tasks | Approximate (with guarantees) | arXiv:2410.16179 (ICLR 2025 Spotlight) |
| **ArkVale** | Page-level eviction with recallable offload; bounding-volume scoring | >95% passkey retrieval across budgets | Yes (recall from offload) | NeurIPS 2024 (OpenReview 4oAt5L4lYe) |

These are the closest analogs to "LRU/LFU cache across tiers." The key difference from MoE expert caching: they must solve the *query-dependent selection* problem (which pages matter?) before the fetch, because attention is data-dependent. MoE routing is model-parameter-dependent and known before the expert load.

### (c) Paging / offload across tiers

| System | Mechanism | Quality cost | Lossless? | Ref |
|---|---|---|---|---|
| **vLLM PagedAttention** | OS-style virtual-memory paging for KV blocks; swap to CPU DRAM | None (exact attention) | Yes | arXiv:2309.06180 (SOSP 2023) |
| **FlexGen** | GPU/CPU/disk hierarchy with LP-optimized tensor placement; 4-bit KV compression | Latency penalty; throughput-oriented | Nearly lossless | arXiv:2303.06865 (ICML 2023) |
| **LMCache** | Content-addressable KV blocks across GPU/CPU/storage via vLLM connector | None | Yes | arXiv:2510.09665 (2025) |
| **Mooncake** | Disaggregated cluster-wide KV store (DRAM+NVMe, RDMA) | None | Yes | arXiv:2407.00079 (2024) |
| **CachedAttention** | Hierarchical KV storage (HBM/DRAM/disk) with scheduler-aware prefetch and positional-encoding decoupling | None | Yes | arXiv:2403.19708 (USENIX ATC 2024) |
| **HCache** | Saves/restores hidden states (not KV) to SSD; bubble-free scheduler | None | Yes | arXiv:2410.05004 (EuroSys 2025) |

These are multi-user serving systems. vLLM's swap, Mooncake, LMCache, and CachedAttention solve the *parking* problem (idle sessions' KV occupies VRAM) — not the *decode-time fetch* problem. They do not help a single active session whose KV exceeds VRAM during decode.

### (d) Compute attention where the KV lives

| System | Mechanism | Quality cost | Lossless? | Ref |
|---|---|---|---|---|
| **FastDecode** | Full KV + attention offloaded to distributed CPU nodes | None (exact) | Yes | arXiv:2403.11421 (2024) |
| **HeadInfer** | Head-wise KV offload to CPU; one head at a time on GPU | None (exact, lossless) | Yes | arXiv:2502.12574 (2025) |
| **NEO** | Partial offload with asymmetric GPU-CPU pipelining | None | Yes | arXiv:2411.01142 (MLSys 2025) |
| **llama.cpp `--no-kv-offload`** | Keep all KV in host RAM; run attention on CPU | Exact but slow — CPU FLOPS is the bottleneck | Yes | llama.cpp mainline |

HeadInfer is the most relevant to small-GPU setups: it streams one attention head at a time through GPU, keeping the rest in host RAM. On a 24 GB GPU it handles 4M tokens. The penalty is PCIe round-trips proportional to (num_heads x num_layers); on PCIe 3.0 with only 2 KV heads x 10 attention layers this is 20 transfers per token — feasible.

### (e) Shrink it instead

| System | Mechanism | Quality cost | Lossless? | Ref |
|---|---|---|---|---|
| **KIVI** | Asymmetric 2-bit quantization (per-channel keys, per-token values) | <0.1 ppl at 2-bit on LLaMA-2 | No (lossy quant) | arXiv:2402.02750 (ICML 2024) |
| **KVQuant** | Non-uniform quantization + dense-and-sparse; enables 10M context on 8xA100 | <0.1 ppl degradation at 3-bit | No | arXiv:2401.18079 (NeurIPS 2024) |
| **MLA (DeepSeek)** | Multi-head Latent Attention; projects KV into low-rank latent space at architecture level | None (trained with it) | Yes (by design) | DeepSeek-V2 (2024) |
| **Hybrid attention+recurrent** | Replace most attention layers with recurrent (Mamba, GDN, RWKV); fixed-size state | Architecture-dependent; Qwen3.6 competitive with pure-attention peers | Yes (by design) | Qwen3.6 (2025), Jamba, Zamba |
| **llama.cpp `-ctk`/`-ctv`** | Quantize KV cache to q4_0/q5_1/q8_0 at inference time | Model-dependent; q8_0 usually safe | No | llama.cpp mainline |

Your Qwen3.6-35B-A3B already uses approach (e) architecturally: 30/40 layers are recurrent (fixed 62.8 MiB state), leaving only ~20 KB/token for the 10 attention layers. Combining `-ctk q8_0 -ctv q8_0` halves that to ~10 KB/token; `-ctk q4_0 -ctv q4_0` quarters it to ~5 KB/token (test quality before committing).

## 3. What Exists TODAY in llama.cpp Mainline

| Feature | Flag / mechanism | What it does | Hybrid/GDN support | Limits |
|---|---|---|---|---|
| **Host prompt cache** | `--cache-ram <MiB>` (PR #16391, Oct 2025) | Saves idle slot KV+recurrent state to host RAM; restores on prefix match. NOT offloading — it is a prompt-reuse cache. | Yes (saves recurrent state via `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`) | Default 8192 MiB; on 16 GB RAM with model loaded, likely 2-4 GB usable. Causes RAM pressure at your spec. |
| **Slot save to disk** | `--slot-save-path <dir>` + `/slots` API | Manual save/restore of full slot state (KV + recurrent) to disk files. Survives restart. | Yes | Manual only (curl to save/restore). No automatic tiering. No incremental checkpoints. |
| **Context checkpoints** | `-ctxcp <N>` (default 32), `-cms <N>` (default 8192) | RAM-resident checkpoints of non-reconstructible state (SWA KV + recurrent state) for efficient context shifting. | Yes — specifically designed for hybrid models | RAM-only. Checkpoint count and step size are tunable. |
| **KV quantization** | `-ctk <type>` / `-ctv <type>` | Quantize K and V caches independently (q4_0, q5_1, q8_0, f16, etc.) | Yes for attention layers; recurrent state unaffected | Asymmetric configs (e.g., q5 keys / q4 values) sometimes fail; test per model. |
| **`--no-kv-offload`** | Flag | Keep KV cache in host RAM; compute attention on CPU | Yes | CPU becomes the bottleneck for attention layers. On i5-10400F (6C), expect 30-50% decode slowdown with 10 attention layers. |
| **SWA cache handling** | Automatic for SWA-capable models | Sliding-window layers use a ring buffer; only global-attention layers grow with context | Partial — Gemma4 uses SWA+global; Qwen3.6 attention layers are all full-attention | Reduces KV for SWA models but not for GDN hybrids where attention layers are full. |
| **Automatic KV tiering** | Does not exist | No `--cache-disk`, no automatic VRAM→RAM→NVMe tiering | N/A | Proposed in discussions; maintainer-declined. No PR exists. |

## 4. Ranked Recommendations for Your Box

### Recommendation 1: KV quantization (`-ctk q8_0 -ctv q8_0`, then try q5_1/q4_0)
**Why first:** Zero-code, zero-latency, no PCIe overhead. Halves KV from ~80 MiB to ~40 MiB at 4K context (q8), or to ~20 MiB (q4). Each 40 MiB freed = one more expert cache slot. On a hybrid model with only 10 attention layers and 2 KV heads, the quantization error surface is small.
**Measure:** Perplexity on a held-out sample at q8 vs f16 vs q4. Needle-in-haystack at your operating context lengths (4K-8K).
**Kill criterion:** >0.3 ppl regression or any NIAH failure at q8 → do not go lower.

### Recommendation 2: `--slot-save-path` for prompt-reuse across restarts
**Why second:** Your stated pain point is re-processing long prompts at 39 tok/s. With `--slot-save-path`, save the slot after a long system-prompt prefill; restore it on next server start. NVMe read at ~3.5 GB/s restores 80 MiB in 23 ms vs. re-prefilling 4K tokens at 39 tok/s = 103 seconds. 4500x speedup for warm start.
**Measure:** Wall-clock TTFT with cold prefill vs. slot restore. Verify output quality is identical (it should be — this is lossless).
**Kill criterion:** If your NVMe write endurance is a concern at the volume of saves, or if slot files break across llama.cpp updates (no format guarantee).

### Recommendation 3: HeadInfer-style head-wise offload (future / if context grows past 8K)
**Why third:** If you ever need 32K+ context, the KV at 20 KB/token = 640 MiB, which exceeds your free VRAM after model + expert cache. HeadInfer's approach (stream one head at a time through GPU, keep rest in RAM) is lossless and feasible with only 2 KV heads x 10 layers = 20 transfers/token. At 12 GB/s pinned PCIe and ~1 KB per head transfer, overhead is <0.1 ms/token.
**Measure:** Implement a prototype or wait for a llama.cpp PR. Measure decode tok/s vs. baseline at 16K/32K context.
**Kill criterion:** If PCIe 3.0 latency (not bandwidth) dominates due to small transfer sizes — measure round-trip latency first.

### What Does NOT Apply to Your Setup

- **Eviction methods (H2O, StreamingLLM, SnapKV, PyramidKV):** Your operating context is 4K-8K tokens, not 128K. At these lengths the full KV fits in VRAM with quantization. Eviction adds complexity and quality risk for no memory benefit.
- **Quest / InfLLM / RetrievalAttention / MagicPIG (sparse retrieval):** These solve the "128K+ context on GPU" problem. At 4K-8K with 20 KB/token KV, the full attention computation is fast; sparse approximation adds overhead without payoff.
- **vLLM PagedAttention / Mooncake / LMCache:** Multi-user serving infrastructure. You are single-user on llama-server.
- **`--cache-ram` (host prompt cache):** At 2-4 GB free RAM while serving, the default 8 GiB allocation will OOM or starve the active KV cache. Only viable if you free RAM (kill other processes) or add more. The slot-save-to-disk approach (Recommendation 2) gives the same prompt-reuse benefit without RAM pressure.
- **`--no-kv-offload`:** Moves ALL KV to CPU and runs attention there. On i5-10400F this is strictly worse than keeping KV on GPU with quantization — you trade 4 GB VRAM headroom for a 30-50% decode throughput hit. Only makes sense if VRAM is so tight the model itself barely fits.
