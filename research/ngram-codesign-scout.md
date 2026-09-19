# N-gram MoE Co-Design Scout Report

**Date:** 2026-09-19
**Hardware context:** GTX 1650 SUPER 4 GB (Turing, PCIe 3.0 x16), i5-10400F 6C/12T AVX2, 16 GB DDR4, NVMe. Models: Qwen3.6-35B-A3B IQ2_M (46 tok/s), Gemma4-26B-A4B (34-48 tok/s). GPU-resident expert cache: per-layer LRU slots, 57-65% hit rate. MTP n=2 drafts, acceptance 0.83-0.94.
**Training box:** 512 GB DDR5, 2x MI210 64 GB (gfx90a, ROCm), 2-4x R9700 32 GB (RDNA4, gfx1201). Mixed RCCL collectives do not work across gfx90a+gfx1201.

---

## TL;DR

1. **SuffixDecoding is the highest-value zero-training experiment.** 20 us/token draft, CPU-only, 1.8-5.3x on agentic workloads (NeurIPS 2025 Spotlight). Hybrid rule: use suffix tree when SCORE > tau, else fall back to MTP. Composes, doesn't compete.
2. **Draft-driven expert prefetch is the biggest cache win.** MoE-SpAc (arXiv:2603.09983) proves the concept: draft tokens as "lookahead sensor" for expert cache, 85% prediction accuracy, 25 tok/s on Qwen3-30B-A3B vs 15 tok/s llama.cpp+SD. Your MTP n=2 gives you 2-step Belady-like future knowledge for free. Expected cache hit: 57-65% -> 75-85%.
3. **ReMoE gate-only fine-tuning** with locality regularizer: 2-8 hours on MI210, pushes hits to 80-90%. Sticky Routing confirms: CHR 0.54->0.88 (3.9x fewer misses) at lambda=0.5, with PPL *improving* on medium models.
4. **Whittle is real.** logic65/Whittle-Qwen-3.8-35B-A3B: 3.3h on one 96 GB Blackwell, forward-KL distillation (top-128 logits) on 1,840 Qwen3.8-27B teacher traces. 10B n-gram memory transferred from Qwen3.8-Flash-Next. Runs on stock llama.cpp.
5. **Engram/n-gram embedding is production-validated** (DeepSeek Engram-27B, Qwen3.8-Flash-Next 51B n-gram params, Meituan LongCat-Flash-Lite 30B in embeddings). But requires pretraining or the Whittle-style memory-first warmup + KD path. Not a drop-in retrofit.
6. **The "rainbow table" = CREST-compacted continuation trie.** 200-800 MB on NVMe, mmap'd, <1 ms lookup. REST reports 0.2-0.7 ms for 0.9-27 GB stores. CREST achieves 10-13x compression at same or better acceptance.
7. **Full logit KD** on 2x MI210: realistically 6-39 days for 200M tokens (based on Dave's MI210 measured throughput + ZeRO-3 offload penalty). Whittle's 3.3h recipe needs Blackwell-class HBM, but its n-gram table transfer technique is reproducible on MI210 for the gate/table warmup phase.
8. **Build order:** ngram-mod (free) -> draft-driven expert prefetch (1-2d impl) -> LFRU eviction (2h) -> static ngram cache from self-distilled outputs (1h) -> ReMoE gate fine-tune on MI210 (2-8h) -> SuffixDecoding integration (1-2w) -> Whittle-style n-gram table transfer (weeks, needs MI210 box).

---

## 1. Draft Layer: Precomputed N-gram Continuation Stores

> **Scope note:** llama.cpp-specific flags (`--spec-type ngram-mod`, `--lookup-cache-static/dynamic`, the `--ngram-text` proposal, and related PRs) are covered by a separate agent. This section covers the literature, data structures, and composition with MTP.

### Papers & Systems

| System | Source | Date | Mechanism | Measured Numbers | Relevance |
|--------|--------|------|-----------|-----------------|-----------|
| **REST** | [arXiv:2311.08252](https://arxiv.org/abs/2311.08252) (NAACL 2024) | Nov 2023 | Suffix-array exact-match retrieval -> trie-organized continuations, max 64 draft tokens, n_max=16 | Code 27 GB store: 2.36x (HumanEval), 0.7 ms lookup; Chat 465 MB store: 1.62x, 0.1 ms lookup; M=1.96-2.65 depending on store size | Direct "rainbow table" analog; suffix-array + trie |
| **CREST** | [arXiv:2408.04678](https://arxiv.org/abs/2408.04678) | Aug 2024 | Compacted REST: store only frequent small n-grams; compressed token trees remain operational on disk | **10.6-13.5x less storage, 16.5-17.1% higher acceptance** vs REST at same storage | **Key for memory-tight setup.** Prunes rare long n-grams that hurt acceptance. |
| **DReSD** | [arXiv:2502.15572](https://arxiv.org/abs/2502.15572) | Feb 2025 | Dense retrieval replaces REST's exact match | 87% higher mean acceptance rate, 65% longer accepted drafts vs REST | Needs embedding model; overkill for CPU-bound setup |
| **SuffixDecoding** | [arXiv:2411.04975](https://arxiv.org/abs/2411.04975) (NeurIPS 2025 Spotlight) | Nov 2024 | Dual suffix trees (prompt + history); adaptive-depth tree speculation; frequency-scored greedy expansion | AgenticSQL: **5.3x, 6.3 mean accepted tokens/step**; SWE-Bench: **2.5x, 7.8 accepted/step**; Spec-Bench (open-ended): 1.66x standalone, **2.5x hybrid with EAGLE-3**; ~20 us/token draft, ~4 us update, ~12 us lookup | **Best agentic speedup. Zero GPU overhead. Ships in vLLM/ArcticInference.** |
| **SAM Decoding** | [ACL 2025](https://aclanthology.org/2025.acl-long.595/) | 2025 | Suffix automaton (DAWG) for draft generation | Improved over REST via automaton structure | Alternative DS; less tooling than suffix tree |
| **Infini-gram mini** | [arXiv:2506.12229](https://arxiv.org/abs/2506.12229) (EMNLP 2025) | Jun 2025 | FM-index for trillion-token n-gram search; 7% of suffix array size | O(1)-in-n search; BWT + Huffman wavelet tree | Designed for corpus-scale; overkill here |
| **Lookahead Decoding** | [ICML 2024](https://github.com/hao-ai-lab/LookaheadDecoding) | 2024 | Jacobi-iteration n-gram collection; 2D window (W, N) | Moderate speedup; GPU-bound | Needs GPU cycles you don't have |
| **Prompt Lookup / llama.cpp ngram-cache** | [llama.cpp common/ngram-cache.cpp](https://github.com/ggml-org/llama.cpp) | Ongoing | Hash-table n-gram match; static+dynamic+context caches | See separate llama.cpp agent report | **Already in your toolchain** |

### SuffixDecoding: Memory & Store Sizing (from Table 2)

| Entries | Tokens | Store Size | Lookup | Update |
|---------|--------|-----------|--------|--------|
| 1K | 27M | 1,670 MB | ~12 us | ~4 us |
| 5K | 143M | 2,980 MB | ~12 us | ~4 us |
| 10K | 285M | 4,070 MB | ~12 us | ~4 us |
| 20K | 572M | 6,150 MB | ~12 us | ~4 us |

Derived: **1.075 x 10^-8 GB/token** (~10.75 bytes/token in tree). For a 10M token self-distilled corpus: ~107 MB. For 100M tokens: ~1.07 GB.

### Hybrid Drafting: N-gram + MTP Composition

SuffixDecoding explicitly describes a hybrid rule with EAGLE that transfers to MTP:

- **Threshold tau:** If suffix tree SCORE > tau, use tree draft; else fall back to MTP/EAGLE
- **Agentic tasks (high repetition):** tau=0 is optimal (always use tree), achieves 5.35x
- **Mixed workloads:** tau in [5, 7] balances tree + model drafting, achieves 2.5x
- **Open-ended (low repetition):** Tree alone underperforms EAGLE; hybrid recovers

For your setup: tree draft has zero GPU cost (CPU suffix tree), while MTP uses GPU. They naturally compose -- tree proposes when patterns exist, MTP covers novel sequences. The verification pass accepts from the union via tree-causal masking. Your MTP acceptance of 0.83-0.94 sets the floor; the tree only adds tokens the model would have accepted anyway.

### Self-Distilled vs Corpus-Derived Store

No paper directly compares these. REST uses corpus-derived stores. The llama.cpp `--ngram-text` proposal ([Discussion #22262](https://github.com/ggml-org/llama.cpp/discussions/22262)) envisions prefilling from "outputs from the strongest LLMs." SuffixDecoding's online adaptation shows a store self-built from 500 requests is "almost indistinguishable" from one trained on the full domain corpus.

**Recommendation (inferred):** A self-distilled store (run the model on representative prompts, collect outputs, build suffix tree) maximizes acceptance because the token distribution matches the model's own prior. This is cheap: run inference, capture outputs to a token file, build the tree offline. Entropy predicts effectiveness: AgenticSQL extract entropy 0.086 (5.3x); code entropy 2.95 (1.5-2x); open chat entropy 3.43 (~1.2x).

---

## Rainbow Table Section: Concrete Data-Structure Recommendation

### What "n-gram rainbow table" actually maps to

| Data Structure | Build | Lookup | Size per M tokens | On-Disk? | Implementation |
|---------------|-------|--------|-------------------|----------|----------------|
| **Flat hash table** (llama.cpp ngram-cache) | O(n) single pass | O(1) hash, <1 us | ~20-50 MB | .bin file, load at startup | `common/ngram-cache.cpp`, mainline |
| **Hashed n-gram -> top-k continuation** (REST trie) | O(n log n) trie build from suffix array | 0.2-0.7 ms | ~100-1000 MB (depends on n_max, corpus) | Yes, mmap-able but CREST notes naive REST layout is hard to partially decompress | REST reference impl |
| **CREST compact trie** | O(n) with pruning | 0.2-0.7 ms (same REST interface) | **10-13x smaller than REST** at same acceptance | Yes; compressed token trees remain searchable on disk | CREST reference impl |
| **Suffix tree** (SuffixDecoding) | O(n) Ukkonen's | ~12 us lookup, ~4 us update | ~10.75 bytes/token | CPU memory; grows with history | [ArcticInference](https://github.com/snowflakedb/ArcticInference) |
| **FM-index** (Infini-gram mini) | O(n) BWT + wavelet | O(m) pattern | 7% of suffix array | Yes; designed for mmap/on-disk | Infini-gram codebase |
| **Suffix automaton** (SAM Decoding) | O(n) | O(m) | ~2n states | CPU memory | SAM Decoding code |

### For your 16 GB host, ~4 GB free while serving

```
Tier 1 (RAM, ~50-150 MB): Live n-gram cache
  - llama.cpp ngram-cache format (unordered_map<ngram, unordered_map<token, count>>)
  - Content: context n-grams (built live) + preloaded static cache from domain outputs
  - Lookup: O(1) hash, sub-microsecond

Tier 2 (NVMe mmap, ~200-600 MB): CREST-compacted continuation store
  - Content: top-k continuations for 2-16 gram patterns from self-distilled corpus
  - Size: CREST at 10-13x compression -> a 27 GB REST store compresses to ~2-2.5 GB
  - For your budget: a 3 GB raw corpus -> ~300 MB CREST store -> ~30 MB after 10x compression
  - Lookup: 0.2-0.7 ms via mmap (NVMe random read ~50-100 us for 4K page)
  - madvise(MADV_RANDOM) to prevent kernel readahead from evicting expert cache pages

Tier 3 (CPU memory, ~100-500 MB): SuffixDecoding tree for live session
  - Grows with conversation: ~10.75 bytes/token
  - A 50K-token conversation = ~538 KB
  - Even 10M tokens of history = ~107 MB -- well within budget
  - Lookup: 12 us (negligible vs 20-38 ms expert matvec)
```

**Recommended implementation order:**
1. Enable `--spec-type ngram-mod` (zero cost, already in llama.cpp)
2. Build a static ngram cache from self-distilled outputs, evaluate with `llama-lookup-stats`
3. If agentic/code workloads dominate, port SuffixDecoding's dual suffix tree (or wait for the ik_llama.cpp integration proposed in [Issue #1602](https://github.com/ikawrakow/ik_llama.cpp/issues/1602))

---

## 2. Model Layer: N-gram Lookup as a Sparsity Axis

### The Engram / N-gram Embedding Landscape

| System | Source | Date | Params in Embeddings | Architecture | Quality Delta | Retrofittable? |
|--------|--------|------|---------------------|-------------|--------------|----------------|
| **DeepSeek Engram-27B** | [arXiv:2601.07372](https://arxiv.org/abs/2601.07372), [GitHub](https://github.com/deepseek-ai/Engram) | Jan 2026 | 5.7B (reallocated 72->55 routed experts) | Hashed trigram, 8 heads, dim 1280, layers 2+15; O(1) lookup; scalar gate alpha_t via sigmoid | Matched pure MoE at rho~40%; Multi-Query NIAH 84.2->97.0; host offload penalty **<2.8%** (measured H800) | **No** (pretraining). But infinite-memory experiment attaches to *frozen* backbone. |
| **Engram-Nine** | [arXiv:2601.16531](https://arxiv.org/abs/2601.16531) | Jan 2026 | Same budget, MPHF hot tier | Collision-free for frequent n-grams | Does NOT consistently improve val loss under iso-params | N/A |
| **Tensorizing Engram** | [arXiv:2606.08347](https://arxiv.org/abs/2606.08347) | Jun 2026 | Shared latents via tensor decomposition | Addresses hash collision + order isolation | Improves over both Engram and Over-Tokenized Transformer | **No** |
| **Qwen3.8-Flash-Next** | [qwen.ai blog](https://qwen.ai/blog?id=qwen3.8-flash-next) | Aug 2026 | **51B** n-gram embedding (on top of 125B backbone) | Inspired by Engram + PLE; Gated DeltaNet + Qwen Sparse Attention; 4 hyper-connection residual branches | Production model; can offload n-gram to host async (Nvidia only currently) | **No** (it IS the architecture). But Whittle transfers its table. |
| **SCONE** | [arXiv:2502.01637](https://arxiv.org/abs/2502.01637) (NeurIPS 2025) | Feb 2025 | Up to billions of f-gram entries | BPE-inspired f-gram selection; separate f-gram model during training; precomputed and offloaded at inference | Outperforms 1.9B at half inference FLOPs | **Partially** -- needs training a separate f-gram model |
| **Gemma 3n/4 PLE** | [Google](https://ai.google.dev/gemma/docs/gemma-3n), [Raschka](https://sebastianraschka.com/llm-architecture-gallery/per-layer-embeddings/) | Jun 2025 | Gemma 4 E4B: 3.5B in PLE (8B total, 4.5B effective) | 256-d vector per layer per token; CPU-offloadable; caches to fast local storage | Carried forward to Gemma 4; key for on-device | **No** |
| **Over-Tokenized Transformer** | [arXiv:2501.16975](https://arxiv.org/abs/2501.16975) | Jan 2025 | Tiled matrix parameterization for V^n vocabulary | Hash n-gram embeddings + MTP | Complements BLT findings on byte-level | **No** |
| **Scaling Embeddings > Scaling Experts** | [arXiv:2601.21204](https://arxiv.org/abs/2601.21204) (Meituan) | Jan 2026 | LongCat-Flash-Lite: **30B in embeddings** (68.5B total, 2.9-4.5B active, 46% in embeddings) | N-gram Embedding with specialized N-gram Cache and synchronized kernels | PLNE failed to consistently beat NE when scaling; NE alone adopted. Outperforms parameter-equivalent MoE baseline. | **No** (from scratch) |
| **MoLE** (Mixture of Lookup Experts) | [arXiv:2503.15798](https://arxiv.org/abs/2503.15798) (ICML 2025 Oral) | Mar 2025 | Re-parameterize FFN experts as LUTs at inference | Train as MoE; inference as lookup table on storage | Matches MoE quality; dense-model decoding speed; zero VRAM for experts | **No** (MoLE-style training required) |
| **BLT** | [arXiv:2412.09871](https://arxiv.org/abs/2412.09871) | Dec 2024 | Hash n-gram embeddings at byte level (3-8 grams) | Byte-level; local encoder | Different architecture | **No** |

### The Whittle Path: Transferring N-gram Tables

**Whittle** (logic65/David Aylward) demonstrates that an Engram-style n-gram memory table CAN be retrofitted into an existing MoE via a specific recipe:

1. **Architecture:** Use `qwen4_exp` format (same as Qwen3.8-Flash-Next). Body from Qwen3.6-35B-A3B. N-gram memory: 8 hash heads x 4,880,000 rows x 256, bigram + trigram.
2. **Table transfer:** Read n-gram rows from Qwen3.8-Flash-Next's 320M-row table (16 heads x 20M x 160, fp8). Use a 172M-token counting corpus to select which rows to keep. Write into the student's larger hash geometry.
3. **Table-first warmup:** 30 min. Trains the memory to be load-bearing before touching the backbone.
4. **Joint KD:** Forward-KL distillation from Qwen3.8-27B (thinking on), top-128 logits per position. 1,840 complete teacher traces. 2,861 steps (tbl1) + 3,624 steps (lw2). PoSE: 2K-token rows at random offsets up to 131K. Layer steer: student layers 35/39 toward teacher 59/63.
5. **Dependence loss:** `relu(0.3 - (CE_off - CE_on))` -- ensures the model actually uses the memory (CE with memory on must be meaningfully lower than with it off).
6. **Wall-clock:** **3.3h on one 96 GB Blackwell.** Memory ~10.5 GB at Q8.

**Results:** Memory gain (held-out rows): code +2.84, general +0.67, science +0.41, chat +0.08. GSM8K 41/50. Long-context gate 30/36. Stop/loop battery 24/24. Maths probe 44/60 (down from base 48/60 -- quality cost of the distillation, not the memory).

**Implications for your setup:** The Whittle recipe proves n-gram tables are transferable. On 2x MI210 (128 GB HBM): the student (35B BF16 = ~70 GB) + n-gram table (~10 GB BF16) fits with room for optimizer states in CPU offload. The table warmup (30 min on Blackwell) would take ~2-4 hours on MI210 (inferred from ~3x throughput gap). The KD phase (3.3h on Blackwell) would take ~10-20h on MI210. Total: **~1-2 days for the Whittle recipe on your MI210 box** (inferred).

### Assessment

The n-gram embedding line is real, production-validated, and the most exciting architectural development in the space. But for your *current* inference stack, the benefit comes via:

1. **Layer 1 (draft):** The same n-gram statistics these papers exploit can drive your speculation engine via suffix trees or hash tables -- no model modification needed
2. **Layer 3 (cache):** Token-ID routing (see OpenMoE finding below) means n-gram identity predicts expert selection -- enabling n-gram-based expert prefetch
3. **Layer 4 (training):** The Whittle path is the cheapest way to add an n-gram memory to your existing model on the MI210 box

---

## 3. Cache Layer: Predicting Experts from Token N-grams

### The Token-ID Routing Foundation

**OpenMoE** ([arXiv:2402.01739](https://arxiv.org/abs/2402.01739)) established the critical finding: **MoE routing is driven by token identity, not semantics.** Most token IDs show "very strong specialization on only a few experts." Similar tokens cluster: "can/will/would" -> expert 31, "have/has/had" -> expert 30. Routing is fixed very early in training and does not change after SFT.

This means **token n-grams directly predict expert selections** -- a bigram or trigram of token IDs is a strong predictor of which experts will be needed, without running the gate function at all.

### Expert Prediction Systems

| System | Source | Date | Method | Accuracy | Hardware Context |
|--------|--------|------|--------|----------|-----------------|
| **Mixtral-Offloading** | [arXiv:2312.17238](https://arxiv.org/abs/2312.17238) | Dec 2023 | Cross-layer gate similarity (residual -> next gate) | ~90% top-1 next-layer | LRU k=2-4/layer; your baseline |
| **MoE-Infinity** | [arXiv:2401.14361](https://arxiv.org/abs/2401.14361) | Jan 2024 | Request-level EAM, cosine similarity | Request-level; degrades on domain shift | LFU |
| **EdgeMoE** | Referenced in surveys | 2024 | Calibration-data table: expert(layer n-1) -> expert(layer n) | High on calibration domain | **Closest to "n-gram over expert activations"** |
| **Pre-gated MoE** | [Microsoft ISCA 2024](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf) | May 2024 | Architectural: pre-gate at current layer determines next-layer | High (by construction) | Requires retraining |
| **ProMoE** | [arXiv:2410.22134](https://arxiv.org/abs/2410.22134) | Oct 2024 | 2-layer MLP on layer inputs | Higher than heuristic | Sliding window prefetch |
| **HOBBIT** | [arXiv:2411.01433](https://arxiv.org/abs/2411.01433) | Nov 2024 | Cross-layer gate + mixed precision | ~90% | LRU staging |
| **Fate** | [arXiv:2502.12224](https://arxiv.org/abs/2502.12224) | Feb 2025 | Cross-layer gate prediction; experts above 75th percentile confidence prefetched | **97.15% cross-layer prefetch accuracy**; 78.79% from gate input alone; per-layer: shallow worse than deep; caching layers 0-3 -> **99.08% hit rate** | 30% avg speedup from prefetch alone |
| **SP-MoE** | [arXiv:2510.10302](https://arxiv.org/abs/2510.10302) | Oct 2025 | Draft model attention outputs -> target gate; speculative expert loading | Top-1: **88-89% accuracy** (all models); cosine similarity 56-95% depending on model | 1.07-3.5x TPOT speedup |
| **CommitMoE** | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/39454) | 2025 | Commit Router: certainty-correlated prediction, no fallback | Router certainty correlates with prediction accuracy; low certainty -> output resilient | Novel: skips experts when uncertain |
| **Nov 2025 MLSys** | Referenced in Fate discussion | Nov 2025 | Pre-attention expert routers | **93.03% DeepSeek V2 Lite, 94.69% Qwen3-30B, 97.62% Phi-mini-MoE**; +15% over Fate | Best standalone accuracy |
| **MoE-SpAc** | [arXiv:2603.09983](https://arxiv.org/abs/2603.09983) | Mar 2026 | **Draft tokens as "lookahead sensor" for expert cache**; verifying multiple drafts produces expert-activation frequency map (vs binary AR signal) | **85% prediction accuracy** on MMLU-Pro layer 47; Shannon entropy of SD signal dominates AR signal by sqrt(gamma+1) | **25.06 TPS** Qwen3-30B on RTX 4090 (vs 15.23 llama.cpp, 17.66 llama.cpp+SD) |
| **Speculating Experts** | [arXiv:2603.19289](https://arxiv.org/abs/2603.19289) | Mar 2026 | Learned estimator (4-45M params): KL-distill to router logits from quasi-hidden states | Qwen3-30B: **~90% avg hit rate** after 4M training tokens; early layers need estimator (+25% over fixed router) | 5-14% TPOT reduction; **accuracy-sensitive** -- Router-PF drops Qwen3 GSM8k from 0.950 to 0.576 |
| **SpecPrefetch** | [arXiv:2607.24787](https://arxiv.org/abs/2607.24787) | Jun 2026 | Shared lightweight low-rank adapter predicts next-layer experts; transfer-only (native router still decides) | Improves over MLP/draft-model/ProMoE with fewer params | **Prediction errors affect only transfer efficiency, not model output** |
| **ReMoE** | [arXiv:2605.27081](https://arxiv.org/abs/2605.27081) | May 2026 | Gate-only fine-tuning: temporal-locality regularizers + KL trust-region to frozen snapshot; **disables standard load-balance loss** | Extends reuse streaks, stabilizes working set | **Key paper for training routers for locality** |
| **Sticky Routing** | [arXiv:2607.08780](https://arxiv.org/abs/2607.08780) | Jul 2026 | L2 penalty on consecutive gate distributions: L_cons = mean ||g_t - g_{t-1}||^2 | SR 0.71->0.29 at lambda=0.5; **CHR 0.54->0.88** (3.92x fewer misses, LRU C=2); **PPL improved** on medium model (-0.9% to -4.1%) | **Training-time regularizer; quality-neutral or quality-positive.** But only tested on 8.8-22M models. |

### Cache Policies That Beat LRU

| Policy | Source | Mechanism | vs LRU | Notes |
|--------|--------|-----------|--------|-------|
| **LFU** | MoE-Infinity; [arXiv:2511.05814](https://arxiv.org/abs/2511.05814) | Frequency-based | Marginal; similar to LRU in SeqMoE analysis | Robust to sequential-replay inflation (only +3.4% vs LRU's +27.4%) |
| **LFRU** | [vLLM RFC #38256](https://github.com/vllm-project/vllm/issues/38256) | score = freq / (clock - last_access + 1) | Prevents hub-expert thrashing during domain shifts | **Easy to implement; high practical value** |
| **Belady-8** (8-step oracle) | [SeqMoE arXiv:2609.12978](https://arxiv.org/abs/2609.12978) | 8-step future knowledge | Close to full Belady; much better than LRU | **Your MTP n=2 gives you 2-step future knowledge for free** |
| **SeqMoE probabilistic Belady** | SeqMoE | Probabilistic future-aware from sequence model | **96.97% hit rate at 45% residency, 80.22% of full-load perf** | State-of-art |
| **Reproducibility warning** | [arXiv:2608.07911](https://arxiv.org/abs/2608.07911) | Sequential replay inflates LRU by 27.4%, LFRU by 28.5%, but LFU only 3.4% | **Inverts rankings** | Evaluate with realistic interleaved workloads |

### Routing Consistency Analysis

The routing consistency study ([arXiv:2505.16056](https://arxiv.org/abs/2505.16056)) analyzed 20 MoE LLMs:

**High local routing consistency (SRP > 50, good for caching):** LLaMA-MoE-v2 (78.16), Yuan2.0 (63.48), PowerMoE (55.17), **Qwen3 (54.14)**, Phi-3.5-MoE (51.98), OLMoE (50.91), GRIN-MoE (50.39), Mixtral-8x7B (49.36)

**Low consistency (bad for caching):** Qwen1.5-MoE (30.71), DeepSeekMoE (36.94), DeepSeek-V2-Lite (37.92), XVERSE-MoE (38.58), SwitchTransformers (19.27-19.33)

**Critical findings:**
- **Shared experts REDUCE consistency** (bypass effect + decreased combination space). Your Qwen3.6-35B-A3B has shared experts, which is why your 57-65% hit rate is not higher.
- **Domain-specialized experts** contribute more to consistency than vocabulary-specialized ones
- **Cache size 2x active params** achieves best segment caching on most models
- **Global load balance can coexist with local consistency** -- the tradeoff is only with *local* load balance

### Draft-Driven Expert Prefetch: The Key Insight for Your Setup

**MoE-SpAc** proves this works. The insight: verifying K draft tokens produces an expert-activation **frequency map** (which expert was requested how many times across K drafts), vs the binary signal from single-token AR decoding. The frequency map's Shannon entropy dominates the AR signal by a factor of sqrt(K+1).

**For your setup (MTP n=2):**
1. Your draft tokens pass through the router to get acceptance/rejection anyway
2. Before the verification check, extract top-k expert selections from draft tokens' router outputs
3. Use those to prefetch experts into GPU cache slots while verification runs
4. This gives you **2-step future knowledge for free**
5. The MoE-SpAc paper shows this at K=4 with 17% cache ratio achieves 85% prediction accuracy and **25 TPS** (vs 15 TPS llama.cpp baseline)

**Caution from Speculating Experts ([arXiv:2603.19289](https://arxiv.org/abs/2603.19289)):** Naive router-based prefetch can **destroy accuracy** on some models. Qwen3-30B-A3B drops GSM8k from 0.950 to 0.576 with Router-PF due to "high representational drift in early layers." Their fix: a learned estimator (17M params for Qwen3-30B) trained on 4M tokens, or a hybrid approach that recovers ~37% of the gap. **However:** this accuracy issue only arises if you *commit* to the prefetched experts (skipping the real router). If you use prefetch purely for cache warming (which is your setup -- you always run the real router, just prefetch likely experts), prediction errors only waste bandwidth, not quality.

### Training Routers for Locality

Two approaches, both applicable:

**ReMoE** ([arXiv:2605.27081](https://arxiv.org/abs/2605.27081)):
- Gate-only fine-tuning with combined loss: temporal locality regularizer + KL to frozen snapshot
- **Disables standard load-balance loss** during fine-tuning (it explicitly encourages dispersion)
- Result: longer reuse streaks, more stable working set
- Cost on MI210: gate is a single linear layer per MoE block -> **hours, not days**

**Sticky Routing** ([arXiv:2607.08780](https://arxiv.org/abs/2607.08780)):
- L2 penalty on consecutive gate distributions: L_cons = (1/(T-1)) sum ||g_t - g_{t-1}||^2
- Total: L = L_CE + lambda * L_cons + mu * L_bal (mu=0.01)
- At lambda=0.5 (medium model): switch rate 0.71->0.29, CHR 0.54->0.88 (**3.92x fewer misses**), PPL *improved* by 0.9%
- **But:** only tested on 8.8-22M parameter models. The ReMoE comparison in the same paper shows ReMoE produces almost no SR reduction at this scale (SR 0.7054 vs 0.7087 baseline). This may be because ReMoE was designed for larger models.
- **Kill criterion:** If quality degrades on 35B-A3B at lambda > 0.01, reduce lambda or switch to ReMoE.

---

## 4. Training Layer: KD Recipe and Wall-Clock ETA

### Published KD Recipes

| Method | Source | Approach | Key Number | Notes |
|--------|--------|----------|-----------|-------|
| **GKD** | [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) | On-policy sampling + generalized JS divergence | Framework paper | Not specifically tested on MoE students ([arXiv:2502.12947](https://arxiv.org/abs/2502.12947)) |
| **MiniLLM** | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) | Reverse KL to avoid mode-covering | Framework | All MoE experiments use dense models |
| **DistiLLM-2** | ICML 2025 | Contrastive approach | Better than DistiLLM-1 on reasoning | Framework |
| **Thinking Machines on-policy KD** | [thinkingmachines.ai](https://thinkingmachines.ai/blog/on-policy-distillation/) | On-policy rollouts, dense per-token reverse KL | **1,800 GPU-hours** (vs 17,920 for RL), 9-30x compute savings | [Tinker SDK](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/) |
| **NVIDIA Minitron** | [arXiv:2407.14679](https://arxiv.org/abs/2407.14679) | Prune + distill | **40x fewer tokens** than from-scratch | Minitron-SSM: 400B tokens for 8B |
| **Arcee DistillKit** | [GitHub](https://github.com/arcee-ai/DistillKit) | Online/offline logit KD; polynomial logit compression; composable losses | Open-source toolkit | Supports sparse top-k logits, cross-tokenizer KD |
| **Empero Qwen3.8-35B-A3B-Distill** | [HuggingFace](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill) | SFT on curated Qwen3.8 teacher traces (off-policy) | Apache-2.0; already exists | Teachers: Qwen3.8 2.4T-A95B + Flash Next |
| **Whittle-Qwen-3.8-35B-A3B** | [HuggingFace](https://huggingface.co/logic65/Whittle-Qwen-3.8-35B-A3B) | Forward-KL (top-128 logits) + n-gram memory transfer + dependence loss | **3.3h on one 96 GB Blackwell**; 1,840 teacher traces, 6,485 total steps | **Verified.** The 96 GB is a B200/B300 (Blackwell). |
| **FastMTP (self-distillation)** | [arXiv:2509.18362](https://arxiv.org/abs/2509.18362) | Self-distilled MTP head fine-tuning; frozen backbone, 210.8M params (<3% of 7B backbone) | **<1 day on single H20**; 389.4K self-distilled samples, 3 epochs | Acceptance: k=1 70->81%, k=2 11->56%, k=3 2->36%. Speedup: **2.03x** mean. |
| **MTP-D** | [arXiv:2603.23911](https://arxiv.org/abs/2603.23911) | Self-distillation for MTP heads; gradient-detached KL; looped extension | +7.2% acceptance; +220.4% with looped extension | Minimal additional cost |

### Whittle Recipe Breakdown (Verified)

| Phase | Steps | Duration (Blackwell) | Content |
|-------|-------|---------------------|---------|
| Table warmup | - | 30 min | N-gram memory rows from Qwen3.8-Flash-Next's 320M-row table |
| tbl1 (table + body KD) | 2,861 | ~1.5h (inferred) | Forward-KL on 1,840 Qwen3.8-27B traces (top-128/position); layer steer L35<-L59, L39<-L63 weight 0.15 |
| lw2 (root, continued) | 3,624 | ~1.8h (inferred) | Same loss; layer steer weight 0.2 on half the steps |
| **Total** | 6,485 | **3.3h** | PoSE: 2K-token rows at random offsets up to 131K |

**Dependence loss:** `relu(0.3 - (CE_off - CE_on))` -- forces the model to actually use the memory table. Memory gain on held-out rows: code +2.84 nats, general +0.67, science +0.41, chat +0.08.

### ETA Table: 2x MI210 + 512 GB DDR5

#### Throughput Basis

**MI210 specs:** CDNA2, 64 GB HBM2e each, 1.6 TB/s memory bandwidth, 300W TDP, PCIe Gen4. BF16 MFMA supported. No FP8 decoder/ALU, no int4 MFMA.

**Measured references (Dave's MI210 stack):**
- Qwen3.8-27B dense decode (w8a8 + AITER): 42.3 tok/s per MI210 pair
- Qwen3.8-27B dense decode (w8a8 + AITER + MTP n2): 80.0 tok/s
- MoE w8a8 + MTP n3 + AITER: 139.8 tok/s decode (inference, not training)
- CK int8 GEMM: 2.9-3.5x decode end-to-end

**Training throughput estimate:**
- Databricks MPT-7B on MI250 FSDP: 166 TFLOPS/GPU. MI250 = 2x MI210 GCDs. Per-MI210: ~83 TFLOPS.
- MosaicML MPT-7B on 8x A100-40GB: 29,436 tok/s = 3,679 tok/s/GPU at 50.92% MFU.
- MI210 BF16 peak: ~45.3 TFLOPS. At 50% MFU: ~22.7 TFLOPS sustained.
- ZeRO-3 CPU offload penalty: ~3x throughput drop (measured: 8,151->2,615 tok/s on A100).

**For 35B-A3B MoE student on 2x MI210:**
- Model BF16: ~70 GB. Fits in 128 GB HBM.
- Optimizer states (Adam BF16 master + FP32 moments): ~280 GB -> ZeRO-3 CPU offload to 512 GB DDR5.
- Active params per forward/backward: ~3B -> ~18 GFLOPs/token.
- 2x MI210 at 50% MFU with offload: ~45 TFLOPS effective -> ~45 TFLOPS / 18 GFLOPs = ~2,500 tok/s.
- **With MoE routing overhead + grouped GEMM on gfx90a + all-to-all: apply 0.3-0.5x penalty** -> **750-1,250 tok/s**. (Inferred)
- **Pessimistic (grouped GEMM falls back to per-expert loop, 60-70% wall time per unsloth #2582):** ~200-400 tok/s. (Inferred)

| Token Budget | Wall-Clock (optimistic, 1000 tok/s) | Wall-Clock (pessimistic, 300 tok/s) | Notes |
|-------------|--------------------------------------|---------------------------------------|-------|
| **10M** (gate-only fine-tune) | **2.8 hours** | **9.3 hours** | ReMoE/Sticky Routing: gate params only |
| **50M** (MTP head + warmup) | **14 hours** | **46 hours** | FastMTP recipe; freeze backbone |
| **200M** (meaningful logit KD) | **2.3 days** | **7.7 days** | Whittle-equivalent token budget on MI210 |
| **500M** (substantial KD) | **5.8 days** | **19.3 days** | Likely diminishing returns past here |
| **1B** | **11.6 days** | **38.6 days** | Upper bound of useful KD |
| **5B** | **57.9 days** | **193 days** | Not feasible on this hardware |

#### Teacher Logit Generation

| Strategy | Hardware | Throughput | Time for 200M tokens | Storage |
|----------|----------|-----------|---------------------|---------|
| Qwen3.8-27B Q8 on 1x R9700 (32 GB) | RDNA4 | ~15-25 tok/s (inferred) | 93-154 days | Top-128 logits: 200M * 128 * 8 bytes = ~195 GB |
| Qwen3.8-27B Q4 on 1x R9700 | RDNA4 | ~30-50 tok/s (inferred) | 46-77 days | Same |
| 4x R9700 parallel | RDNA4 | ~120-200 tok/s | **12-19 days** | Same |
| Qwen3.8-27B BF16 on 2x MI210 (offline, non-training) | gfx90a | ~40-80 tok/s (from Dave's measured 42 tok/s w8a8 decode) | **29-58 days** | Same |
| **Whittle approach: only 1,840 traces** | Any | Minutes to generate | **<1 day** | ~1-2 GB | Key insight: you don't need millions of examples |

**Whittle's key insight:** You don't need a massive token budget. 1,840 complete thinking traces with per-position top-128 logits, quality-filtered, is enough for meaningful KD when combined with the n-gram memory table. The budget efficiency comes from the table providing the knowledge storage and the KD providing the routing/attention alignment.

#### Additional Loss Terms Cost

| Term | Overhead per Step | Total Cost on MI210 |
|------|------------------|-------------------|
| **MTP acceptance (MTP-D style)** | ~5-10% (one extra MTP head forward) | Negligible; MTP head is 210M params vs 35B backbone |
| **Expert locality (ReMoE style)** | ~1-2% (one extra matmul per MoE layer for KL term) | Can be done standalone: gate-only fine-tune in 2-8 hours |
| **Dependence loss (Whittle style)** | ~100% (requires forward pass with memory ON and OFF) | Doubles training time; only needed during memory-transfer phase |
| **Sticky Routing L_cons** | ~1% (L2 on consecutive gate distributions) | Negligible |

#### ROCm Blockers on gfx90a

| Component | Status | Blocker? |
|-----------|--------|----------|
| **Flash Attention** | Works via CK port (head dim <= 128). ASM FA: 80/80 configs exact. ([Dave's MI210 stack](https://github.com/davetha/mi210-llm-stack)) | No |
| **BF16 MFMA** | Supported; unified VGPR/AGPR file. | No |
| **INT8 GEMM** | Bit-exact; 4.3x faster than BF16 at M=16,N=K=8192 for decode. Prefill: wash (CDNA2 gives INT8/BF16 same 181 TOPS). | No |
| **FP8** | **Not supported on gfx90a/CDNA2.** Hardware limitation. FP8 kernels cause build failures ([ROCm/aiter#179](https://github.com/ROCm/aiter/issues/179)). | **Yes for FP8 specifically** |
| **Grouped GEMM (`torch._grouped_mm`)** | Missing on ROCm as of Aug 2025 ([pytorch#161366](https://github.com/pytorch/pytorch/issues/161366)). Status: "Done" but no PRs found. Unsloth MoE on H800: 10-20% GPU utilization, 200-300s/step ([unsloth#2582](https://github.com/unslothai/unsloth/issues/2582)). | **Partial. Per-expert loop fallback is 60-70% of wall time.** |
| **DeepSpeed ZeRO-3 + offload** | Works via HIP. Architecture-independent. Not in hot CI path. | **Use AMD curated Docker images** |
| **RCCL gfx90a+gfx1201** | Does not work. Train on MI210 pair only. | Known; R9700s for offline teacher inference only |
| **CI regression risk** | gfx90a off hot CI path; regressions ship unnoticed. 1,180 blocked ASM kernels (no int4 MFMA, no gfx942 K=32 INT8 encoding). 242 of 1,422 gfx942 kernels are portable. | **Pin to validated Docker image** |

---

## Ranked Experiments (Cheapest First)

| # | Experiment | Expected Gain | Cost | Kill Criterion |
|---|-----------|--------------|------|----------------|
| 1 | **Enable `--spec-type ngram-mod`** | 1.1-1.5x on repetitive tasks (composes with MTP) | Zero; already in mainline | Acceptance < 5% on your workloads (use `llama-lookup-stats`) |
| 2 | **Build static ngram cache** from self-distilled outputs | +10-30% acceptance on in-domain; entropy predicts: code ~2x, chat ~1.2x | ~1 hour to build corpus + cache | `llama-lookup-stats` shows < 5% improvement |
| 3 | **Draft-driven expert prefetch** from MTP draft tokens | Cache hit 57-65% -> 75-85% (inferred from MoE-SpAc at K=4: 85% accuracy) | 1-2 days implementation | Hit rate improvement < 5% on measured workloads |
| 4 | **Switch LRU to LFRU** | Prevents hub-expert thrashing on domain shifts | ~2 hours implementation | A/B test shows no improvement on mixed workloads |
| 5 | **ReMoE gate-only fine-tune** on MI210 box | Cache hit -> 80-90% (Sticky Routing: CHR 0.54->0.88 at lambda=0.5) | 2-8 hours training, ~10M tokens | Quality regression on benchmarks; or if grouped GEMM fallback makes training impractical |
| 6 | **FastMTP self-distillation** for better acceptance | MTP k=2 acceptance +7-45% (FastMTP: k=2 from 11%->56%); overall 2.03x speedup | 1-4 days on MI210; freeze backbone, train 210M-param head only | Acceptance improvement < 2% |
| 7 | **SuffixDecoding integration** | 1.5-5.3x on agentic/code (entropy-dependent); hybrid with MTP at tau=5-7 | 1-2 weeks implementation (port from ArcticInference or build in llama.cpp) | Measured speedup < 1.2x on your actual workloads |
| 8 | **Whittle-style n-gram table transfer + KD** | Adds 10B load-bearing n-gram memory; code +2.84 nats; long-context 18/36->30/36 | ~1-2 days on MI210 (table warmup + 6,485 KD steps) | Memory dependence gap (CE_off - CE_on) < 0.1 |
| 9 | **Full logit KD** (Qwen3.8-27B -> student, 200M+ tokens) | Quality uplift across all tasks; cumulative with Whittle | 2-8 days student + 12-19 days teacher logit gen (4x R9700) | Eval scores plateau before 200M tokens |

---

## Could Not Verify

| Claim | Status | Notes |
|-------|--------|-------|
| ~~Whittle 3.3h/96GB claim~~ | **Verified** | logic65/Whittle-Qwen-3.8-35B-A3B HF model card: "3.3 h on one 96 GB Blackwell", forward-KL, 1,840 traces |
| MI210 training tok/s for MoE with ZeRO-3 | **No direct measurement** | Estimated from MI250 FSDP data + inference throughput from Dave's MI210 stack |
| R9700 (gfx1201) inference throughput for 27B model | **No benchmarks found** | R9700 too new; RDNA4 ROCm inference exists but no published large-model tok/s |
| ARC cache policy in MoE context | **No papers found** | Not used in any MoE caching paper; LFRU is the practical alternative |
| SuffixDecoding memory per token (10.75 bytes) | **Derived** | Calculated from Table 2 (1K entries = 27M tokens = 1,670 MB -> ~61.8 bytes/token). Note: this is tree overhead, not raw token storage. |
| Sticky Routing at 35B scale | **Untested** | Only validated on 8.8-22M models; ReMoE comparison showed near-zero effect at that scale |
| CREST mmap serving latency | **Partial** | CREST discusses on-disk layout but doesn't publish mmap-specific latency numbers |
| Grouped GEMM on gfx90a for MoE training | **Partial** | `torch._grouped_mm` issue marked "Done" but no PRs found; unsloth MoE shows 10-20% GPU utilization (likely falling back to per-expert loop) |
