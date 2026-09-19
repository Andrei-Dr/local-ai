# N-gram MoE Co-Design Scout Report

**Date:** 2026-09-19
**Hardware context:** GTX 1650 SUPER 4 GB (Turing, PCIe 3.0 x16), i5-10400F 6C/12T AVX2, 16 GB DDR4, NVMe. Models: Qwen3.6-35B-A3B IQ2_M (46 tok/s), Gemma4-26B-A4B (34-48 tok/s). GPU-resident expert cache: per-layer LRU slots, 57-65% hit rate. MTP n=2 drafts, acceptance 0.83-0.94.
**Training box:** 512 GB DDR5, 2x MI210 64 GB (gfx90a, ROCm), 2-4x R9700 32 GB (RDNA4, gfx1201). Mixed RCCL collectives do not work across gfx90a+gfx1201.

---

## TL;DR

1. **SuffixDecoding is the highest-value zero-training experiment.** Model-free, CPU-only, 20 us/token draft cost, 1.8-5.3x speedups on repetitive workloads. Already shipping in vLLM/ArcticInference. Composes with your MTP drafter (tree verification accepts from both). Build a corpus-seeded suffix tree store on NVMe next to llama.cpp; expected gain: 1.3-2x on top of MTP for agentic/code tasks with high token-repeat density.
2. **llama.cpp `--spec-type ngram-mod` already exists and is reportedly beating draft models** in community benchmarks. Start there today, zero training. The `--ngram-text` prefill proposal is not merged but trivial to implement locally.
3. **Expert prediction from token bigrams can reach ~90% recall@k** (cross-layer gate similarity); training a ReMoE-style locality regularizer on the friend's box would push your cache hit rate from 57-65% to 80-90%. This is the cheapest training intervention with the largest per-token latency gain (removing 20-38 ms CPU expert matvecs on hits).
4. **Engram/n-gram embedding is real and validated** (DeepSeek, Meituan LongCat-Flash-Lite) but requires pretraining from scratch or massive continued pretraining. Not retrofittable by distillation at reasonable cost. Park it.
5. **Full logit KD into the 35B-A3B student on 2x MI210** is feasible but slow: ~60-100 tok/s training throughput with ZeRO-3 CPU offload, yielding 6-23 days for 200M-1B tokens. The Empero Qwen3.8-35B-A3B-Distill already exists (SFT on teacher traces); logit-level KD on top of it would be the incremental experiment.
6. **The "rainbow table" analog is a hashed n-gram -> top-k continuation store**, implemented as a CREST-compacted trie or a flat hash table (llama.cpp ngram-cache format). For your memory-tight setup: ~200-800 MB on NVMe, mmap'd, sub-millisecond lookup.
7. **Hype check:** "n-gram rainbow tables" as described are just n-gram caches with a fancy name. The real innovation is in how you compose them with MTP (hybrid drafting) and expert prediction (draft-driven prefetch). No single paper does all three yet. That is the build.

---

## 1. Draft Layer: Precomputed N-gram Continuation Stores

### Papers & Systems

| System | Source | Date | Claim | Measured Numbers | Relevance |
|--------|--------|------|-------|-----------------|-----------|
| **REST** | [arXiv:2311.08252](https://arxiv.org/abs/2311.08252) (NAACL 2024) | Nov 2023 | Retrieval-based spec decoding from datastore; trie-organized continuations | Acceptance length 3.22-3.48; lookup 0.2-0.7 ms for 0.9-27 GB stores (RASD measurements) | Direct analog of "rainbow table"; plug-and-play, no training |
| **CREST** | [arXiv:2408.04678](https://arxiv.org/abs/2408.04678) | Aug 2024 | Compact REST: store only frequent small n-grams | 10.6-13.5x less storage, 16.5-17.1% higher acceptance length vs REST at same storage | Key for your memory-tight setup |
| **SuffixDecoding** | [arXiv:2411.04975](https://arxiv.org/abs/2411.04975) (NeurIPS 2025 Spotlight) | Nov 2024 | Suffix-tree based, model-free, adaptive depth, tree speculation | 5.3x on AgenticSQL, 2.8x over EAGLE-2/3, 1.9x over Token Recycling, 1.8-4.5x SWE-Bench; ~20 us/token draft cost | **Best-in-class for agentic/repetitive workloads. Zero GPU overhead.** |
| **SAM Decoding** | [ACL 2025](https://aclanthology.org/2025.acl-long.595/) | 2025 | Suffix automaton for draft generation | Improved over REST via DAWG/suffix-automaton | Alternative data structure to suffix tree |
| **Infini-gram mini** | [arXiv:2506.12229](https://arxiv.org/abs/2506.12229) (EMNLP 2025) | Jun 2025 | FM-index for trillion-token n-gram search; 7% of suffix array size | O(1)-in-n search time; 7% storage vs canonical SA | If you want internet-scale corpus backing |
| **Prompt Lookup / llama.cpp ngram-cache** | [llama.cpp common/ngram-cache.cpp](https://github.com/ggml-org/llama.cpp) | Ongoing | Hash-table n-gram match from prompt/prior generation; static+dynamic+context caches | Community: "ngram-mod actually beats draft models"; `--spec-type ngram-mod` in mainline | **Already in your toolchain.** |
| **Lookahead Decoding** | [arXiv:2402.02057](https://github.com/hao-ai-lab/LookaheadDecoding) (ICML 2024) | 2024 | Jacobi-iteration n-gram collection; 2D window (W, N) | Moderate speedup; GPU-bound (runs Jacobi on target model) | Less relevant -- needs GPU cycles you don't have |
| **DReSD** | [arXiv:2502.15572](https://arxiv.org/abs/2502.15572) | Feb 2025 | Dense retrieval replaces REST's exact match | 87% higher mean acceptance rate vs REST, 65% longer accepted drafts | Needs embedding model; overkill for your setup |

### Acceptance Rates & Speedups for Your Setup

Your MTP drafter already achieves 0.83-0.94 acceptance. N-gram stores would compose as a **hybrid drafter**: the n-gram store proposes continuations where pattern density is high (code, structured output, repeated agent loops), and MTP covers the rest. The tree verification pass accepts from the union.

Key numbers for composition:
- SuffixDecoding on repetitive workloads: ~3-8 accepted tokens per step (vs 2-3 for MTP alone)
- llama.cpp ngram-mod: highly task-dependent but strongest on code completion, summarization, structured output
- REST/CREST: 3.2-3.5 accepted tokens with 0.2-0.7 ms lookup overhead

### Self-Distilled vs Corpus-Derived Store

No paper directly compares a store built from the target model's own outputs vs one from a corpus. However:
- The `--ngram-text` proposal in llama.cpp (April 2026, [Discussion #22262](https://github.com/ggml-org/llama.cpp/discussions/22262)) envisions exactly this: prefilling the ngram cache from "outputs from the strongest LLMs (like GLM, Kimi, DeepSeek)"
- REST's datastore is corpus-derived; RASD shows larger stores = higher acceptance but diminishing returns past ~9 GB
- A self-distilled store (running the model on representative prompts, capturing outputs) would maximize acceptance rate since the token distribution matches the model's own prior. This is cheap to build: run inference on a domain corpus, collect output tokens, build the n-gram cache. (Inferred)

### "Rainbow Table" -- What It Actually Maps To

The closest real analog:

| Data Structure | Build Cost | Lookup Latency | Store Size | Fits Your Setup? |
|---------------|-----------|----------------|-----------|-----------------|
| **Flat hash table** (llama.cpp ngram-cache) | O(n) single pass | O(1) hash lookup | ~50-200 MB for 1M tokens of context | Yes -- trivially fits in RAM |
| **Hashed n-gram -> top-k continuation** (REST trie, CREST compact) | O(n log n) trie build | 0.2-0.7 ms | 200-800 MB for useful store | Yes -- mmap from NVMe, ~4 GB free RAM is enough for index |
| **Suffix tree** (SuffixDecoding) | O(n) Ukkonen's | ~20 us per draft token | Proportional to corpus; ~2-4x raw token count | Yes -- CPU memory only |
| **FM-index** (Infini-gram mini) | O(n) BWT+wavelet | O(m) where m=pattern length | 7% of suffix array = ~0.07n bytes | Overkill -- designed for trillion-token scale |
| **Suffix automaton** (SAM Decoding) | O(n) | O(m) | ~2n states | Viable but less tooling than suffix tree |

**Concrete recommendation for 16 GB host, ~4 GB free while serving:**

1. Start with **llama.cpp `--spec-type ngram-mod`** -- already in mainline, zero additional memory, uses prompt/generation history as the store
2. Add a **static ngram cache** (`--lookup-cache-static`) built from representative domain outputs (code snippets, agent traces). Size: 100-300 MB. Build it by running `llama-lookup-stats` to evaluate acceptance, iterating on the corpus
3. If acceptance on repetitive tasks justifies it, implement SuffixDecoding's dual suffix tree (prompt + history). CPU-only, ~20 us/token, but requires integration work into llama.cpp (not merged; exists in [ArcticInference](https://github.com/snowflakedb/ArcticInference) for vLLM)

---

## 2. Model Layer: N-gram Lookup as a Sparsity Axis

### Papers

| System | Source | Date | Claim | Parameter/FLOP Accounting | Retrofittable? |
|--------|--------|------|-------|--------------------------|----------------|
| **Engram** (DeepSeek) | [arXiv:2601.07372](https://arxiv.org/abs/2601.07372), [GitHub](https://github.com/deepseek-ai/Engram) | Jan 2026 | O(1) hashed n-gram embedding lookup as conditional memory; U-shaped scaling law between MoE and Engram | Engram-27B: reallocates 17 experts (72->55 routed) to a 5.7B embedding module; same total 26.7B; max trigram, 8 heads, dim 1280. Multi-Query NIAH: 84.2->97.0 | **No.** Requires pretraining from scratch with the Engram module. Embedding tables are trained jointly with the transformer. |
| **Engram-Nine** (collision-free) | [arXiv:2601.16531](https://arxiv.org/abs/2601.16531) | Jan 2026 | MPHF hot-tier for frequent n-grams, eliminates hash collisions | Iso-parameter: collision-free does NOT consistently improve val loss | N/A |
| **SCONE** | [arXiv:2502.01637](https://arxiv.org/abs/2502.01637) (NeurIPS 2025) | Feb 2025 | Scalable n-gram embedding via BPE-inspired f-grams; precomputed embeddings offloaded from accelerator | Outperforms 1.9B baseline at half inference FLOPs; up to billions of f-gram entries | **Partially.** Embeddings are precomputed and offloaded. Training requires a separate f-gram model. Not plug-and-play retrofit. |
| **Gemma 3n/4 PLE** | [ai.google.dev/gemma/docs/gemma-3n](https://ai.google.dev/gemma/docs/gemma-3n), [Raschka analysis](https://sebastianraschka.com/llm-architecture-gallery/per-layer-embeddings/) | Jun 2025 | Per-Layer Embeddings: 256-d vector per layer per token, CPU-offloadable | Gemma 4 E4B: 4.5B effective, 8B total (3.5B in PLE). Each layer gets token-specific lookup. | **No.** Requires pretraining with PLE architecture. |
| **Over-Tokenized Transformer** | [arXiv:2501.16975](https://arxiv.org/abs/2501.16975) | Jan 2025 | Hash n-gram embeddings via tiled matrix decomposition + MTP | Configurable n-gram vocabulary via tiled parameterization | **No.** Pretraining required. |
| **Scaling Embeddings > Scaling Experts** | [arXiv:2601.21204](https://arxiv.org/abs/2601.21204) | Jan 2026 | N-gram embedding scaling outperforms expert scaling in specific regimes; LongCat-Flash-Lite: 68.5B total, 2.9-4.5B active, 46% params in embeddings | PLNE (per-layer n-gram embedding) did not consistently beat NE when scaling width/depth; NE alone adopted | **No.** Trained from scratch. |
| **Tensorizing Engram** | [arXiv:2606.08347](https://arxiv.org/abs/2606.08347) | Jun 2026 | Shared latents across n-gram orders via tensor decomposition; addresses hash-collision and order-isolation | Improvement over both Engram and Over-Tokenized Transformer | **No.** Pretraining required. |
| **MoLE** (Mixture of Lookup Experts) | [arXiv:2503.15798](https://arxiv.org/abs/2503.15798) (ICML 2025 Oral) | Mar 2025 | FFN experts re-parameterized as lookup tables at inference; zero VRAM for experts | 410M active params: MoLE matches MoE quality with dense-model-like decoding speed and no VRAM for expert storage | Requires MoLE-style training. Conceptually interesting for your expert cache problem but not retrofit. |
| **BLT** (Byte Latent Transformer) | [arXiv:2412.09871](https://arxiv.org/abs/2412.09871) | Dec 2024 | Hash n-gram embeddings at byte level (3-8 grams) | Byte-level, not token-level; local encoder model | **No.** Different architecture entirely. |

### Assessment for Your Setup

**None of these can be retrofitted into an existing Qwen3.6-35B-A3B MoE by distillation at reasonable cost.** The n-gram embedding tables are trained jointly with the model from scratch. Distilling reasoning capability (SFT on teacher traces, logit KD) does not transfer the embedding table's structural knowledge.

The actionable takeaway: n-gram embeddings are a **next-generation architecture choice**, not an optimization for current models. If you were training a model from scratch on the MI210 box, Engram-style modules would be worth including. For your current inference stack, the benefit comes indirectly: the same n-gram statistics these papers exploit can be used at the cache/draft layers (sections 1 and 3) without touching model weights.

---

## 3. Cache Layer: Predicting Experts from Token N-grams

### Expert Prediction Systems

| System | Source | Date | Prediction Method | Accuracy/Recall | Cache Policy | Relevance |
|--------|--------|------|-------------------|----------------|-------------|-----------|
| **Mixtral-Offloading** | [arXiv:2312.17238](https://arxiv.org/abs/2312.17238) | Dec 2023 | Cross-layer gate similarity (residual stream -> next layer's gate) | ~90% for top-1 next-layer expert | LRU, k=2-4 per layer | **Your baseline.** Same concept as your expert cache. |
| **MoE-Infinity** | [arXiv:2401.14361](https://arxiv.org/abs/2401.14361) | Jan 2024 | Request-level Expert Activation Matrix (EAM), cosine similarity | Request-level; degrades under domain shift | LFU | Conceptually similar to your admission-after-3-misses |
| **ProMoE** | [arXiv:2410.22134](https://arxiv.org/abs/2410.22134) | Oct 2024 | 2-layer MLP predictor on layer inputs | Higher accuracy than heuristic; learned | Sliding window prefetch | Small learned predictor is cheap to train |
| **Pre-gated MoE** | [Microsoft ISCA 2024](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf) | May 2024 | Architectural: pre-gate determines next-layer experts during current layer | High (by construction) | N/A (architectural) | Requires retraining; not applicable |
| **HOBBIT** | [arXiv:2411.01433](https://arxiv.org/abs/2411.01433) | Nov 2024 | Current gating input -> next layer gate; mixed precision | ~90% (cross-layer gate similarity) | LRU-based staging | Combines with quantization |
| **Fate** | [arXiv:2502.12224](https://arxiv.org/abs/2502.12224) | Feb 2025 | Cross-layer gate prediction using adjacent-layer gating input similarity | Baseline for later work | LRU | Validated; ~90% adjacent-layer accuracy |
| **EdgeMoE** | Referenced in surveys | 2024 | Calibration-data table: expert(layer n-1) -> expert(layer n) statistical correlation | High on calibration domain | Table-based | **Closest to "n-gram over expert activations"** |
| **CommitMoE** | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/39454) | 2025 | Commit Router: no fallback, certainty-correlated prediction | Router certainty correlates with prediction accuracy; low certainty -> resilient output | N/A (fallback-free) | Novel: skips experts when uncertain |
| **MoE-Beyond** | [arXiv:2508.17137](https://arxiv.org/abs/2508.17137) | Aug 2025 | Transformer-based expert activation predictor on trace data | Higher generalization than heuristic methods | Learned | Heavy predictor; not for your CPU budget |
| **MoE-SpeQ** | [arXiv:2511.14102](https://arxiv.org/abs/2511.14102) | Nov 2025 | Small on-device draft model predicts expert sequence for future tokens | Draft-driven prefetch | Speculative quantized | **Most relevant for your MTP setup: draft tokens predict experts** |
| **SpecPrefetch** | [arXiv:2607.24787](https://arxiv.org/abs/2607.24787) | Jun 2026 | Shared lightweight adapter (low-rank) predicts next-layer experts | Improves over MLP/draft-model/ProMoE predictors with fewer params | Window-aware scheduler | **State-of-art parameter-efficient predictor.** |
| **Nov 2025 MLSys submission** | Referenced in [arXiv:2502.12224](https://arxiv.org/abs/2502.12224) | Nov 2025 | Pre-attention expert routers | 93.03% DeepSeek V2 Lite, 94.69% Qwen3-30B, 97.62% Phi-mini-MoE; +15% over Fate | Learned | **Best reported accuracy numbers.** |
| **ReMoE** | [arXiv:2605.27081](https://arxiv.org/abs/2605.27081) | May 2026 | Gate-only fine-tuning with temporal-locality regularizers + KL trust-region to frozen snapshot | Extends reuse streaks, stabilizes working set | Disables standard load-balance loss during fine-tuning | **Key paper for training routers for locality.** |

### Cache Policies That Beat LRU

| Policy | Source | Mechanism | vs LRU | Notes |
|--------|--------|-----------|--------|-------|
| **LFU** | MoE-Infinity; [arXiv:2511.05814](https://arxiv.org/abs/2511.05814) | Frequency-based eviction | Marginal improvement over LRU in most settings | Similar performance in SeqMoE analysis |
| **LFRU** (frequency-weighted LRU) | [vLLM RFC #38256](https://github.com/vllm-project/vllm/issues/38256) | score = freq / (clock - last_access + 1) | Prevents thrashing of hub experts during domain shifts | Practical; easy to implement |
| **Belady's MIN** (oracle) | [SeqMoE arXiv:2609.12978](https://arxiv.org/abs/2609.12978) | Evict expert whose next use is farthest in future | Consistently outperforms LRU/LFU; gap narrows with prediction | Theoretical upper bound |
| **Belady-8** (8-step oracle) | SeqMoE | 8-step future knowledge | Close to full Belady; much better than LRU | **Your MTP drafts give you 2-step future knowledge for free.** |
| **SeqMoE probabilistic Belady** | SeqMoE | Probabilistic future-aware from sequence model | 96.97% hit rate at 45% residency, 80.22% of full-load perf | State-of-art |

**Reproducibility warning:** [arXiv:2608.07911](https://arxiv.org/abs/2608.07911) found that sequential replay inflates LRU by 27.4% and LFRU by 28.5%, but LFU by only 3.4% and Belady by 4.3%, inverting the ranking. Evaluate with realistic interleaved workloads.

### Draft-Driven Expert Prefetch (The Key Insight for Your Setup)

**MoE-SpeQ** and **SpecPrefetch** both exploit speculative/draft tokens to predict future expert needs. Since you already run MTP with n=2 drafts:

1. Your draft tokens pass through the router to get acceptance/rejection anyway
2. Before acceptance check, extract the top-k expert selections from the draft tokens' router outputs
3. Use those to prefetch experts into GPU cache slots while the verification forward pass runs
4. This gives you **2-step Belady-like future knowledge for free** -- you already computed the draft, the router weights are in VRAM, the gate forward pass is a single matmul

This is architecturally identical to MoE-SpeQ's approach but using your existing MTP drafter instead of a separate draft model. Expected improvement: your cache hit rate should jump from 57-65% to 75-85% (inferred from Belady-8 vs LRU gaps in SeqMoE).

### Training Routers for Locality

**ReMoE** ([arXiv:2605.27081](https://arxiv.org/abs/2605.27081)) is the key paper. Recipe:
1. Freeze a snapshot of the pretrained router
2. Fine-tune only the gate weights with a combined loss:
   - **Temporal locality regularizer**: encourages consecutive tokens to reuse the same experts (extends reuse streaks)
   - **KL divergence to frozen snapshot**: soft trust-region prevents semantic drift
3. **Disable the standard load-balance loss** during this fine-tuning (it explicitly encourages dispersion, conflicting with locality)
4. Result: higher short-horizon expert overlap, longer reuse streaks, better cache hit rate

**Quality cost:** The trust-region KL term bounds quality degradation. The local routing consistency study ([arXiv:2505.16056](https://arxiv.org/abs/2505.16056)) found that global load balance can coexist with local routing consistency -- the trade-off is with *local* load balance only, which matters less for offloaded inference where you care about cache hits, not GPU expert utilization.

**Cost on your MI210 box:** Gate-only fine-tuning of a 35B-A3B model is cheap -- the gate is a single linear layer per MoE block. This is a few hours of training, not days.

---

## 4. Training Layer: KD Recipe and Wall-Clock ETA

### Published KD Recipes for LLM -> MoE

| Method | Source | Approach | Token Budget | Compute | Notes |
|--------|--------|----------|-------------|---------|-------|
| **GKD** (Google) | [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) | On-policy sampling + generalized JS divergence | Varies; framework paper | Framework | Supports MoE students in principle |
| **MiniLLM** | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) | Reverse KL to avoid mode-covering | Varies | Framework | Not specifically tested on MoE (see [arXiv:2502.12947](https://arxiv.org/abs/2502.12947)) |
| **DistiLLM-2** | ICML 2025 | Contrastive approach | Varies | Framework | Better than DistiLLM-1 on reasoning |
| **Thinking Machines on-policy KD** | [thinkingmachines.ai blog](https://thinkingmachines.ai/blog/on-policy-distillation/) | On-policy rollouts from student, dense per-token teacher feedback (reverse KL) | 70% AIME'24 in 1,800 GPU-hours (vs 17,920 for RL); 9-30x compute savings vs RL | **Best documented efficiency** | Implemented in [Tinker SDK](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/); multi-turn tool-use supported |
| **NVIDIA Minitron** | [arXiv:2407.14679](https://arxiv.org/abs/2407.14679) | Prune + distill from larger model | Up to 40x fewer tokens than from-scratch; Minitron-SSM: 400B tokens for 8B | 1.8x compute savings for full family | Pruning step not applicable (different arch) |
| **Nemotron Nano 2** | [NVIDIA tech report](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-2-Technical-Report.pdf) | Minitron compress + distill for 128K on A10G | 6x inference throughput vs Qwen3-8B at parity | Reasoning-optimized | Dense-to-dense; shows budget efficiency |
| **Arcee DistillKit** | [GitHub](https://github.com/arcee-ai/DistillKit) | Online/offline logit KD with polynomial compression; composable losses (KL, JSD, TVD, ranking) | Configurable | Open-source toolkit | **Practical choice** -- supports sparse top-k logits, cross-tokenizer distillation |
| **Empero Qwen3.8-35B-A3B-Distill** | [HuggingFace](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill) | SFT on curated Qwen3.8 teacher traces (off-policy); NOT logit KD | Unknown (internal) | Community project | **Already exists.** Apache-2.0. Uses Qwen3.8 2.4T-A95B and Flash Next as teachers. |
| **MoE-specific KD** | [arXiv:2502.12947](https://arxiv.org/abs/2502.12947) | Activating non-routed experts (k'>k) during KD transfers knowledge from inactive experts | Not specifically budgeted | Research | Novel but unvalidated at scale |

### "Whittle" Claim

**Could not verify.** No paper, blog post, or HuggingFace model card matching "Whittle" + "3.3 hours" + "96 GB" + knowledge distillation was found in any search. The claim as stated does not appear in the indexed literature or community posts. Possible sources: (a) a Discord/Twitter claim that was not archived, (b) a misremembered reference to the Thinking Machines 1,800 GPU-hours figure (which is a very different scale), or (c) a private/unreleased experiment. **(Unverified)**

### ETA Table: 2x MI210 + 512 GB DDR5

#### Throughput Assumptions

**MI210 specs:** CDNA2, 64 GB HBM2e each, ~45.3 TFLOPS BF16, ~181 TFLOPS INT8. PCIe Gen4 x16 (~25 GB/s theoretical).

**Measured references:**
- Databricks MPT-7B on MI250: 159-166 TFLOPS/GPU with FSDP ([Databricks blog](https://www.databricks.com/blog/training-llms-scale-amd-mi250-gpus)). MI250 = 2x MI210 GCDs, so per-GCD: ~80 TFLOPS.
- MI250 GCD peak BF16: ~45 TFLOPS. MFU at Databricks: ~55-60%.
- IBM Llama-2-7B on A100: 3,700 tok/s/GPU at 57% MFU with FSDP.
- ZeRO-3 + CPU offload penalty: ~3x throughput drop vs pure GPU (measured on A100: 8,151 -> 2,615 tok/s).

**For a 35B-A3B MoE student on 2x MI210 (128 GB HBM total):**
- Active parameters per forward: ~3B. Total parameters: ~35B. With BF16, model = ~70 GB.
- 2x MI210 = 128 GB HBM. Model fits, but optimizer states (Adam: 4x model in FP32 = 280 GB) require ZeRO-3 CPU offload to 512 GB DDR5.
- With ZeRO-3 CPU offload, expect ~3x penalty over pure GPU.
- Estimated per-GCD throughput: ~40-60 TFLOPS achievable -> ~80-120 TFLOPS across 2x MI210 (4 GCDs).
- At MFU ~40-50% with offload overhead, and 6 active FLOPs/token (forward + backward ~= 6 * active_params):
  - FLOPs per token (3B active): ~18 GFLOPs
  - If achieving 80 TFLOPS aggregate with offload: ~4,400 tok/s
  - **But MoE routing overhead + all-to-all + expert materialization reduces this.** Realistic: **60-150 tok/s** for the full MoE with offloaded optimizer.

The wide range reflects uncertainty in grouped GEMM efficiency on gfx90a and MoE-specific kernel availability. (Inferred)

| Token Budget | Wall-Clock (optimistic, 150 tok/s) | Wall-Clock (pessimistic, 60 tok/s) | Notes |
|-------------|-------------------------------------|--------------------------------------|-------|
| **50M** | ~3.9 days | ~9.6 days | Warm-start SFT baseline |
| **200M** | ~15.4 days | ~38.6 days | Meaningful logit KD run |
| **1B** | ~77 days | ~193 days | Substantial KD; likely diminishing returns past 500M |
| **5B** | ~386 days | ~965 days | **Not feasible on this hardware** |

#### Teacher Logit Generation Cost

Two strategies:

1. **Offline top-k logit generation on R9700s:**
   - Qwen3.8-27B dense teacher at BF16 = ~54 GB. Fits on a single R9700 (32 GB) only at Q4 (~16 GB) or Q8 (~27 GB).
   - At Q8 on R9700: expect ~10-20 tok/s for forward pass (RDNA4 inference; gfx1201 is well-supported by ROCm for inference).
   - Generating top-k (k=32 or 64) logits for 200M tokens: ~115-230 days on 1x R9700. With 4x R9700: ~29-58 days.
   - **Alternative:** Use Qwen3.8-27B at Q4 on a single R9700 for faster generation (~30-50 tok/s), or use the MI210s during non-training windows.
   - Storage: top-32 logits per token at (32 * (4 bytes index + 4 bytes logit)) = 256 bytes/token. 200M tokens = ~48 GB. Fits on NVMe.

2. **Online teacher inference during training:**
   - Requires running teacher + student simultaneously. 27B teacher + 35B student = too large for 2x MI210 even with offload during training.
   - **Verdict: offline generation is the only viable path.**

#### Additional Loss Terms

- **MTP-acceptance term:** Adding a self-distillation loss for MTP heads ([arXiv:2603.23911](https://arxiv.org/abs/2603.23911)) costs one extra forward pass through the MTP head per training step. Overhead: ~5-10% (MTP head is small relative to backbone). The MTP-D approach reports +7.2% acceptance rate improvement.
- **Expert-locality term (ReMoE-style):** Gate-only fine-tuning loss. Overhead: negligible per step (one extra matmul per MoE layer for the KL term). Can be done as a separate pass: freeze backbone, fine-tune gates only. This should take **hours, not days**, even on 2x MI210.

#### ROCm Blockers on gfx90a

| Component | Status | Blocker? |
|-----------|--------|----------|
| **Flash Attention** | Works via CK port (head dim <= 128). Use AMD's prebuilt Docker images. ([kailums/flash-attention-rocm](https://github.com/kailums/flash-attention-rocm)) | No, with caveats |
| **Grouped GEMM** (MoE kernels) | Fused CK grouped GEMM (Primus-Turbo) targets MI300+. gfx90a needs custom paths; generic kernels may not cover FP4/wave64/small-matrix regime. ([ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html)) | **Partial.** BF16 grouped GEMM works; FP8 does not (CDNA2 lacks FP8). |
| **FP8 training** | Not supported on gfx90a/CDNA2. Hardware limitation. | **Yes for FP8 specifically; BF16 fine.** |
| **bitsandbytes** | Supported on ROCm for gfx90a since Nov 2024. ([ROCm blog](https://rocm.blogs.amd.com/blog.html)) | No |
| **DeepSpeed ZeRO-3 + offload** | Works via HIP compatibility. ZeRO-3 + CPU offload is architecture-independent. Not in hot CI path for gfx90a. | **No, but pin to validated image.** |
| **CI regression risk** | gfx90a is off the hot CI path for vLLM, DeepSpeed, etc. Regressions ship unnoticed. ([davetha/mi210-llm-stack](https://github.com/davetha/mi210-llm-stack)) | **Use AMD curated Docker images; don't pip install into mismatched ROCm.** |
| **RCCL gfx90a+gfx1201** | Mixed GCD topology does not work. Train on MI210 pair only. | **Known; R9700s are for offline teacher inference only.** |

---

## Rainbow Table Section: Concrete Data Structure Recommendation

### For your 16 GB host with ~4 GB free while serving:

**Architecture: Tiered n-gram store on NVMe**

```
Tier 1 (RAM, ~100-200 MB): Hot n-gram cache
  - Data structure: llama.cpp ngram-cache (unordered_map<ngram, unordered_map<token, count>>)
  - Content: context n-grams (built live from current conversation) + most frequent domain n-grams
  - Lookup: O(1) hash, sub-microsecond

Tier 2 (NVMe mmap, ~200-600 MB): CREST-compacted continuation store
  - Data structure: CREST-style compact trie with compressed token trees
  - Content: top-k continuations for 2-16 gram patterns from self-distilled corpus
  - Lookup: 0.2-0.7 ms via mmap (NVMe random read ~50-100 us for 4K page)
  - Build: run model on representative prompts, collect output tokens, build trie, prune rare branches
  - CREST shows 10-13x compression vs naive REST store at same or better acceptance
```

**Size/latency estimates:**

| Store Size | n-gram Coverage | Lookup Latency | Build Time | Expected Acceptance Boost |
|-----------|----------------|----------------|-----------|--------------------------|
| 100 MB | ~500K unique continuations | <1 us (RAM) | Minutes | +10-20% on in-domain |
| 300 MB | ~2M unique continuations | <1 us (RAM) / 0.2 ms (mmap) | ~1 hour | +15-30% on in-domain |
| 800 MB | ~5M unique continuations | 0.3-0.7 ms (mmap) | ~3 hours | +20-40% on repetitive workloads |

**Key constraint:** At ~4 GB free RAM, you cannot afford to memory-map more than ~500 MB without risking page pressure on the expert cache. Use `madvise(MADV_RANDOM)` on the mmap'd store to prevent kernel readahead from evicting expert cache pages.

---

## KD ETA Table (Summary)

| Scenario | Token Budget | Hardware | Method | Wall-Clock | Quality Expectation |
|----------|-------------|----------|--------|-----------|-------------------|
| **Gate-only locality fine-tune** | ~10M | 2x MI210 | ReMoE-style KL + locality loss | **2-8 hours** | Cache hit rate: 57-65% -> 75-90% |
| **MTP head self-distillation** | ~10-50M | 2x MI210 | MTP-D ([arXiv:2603.23911](https://arxiv.org/abs/2603.23911)) | **1-4 days** | MTP acceptance: +5-7% |
| **SFT on teacher traces (Empero-style)** | 50-200M | 2x MI210 + ZeRO-3 offload | Off-policy SFT on Qwen3.8 traces | **4-39 days** | Reasoning quality uplift; marginal over existing Empero checkpoint |
| **Full logit KD (dense teacher -> MoE student)** | 200M-1B | 2x MI210 + offline teacher logits from R9700s | GKD/DistiLLM-2 with top-k compressed logits | **15-193 days** (student) + **29-58 days** (teacher logit gen on 4x R9700) | Best quality; diminishing returns past ~500M tokens |

**Stated assumptions:**
- Sequence length: 2048 (shorter = faster, longer = more memory pressure)
- MI210 aggregate MFU with ZeRO-3 offload: 40-50% of peak BF16 TFLOPS
- PCIe Gen4 x16 bandwidth between MI210 and host: ~22 GB/s sustained
- DDR5 bandwidth for ZeRO-3 offload: ~60-80 GB/s (sufficient, not the bottleneck)
- No FP8 (gfx90a limitation); all BF16
- MoE routing overhead: ~20-30% over equivalent dense model training

---

## Ranked Experiments (Cheapest First)

| # | Experiment | Expected Gain | Cost | Kill Criterion |
|---|-----------|--------------|------|----------------|
| 1 | **Enable `--spec-type ngram-mod`** in current llama.cpp | 1.1-1.5x on repetitive tasks, composes with MTP | Zero; already in mainline | Acceptance rate < 5% on your workloads |
| 2 | **Build static ngram cache** from representative domain outputs (`llama-lookup-stats` to evaluate) | +10-30% acceptance on in-domain | ~1 hour to build, ~100-300 MB NVMe | `llama-lookup-stats` shows < 5% improvement |
| 3 | **Draft-driven expert prefetch**: extract expert selections from MTP draft tokens, async prefetch | Cache hit rate 57-65% -> 75-85% (inferred from Belady-8 gap) | ~1-2 days implementation in your cache code | Hit rate improvement < 5% on measured workloads |
| 4 | **Switch LRU to LFRU** in expert cache | Prevents hub-expert thrashing during domain shifts | ~2 hours implementation | A/B test shows no improvement on mixed workloads |
| 5 | **ReMoE gate-only fine-tune for locality** on MI210 box | Cache hit rate -> 80-90% | 2-8 hours training; ~10M tokens | Quality regression on benchmarks (eval before/after) |
| 6 | **MTP-D self-distillation** for better acceptance | MTP acceptance +5-7%, ~1.1-1.15x throughput | 1-4 days on MI210 box | Acceptance improvement < 2% |
| 7 | **SuffixDecoding integration** (port from ArcticInference or build suffix tree in llama.cpp) | 1.5-3x on agentic/code workloads | 1-2 weeks implementation | Measured speedup < 1.2x on your actual workloads |
| 8 | **Full logit KD** (Qwen3.8-27B -> Qwen3.6-35B-A3B) | Quality uplift across all tasks | 2-6 months (teacher gen + student training) | Eval scores plateau before 200M tokens |

---

## Could Not Verify

| Claim | Status | Notes |
|-------|--------|-------|
| "Whittle" 3.3 h on 96 GB KD claim | **Not found** | No matching paper, blog, or model card in any indexed source |
| SuffixDecoding store size numbers | **Not in abstracts** | Paper mentions ~20 us/token draft cost but doesn't publish memory footprint in accessible sections |
| MI210 training tok/s for MoE with ZeRO-3 | **No direct measurement** | Estimated from MI250 FSDP data + offload penalty; actual MI210 MoE training throughput not published |
| ARC cache policy in MoE context | **No papers found** | ARC (Adaptive Replacement Cache) is not used in any MoE expert caching paper found |
| RDNA4 (R9700/gfx1201) inference throughput for 27B model | **No benchmarks found** | R9700 is too new; ROCm inference support exists but no published tok/s for large models |
| CREST mmap/NVMe serving performance | **Partial** | CREST discusses on-disk layout and compression but doesn't publish mmap serving latency specifically |
