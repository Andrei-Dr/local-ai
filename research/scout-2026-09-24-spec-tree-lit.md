# Tree / Multi-Draft Speculative Decoding: Literature Report

*Date: 2026-09-24. Setting: MoE hybrid (Qwen3.6-35B-A3B, 256 experts top-8, 30 GDN + 10 attn layers), 4 GB GPU, CPU expert compute, MTP draft head n=3 ~83% per-token.*

---

## 1. Tree / Multi-Draft Speculative Decoding with Exact Output

### SpecInfer (Miao et al., 2024)
**arXiv: 2305.09781** — "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification"

Multiple small draft models jointly build a token tree. The target verifies the whole tree in one pass using **multi-step speculative sampling (MSS)**: each draft k is accepted with probability min{1, q^(k)/p^(k)}, where q^(k) is defined via the normalized positive residual of the previous round. If all K drafts reject, a token is sampled from the final residual. Output distribution is exactly preserved. Branches are sampled (not top-k). Reported 1.5--3.5x speedup; MSS improves verified tokens by 1.2--1.3x over naive per-draft rejection.

### SpecTr (Sun et al., NeurIPS 2023)
**arXiv: 2310.15141** — "SpecTr: Fast Speculative Decoding via Optimal Transport"

Frames multi-draft acceptance as an optimal transport problem with membership cost (extension of maximal coupling). Allows K i.i.d. drafts per position. The optimal transport plan is solvable via LP (exponential in K); they propose a (1-1/e)-optimal algorithm computable in near-linear time in vocabulary size. Drafts are sampled. Tree is flat (K candidates per position, not deep).

### SpecHub (Sun et al., 2024)
**arXiv: 2411.05289** — "SpecHub: Provable Acceleration to Multi-Draft Speculative Decoding"

Simplifies the OT problem into a compact LP with linear overhead by exploiting sparsity of the joint distribution. Generates 0.05--0.27 more tokens/step than recursive rejection sampling (RRS). ICLR 2025 companion paper (arXiv: 2410.18234, "Multi-Draft Speculative Sampling: Canonical Decomposition and Theoretical Limits") establishes the theoretical limits.

### Sequoia (Chen et al., 2024)
**arXiv: 2402.12374** — "Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding"

Uses **dynamic programming** to find the tree structure maximizing expected accepted tokens under a budget (tree size) and optional depth constraint. Recursive sub-structure: the DP finds tree T of size n (depth <= d) maximizing F(T). Uses **sampling without replacement** for robustness across temperatures. A separate **hardware-aware optimizer** selects tree size/depth given target/draft latencies. Speedups: up to 4.04x (Llama2-7B, A100). In offloading: Llama2-70B on single RTX-4090 at 0.57s TBT (8x faster than optimized offloading baselines). Tree is fixed offline per model pair + hardware.

### EAGLE-2 (Li et al., EMNLP 2024)
**arXiv: 2406.16858** — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"

Observes that the EAGLE draft model is well-calibrated: its confidence scores approximate acceptance rates. Builds a **context-aware dynamic tree** at each step via expansion (grow by confidence) then reranking (prune). No additional training beyond the EAGLE draft head. 3.62x mean speedup (20--40% over EAGLE-1). Branches are top-k selected by confidence, then verified with standard speculative sampling.

**EAGLE-3 (2025)** shifts from feature extrapolation to direct token prediction with richer hidden-layer information. **RADAR (2025, arXiv: 2512.14069)** uses RL to learn tree construction.

### Medusa (Cai et al., ICML 2024)
**arXiv: 2401.10774** — "MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"

Adds K lightweight decoding heads to the target model itself (no separate draft model). Head k predicts position t+k+1. Top predictions per head form a tree verified via tree attention mask in a single forward pass. 2.2x speedup (heads-only fine-tune), 2.3--3.6x (backbone also updated). Branches are top-k per head; tree is static (pruned by probability).

### Traversal Verification (Weng et al., 2025)
**arXiv: 2505.12398** — "Traversal Verification for Speculative Tree Decoding"

Reverses the verification direction: **leaf-to-root** instead of root-to-leaf. If a leaf-to-root path is accepted, the entire sequence is kept. Preserves subsequences that top-down methods prematurely discard when a parent node rejects. Under review.

### RheoSampling (2026)
**arXiv: 2609.21827** — "RheoSampling: Resolving the One-Hot Dilemma in Stochastic Dynamic-Tree Speculative Decoding"

Identifies that dynamic-tree methods (EAGLE-3) collapse draft distributions to one-hot under T>0, killing acceptance. Decouples tree construction (proxy probability for top-K expansion) from verification (true sampling probability). First dynamic-tree method with both context-aware construction and stochastic sampling while remaining lossless. Uses OT-based verification.

---

## 2. Hybrid Tree Schemes

### Deterministic Head + Stochastic Tail
RheoSampling (above) is the most direct: deterministic top-K slots plus one sampled token, each carrying separate proxy/true probabilities. BASTION (arXiv: 2605.29727, "Budget-Aware Speculative Decoding with Tree-structured Block Diffusion Drafting," 2026) frames tree expansion as cost-aware best-first search: an acceptance surrogate estimates expected accepted length via path confidence, an online latency estimator calibrates a hardware-aware roofline model, and adaptive expansion grows the tree until marginal gains no longer justify cost.

### Mixed Drafting Strategies
**STAND** (arXiv: 2506.04708, "Accelerated Test-Time Scaling with Model-Free Speculative Sampling," 2025) combines multi-level n-gram lookahead (4-gram down to unigram) with stochastic tree drafting via Gumbel-Top-K parallel sampling (precomputed cached noise). No neural draft model needed.

### Optimal Tree Shape Given Per-Node Verify Cost
Sequoia's DP is the canonical answer for fixed offline trees. For online cost-aware shaping, BASTION's best-first expansion with a roofline latency model is the most relevant. Neither paper models per-node cost as a function of distinct-expert count (our setting), but the DP framework is directly extensible: replace the acceptance-probability objective with a cost-penalized objective where cost = f(distinct experts touched by the batch).

---

## 3. Speculative Decoding for MoE Models

### The Core Problem
Draft tokens in a verify batch route to different experts. Distinct-expert count grows with batch size (our measurements: 8, 22, 36, 56 for K=1,4,8,16). This increases weight-transfer volume proportionally, eroding speedup. Cascade (below) measured verification cost 2--3x higher with speculation enabled, causing up to 1.5x slowdown.

### Cascade / Utility-Driven Speculation (Saxena et al., 2025)
**arXiv: 2506.20675** — "Utility-Driven Speculative Decoding for Mixture-of-Experts"

Defines **speculation utility** = token gains / verification cost. Exhibits iteration-level locality. Disables speculation when utility < 1; otherwise tests multiple K values and picks the utility-maximizing K. Implemented in vLLM on five MoE models. Limits slowdown to 5% (vs 1.5x baseline), improves throughput 7--14% over static K.

### EVICT (Pan et al., 2026)
**arXiv: 2605.00342** — "Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding"

Training-free, hyperparameter-free, lossless. Truncates the draft tree before verification, retaining only the cost-effective prefix. Uses drafter signals to estimate candidate benefit, combined with offline-profiled verification cost. Up to 2.35x over autoregressive, average 1.21x over EAGLE-3. Compatible with SGLang.

### The Limits of Speculation in MoE (Amankulov & Mamatin, 2026)
**arXiv: 2609.22156** — "The Limits of Speculation: Bounding Speculative Decoding in Mixture-of-Experts"

Frames budget selection as an offline Stochastic Shortest Path (SSP) problem. Builds a counterfactual "Oracle" simulation. Finds that rejected candidates form a strict linear boundary in "Delta Space," suggesting a simple marginal-cost-vs-progress condition drives the optimum. 24 pages, 17 figures, 7 tables.

### Expert Coactivation (Training-Time, 2026)
**arXiv: 2609.22471** — "Efficient Mixture-of-Experts with Speculative Decoding via Expert Coactivation"

Changes MoE training (not serving) to encourage neighboring tokens to choose similar expert sets during verification. Four changes to standard MoE training; SD algorithm and drafting mechanism are untouched.

### Expert Prefetching (Offloading + Speculation)
- **SP-MoE** (arXiv: 2510.10302): First SD-aware expert-offloading framework. Uses structural correspondence between draft/target to prefetch likely experts ahead of verification. Cutoff-layer policy bounds per-layer prefetch depth. 1.07--3.5x TPOT speedup.
- **MoE-SpeQ** (arXiv: 2511.14102): Draft model predicts required expert sequence; runtime prefetches from host memory. Adaptive governor via Amortization Roofline Model. Up to 2.34x on Phi-MoE.
- **SPICE** (arXiv: 2608.21240): Lightweight draft + confidence-aware adaptive lookahead. Up to 3.12x TPOT on DeepSeek-V2-Lite / Qwen2-57B.

---

## 4. Speculation with Recurrent / Hybrid Models

### The State Problem
Transformers verify a tree by packing it into a sequence with a topology-aware attention mask. Recurrent models (Mamba, GDN) carry a fixed-size state updated causally. State cannot be truncated like a KV cache; snapshots per draft position are needed for rollback, and they cannot be shared across branches.

### STree (Wu et al., 2025)
**arXiv: 2505.14969** — "STree: Speculative Tree Decoding for Hybrid State-Space Models"

First tree-based speculative decoding for SSMs/hybrids. Exploits the structure of accumulated state transition matrices for Mamba2 (scalar decay => cheap cumulative sum). Does not generalize to non-commuting matrix transitions (GDN). Code: github.com/wyc1997/stree.

### SpecLA (Wang et al., 2026)
**arXiv: 2607.16673** — "SpecLA: Efficient Speculative Decoding for Linear-Attention Models"

Runtime for stateful linear-attention targets. Three verification paths depending on draft shape: state-resident serial (short chains), tree-masked factorized (inserts tree dependency mask into GDN UT transform for branching), chain-decomposed hybrid. Stores compact factors during verification to recover accepted states. Confidence pruning + target-aligned EAGLE-style drafter. Up to 1.70x on GDN-1.3B (H100).

### TreeWY (Ghantasala, 2026)
**arXiv: 2608.20961** — "TreeWY: Speculative Verification for Gated DeltaNet Hybrids"

Directly targets GDN (our architecture). Eliminates per-node state snapshots via a **tree-structured WY transform of the gated delta rule**: computes every draft node's output with a single triangular solve, reconstructs only the accepted state on commit. Stores a small pseudo-value matrix instead of per-node states. Tested on Qwen3.5-35B and 397B. Reduces speculative recurrent-state memory at identical acceptance length; freed HBM converts to higher throughput / lower TTFT where memory is binding. Derivation depends only on the gated delta rule, not other architectural details. **This is the most directly relevant paper for our setting.**

### Bole (SGLang, concurrent Aug 2026)
UNCONFIRMED — mentioned in TreeWY as concurrent work. Rewrites the linear-attention recurrence into a tree-structured closed form for parallel tree verification. No separate arXiv ID found.

### SpecMamba (2025)
**arXiv: 2509.19873** — "SpecMamba: Accelerating Mamba Inference on FPGA with Speculative Decoding"

FPGA accelerator. Memory-aware hybrid backtracking strategy; FIFO-based tree verification with tiling. Hardware co-design rather than algorithmic contribution.

---

## 5. Improving MTP / Draft Heads Cheaply

### MTP-D: Self-Distillation for MTP (Zhao et al., 2026)
**arXiv: 2603.23911** — "Self-Distillation for Multi-Token Prediction"

Self-distillation with minimal additional training cost. Boosts MTP head acceptance rates by **+7.5%**. Looped extension strategy enables economical MTP head extension, further speedup of +220.4% over 1-head MTP.

### FastMTP (2025)
**arXiv: 2509.18362** — "FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction"

Documents the **acceptance-rate collapse** in vanilla MTP: ~70% at k=1, ~10% at k=2, ~0% at k=3. After self-distilled fine-tuning: 80% at k=1, 56% at k=2, 36% at k=3. Combined with vocab compression: 2.03x speedup (82% improvement over vanilla MTP 1.21x).

### Gumbel Distillation (2026)
**arXiv: 2603.22216** — "Gumbel Distillation for Parallel Text Generation"

Applied to Medusa-style heads. Gumbel conditioning improves per-head acceptance: +4.5% (Head 1) to +22.0% (Head 3) on GPT-2-Small; +8.9% to +37.6% on Vicuna-7B. Gains grow with head index (later heads benefit more).

### AdaMTP (2026)
**arXiv: 2608.00434** — "AdaMTP: An Adaptive Training Paradigm for Multi-Token Prediction"

Adaptive training paradigm. UNCONFIRMED — title and arXiv ID from search results; no abstract fetched.

### Vocabulary Trimming for Draft Heads

- **VOCABTRIM** (arXiv: 2506.22694, 2025): Training-free inference-time pruning. Reconstructs drafter LM head with only frequent tokens. Slightly reduces acceptance rate but significantly cuts drafting latency in memory-bound settings. +16% MBSU for Llama-3.
- **Balancing Coverage and Draft Latency** (arXiv: 2603.05210, 2026): Training-time approach. Formulates vocab selection as constrained optimization (coverage vs FLOPs proxy). Evaluated on EAGLE-3 / SGLang.
- **NanoSpec** (arXiv: 2605.26444, 2026): Dynamic per-context vocab (max 3k tokens). Bridges gap between oracle speed of small vocabs and acceptance-rate loss from static pruning.

### Dynamic Draft Length / Confidence-Based Stopping
- **SpecDec++** (ICML 2024): Adaptive candidate length K.
- **AdaEAGLE** (Dec 2024): 3-layer MLP predicts draft length from penultimate token's embedding.
- **TapOut** (arXiv: 2511.02017): Bandit algorithm for online dynamic speculation; no threshold tuning.
- **FailFast** (2025): Confidence threshold tau; extends speculation by N if all tokens exceed tau, repeats until low-confidence token or max length.
- **SGLang built-in**: EMA of accepted length, switches between predefined length tiers with pre-captured CUDA graphs.

---

## 6. Offloaded / CPU-GPU Hybrid Inference + Speculation

### SpecExec (Svirschevski et al., NeurIPS 2024)
**arXiv: 2406.02532** — "SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices"

**Key insight**: with offloading, verify is so memory-bound that hundreds to thousands of tokens can be verified in the same time as one token. This inverts the tree-size calculus: use a **massive tree** (much larger than on-chip methods). Prior methods plateau at ~10 accepted tokens; SpecExec generates up to 20 tokens per target iteration.

**Draft model choice**: favors a large, capable draft (e.g., Llama2-7B drafting for 70B) because the draft fits on-GPU and a single draft forward pass is still far cheaper than one offloaded target pass.

**Tree construction**: deterministic (top-k most probable continuations), not sampled. Builds a "cache tree" for the target model. Deterministic construction maximizes cache hit rates in offloading.

**Results**: 50B+ models on consumer GPUs with RAM offloading at 4--6 tok/s (4-bit) or 2--3 tok/s (16-bit). Code: github.com/yandex-research/specexec.

### Sequoia in Offloading Mode
Sequoia (Section 1) also targets offloading: Llama2-70B on RTX-4090, 0.57s TBT (8--9x over DeepSpeed-Zero-Inference). Uses its DP-optimal trees with large budgets tuned for offloading latency ratios.

### MoE-Specific Offloading + Speculation
SP-MoE, MoE-SpeQ, SPICE, SpecPrefetch (Section 3) all combine speculation with expert prefetching across PCIe for offloaded MoE inference. The key shared insight: use the draft model's routing predictions to prefetch the experts the target will need, hiding PCIe transfer latency behind draft computation.

---

## Summary of Relevance to Our Setting

| Concern | Most relevant work |
|---|---|
| Tree verification with GDN recurrent state | **TreeWY** (eliminates snapshots via WY transform), **SpecLA** (topology-aware kernels) |
| MoE verify cost scaling with distinct experts | **EVICT** (truncate tree to cost-effective prefix), **Cascade** (utility-driven K selection) |
| Optimal tree shape given per-node verify cost | **Sequoia DP** (extensible to cost-penalized objective), **BASTION** (online cost-aware best-first) |
| Large tree because verify is memory-bound | **SpecExec** (massive trees for offloaded inference) |
| Improving MTP acceptance cheaply | **MTP-D** (+7.5% acceptance via self-distillation), **FastMTP** (collapse fix + vocab trimming) |
| Expert prefetching across PCIe | **SP-MoE**, **MoE-SpeQ**, **SPICE** (draft routing predicts target experts) |
