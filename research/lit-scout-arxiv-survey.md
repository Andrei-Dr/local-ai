# Literature Survey: Decode-Speed Techniques for Ternary-Bonsai-27B on a 4 GB Turing Box

- **Date:** 2026-09-19
- **Sources searched:**
  - **arXiv API** (`export.arxiv.org/api/query`), relevance- and date-sorted, live through submissions dated 2026-09-17. Abstracts pulled via `id_list`; full text read from `arxiv.org/html/<id>` for the decision-critical papers.
  - **alphaXiv** (`alphaxiv.org`, `api.alphaxiv.org/papers/v3/<id>`). Yield: effectively nothing. The homepage trending list contained one tangentially relevant paper (DeepSeek-V4.1-Flash KV compression, 2609.19969). The API returned empty `resources` and zero citations for all 12 key papers queried, and no comments endpoint was reachable, so no community implementation notes were obtained.
  - **GitHub:** every repository named below returned HTTP 200 at survey time. READMEs of `davetha/vllm-expert-cache` and `davetha/r9700-lru-expert-cache` were read for the expert-cache mechanism.
- **Helper scripts:** `research/scripts/axv/` (`q.py` = arXiv search, `ids.py` = abstracts by ID, `ax.py` = alphaXiv metadata probe).

## Tagging

- **[P]** = read in the paper or its abstract.
- **[I]** = my inference; not stated by any source.
- **[M]** = measured on the target box by the team lead.

Papers with **no released code**: TreeWY, Bole, SpecLA, SpecMD, Prox, BITCOS.

## System under optimization

GTX 1650 SUPER 4 GB (Turing cc 7.5), i5-10400F 6c/12t AVX2-only, 15 GB DDR4-2667 dual-channel, PCIe 3.0 x16, PrismML-Eng/llama.cpp fork (`prism` branch). Model: Ternary-Bonsai-2-27B, dense, ~2.1-2.45 bpw group-128, 64 blocks (16 full-attention GQA 24q/4kv head-dim 256, 48 gated-delta-net), optional grafted MTP head. Planned next: ~30B-class MoE (~3-4B active) with a device-side expert cache.

---

## 1. Corrections driven by measurements

**[M]** The fork's x86 PQ2_0 CPU kernel was scalar-only without VNNI; an AVX2 path (38.5 -> 5.6 cycles per 32 weights) took decode from 0.71 to 2.99 tok/s.

**[M]** With `GGML_OP_OFFLOAD_MIN_BATCH=2`, CPU-resident weights stream over PCIe for GPU verification. Forward-pass cost:

| Batch N | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| ms | 334 | 519 | 558 | 649 | 946 |

Consequences:

1. My first-pass inference **[I]** that streamed-weight GPU verification would be compute-bound on the 1650S was **wrong**. The curve is PCIe-dominated: about 185 ms fixed jump from N=1 to N=2, then about 23 ms per token up to N=8 and about 37 ms per token from 8 to 16. This is exactly the SpecExec regime (2406.02532 **[P]**: with offloaded parameters the engine can process hundreds of tokens at about the cost of one). SpecExec-style operation is un-rejected.
2. Dovetail's short-draft finding (gamma=7 on a 4-core PC, because CPU verification latency grows linearly with draft tokens **[P]**) no longer applies: verification now runs on the GPU, not the CPU.
3. The CPU kernel now matters only for N=1 passes. Once speculation is on, every pass has N>=2 and goes to the GPU. It is off the critical path.

### Expected tok/s from the measured curve **[I]**

Chain drafts, per-token acceptance `a`, expected tokens per pass `(1 - a^N) / (1 - a)`, divided by `T_ver(N)`:

| a | N=4 | N=8 | N=16 |
|---|---|---|---|
| 0.6 | 3.90 | 3.79 | 2.64 |
| 0.7 | 4.54 | 4.84 | 3.50 |
| 0.8 | 5.29 | 6.41 | 5.14 |
| 0.9 | 6.16 | 8.78 | 8.61 |

Non-speculative baseline: 2.99 tok/s. N of about 8 is the chain optimum for `a >= 0.7`; N=4 for `a <= 0.6`; N=16 pays only at `a >= 0.9` (long suffix matches in code or JSON). A tree with 8 nodes beats a chain of 8 whenever `a < ~0.75`, which is what makes item 5 worth doing. This table assumes constant per-position acceptance; see the addendum for why that assumption fails for a single recursive MTP head.

---

## 2. Ranked techniques (decode gain per unit of implementation effort, this box)

### 1. [flag-only] Calibrate draft length to the measured cost curve

- **Bole** (2608.01651, 2026-08-03) Eq. 11-12 **[P]**: cost model `T_ver = T_full(n, L) + T_linear(n, d) + T_dense(N) + T_other(N)`; profile offline and pick `N* = max{N : T_ver(N) <= (1 + eps) * T_dec}`. Reported peak budgets of 128 tokens (A100) and 256 (GB10) are hardware-specific and do not transfer.
- **[I]** On this box maximize `E[accepted(N)] / T_ver(N)` directly. Bole's equal-cost criterion assumes verification costs about the same as decode; here it is only 1.94x at N=8.
- **EntMTP** (2606.27550, 2026-06-25) **[P]**: training-free scheduler that switches between tree topologies on a running estimate of local entropy. 1.15x over static Hydra, 1.36x peak over Medusa.
- **r9700-lru-expert-cache README** **[P]**: MTP-3 vs MTP-4 on the same stack: prose 94.0 vs 89.4 tok/s, JSON 115.7 vs 122.2, code 93.1 vs 105.8. Depth is workload-dependent, not a one-way dial.
- **[I]** Make depth adaptive on a rolling acceptance estimate rather than fixed.

### 2. [small patch] Arbitrated hybrid drafter: suffix/n-gram first, model drafter as fallback

- **SuffixDecoding** (2411.04975, 2024-11-07, NeurIPS 2025 spotlight; code: `snowflakedb/ArcticInference`, vLLM) **[P]**:
  - Suffix trees over the prompt plus previous outputs (a per-request tree and a global tree).
  - `MAX_SPEC = alpha * match_len`, alpha in 1-4. Nodes scored by the product of child-frequency probabilities. Use the suffix draft if its score exceeds `tau`, otherwise the model drafter.
  - Alone on Spec-Bench: 1.75 accepted tokens per step (reasoning 1.50, math_reasoning 1.64, coding 1.96, roleplay 1.26) against EAGLE-3 at 4.65; speedup 1.66x vs 2.37x.
  - Hybrid: 2.5x on Spec-Bench, beating EAGLE-3 (2.4x) and Token Recycling (2.2x). Agentic SQL: hybrid 4.1x with 7.5 tokens per step; standalone 5.3x.
  - Cost: lookup ~11 us and update ~4 us per token, flat as the tree grows.
- **Spec-decoding benchmark for test-time scaling** (2509.04474, 2025-08-30) **[P]**: simple n-gram methods capture the repetition in reasoning traces; recommends combining them with model-based drafters.
- **STAND** (2506.04708, 2025-06-05, EMNLP 2025 oral) **[P]**: logit-preserving n-gram table, Gumbel-top-k sampling, data-driven tree. 60-65% latency reduction on AIME-2024, GPQA-Diamond, LiveCodeBench. Largest gains are multi-trajectory; single-trajectory gains also claimed.
- **SSR** (2608.20359, 2026-06-17) **[P]**: partial-CoT answer distribution as drafter, plus a suffix cache seeded from the draft. Up to 24.1% latency improvement on Qwen3.5 and Gemma-4. Training-free.
- **SwitchSD** (2609.20186, 2026-09-17) **[P]**: accidental n-gram overlaps cause false-positive triggers that degrade throughput. Trained probes on hidden states (AUC > 0.99) gate copy vs neural drafting; +15% over EAGLE-3. The probe **needs training**.
- **Oilbird** (2608.03839, 2026-08-04) **[P]**: re-keys the lookup pool by hidden states the verifier already computed and merges into an existing lexical drafter's tree. +24-29% accepted length in three drafters; 4.4x on API-Bank vs 3.9x for the best training-free baseline and 2.0x for EAGLE-3. Training-free.
- **[I]** With the measured curve, a wrong 8-token n-gram draft costs 1.94x for one token. Gate on match length >= 3-4, and extend an n-gram draft with model-drafted tokens only after the n-gram part ends.

### 3. [flag-only first, then small patch / real kernel work] Cut bytes streamed per verification pass

The verify pass is PCIe-bound, so bytes equal time.

- **a. Quantize KV to buy resident layers [flag-only].** **[I]** KV is 16 layers x 4 heads x 256 x 2 = ~65 KB per token at fp16, about 2.1 GB at 32K context. q8 or q4 frees 1-1.5 GB, about 8-12 more resident layers of ~117 MB each, about 80-120 ms off every N>=2 pass at ~12 GB/s.
- **b. Pinned host buffers plus copy/compute overlap [small patch].**
  - **ATSInfer** (2607.10183, 2026-07-11) **[P]**: tensor-granularity placement, load-aware dynamic transfer, asynchronous CPU-GPU coordination. Up to 3.29x decode and 1.94x prefill over layer/expert-level systems on consumer devices. No code seen.
  - **SpecOffload** (2505.10259, 2025-05-15; code: `MobiSense/SpecOffload-public`) **[P]**: interleaves draft execution into the offload pipeline. 4.49x GPU utilization, 2.54x throughput. Batch/throughput oriented.
- **c. Denser packing on the wire with GPU-side unpack [real kernel work].**
  - **BITCOS** (2609.16338, 2026-09-14) **[P]** benchmarks **this fork**: baselines are PrismML `Q2_0` (2-bit, fp16 scale per 128) and `TQ1_0`. Format = presence bitmap + compacted sign vector = `2 - z` bpw for zero density `z`.
  - Measured zero density: Bonsai-27B 29.66% (lowest of 29 models; Bonsai 1.7B/4B/8B are 38-40%) -> 1.70 bpw against Q2_0's 2.125.
  - GPU unpack gains 1.09-1.22x (Arc 140V) and 1.02-1.27x (Arc Pro B70). AVX2 CPU path needs 5 instructions per mask where AVX-512 needs 1; end-to-end 1.02-1.15x on Arrow Lake DDR5, 1.10-1.18x on a 64-core server, and **loses on every model** on 8-core Lunar Lake (instruction-bound). No DDR4 platform tested. No code.
  - **[I]** Stream a TQ1_0-class ~1.69 bpw packing and unpack on the GPU: about 20% fewer PCIe bytes per pass.

### 4. [flag-only experiment] A real draft model on the GPU (SpecExec / Dovetail style)

- **SpecExec** (2406.02532, 2024-06-04) **[P]**: the draft builds a most-probable-continuation "cache" tree, validated in one target pass. Up to 20 tokens per target iteration; 4-6 tok/s for 50B+ models with RAM offload on consumer GPUs. Uses 128-256-token trees.
- **Dovetail** (2412.18934, 2024-12-25; code: `ddInference/Dovetail`) **[P]**: draft on GPU, target on CPU.
  - GTX 1050 Mobile 4 GB + i5-9300H: 1.74x (Llama2-7B-8bit, 6.35 tok/s) and 1.79x (Vicuna-13B-8bit, 3.36 tok/s).
  - RTX 2080S 8 GB: 3.08x / 3.78x (MT-bench / HumanEval) vs SpecExec 2.36x / 2.98x and plain offload 0.45x.
  - With 3 GB VRAM: 4.62-5.86 tok/s on Llama2-7B; spare VRAM is used to host target layers.
  - Deeper drafts (5 transformer blocks) beat wider ones; 6 blocks regresses.
- **[I]** A Bonsai-1.7B ternary (~0.5 GB) as `--model-draft` should hold acceptance over 8-16 tokens far better than a single grafted MTP head applied recursively. It costs about 4 resident target layers (~40 ms more per pass). Confirmed as the right direction by the addendum sweep.

### 5. [real kernel work] GDN rollback by reconstruction, then tree verification in CUDA

- **TreeWY** (2608.20961, 2026-08-21) **[P]**:
  - Status quo in vLLM and SGLang: snapshot the full recurrent state at every draft position (k+1 blocks per sequence per GDN layer; 120 MiB per sequence at k=3 on Qwen3.5-35B, 360 MiB on 397B); snapshots are unshareable across tree branches.
  - Tree-structured WY transform of the gated delta rule: one triangular solve for the whole draft window; stores a pseudo-value matrix `O(N d_v)`, 128x smaller per head; reconstructs only the accepted state on commit.
  - 0.97-0.99x where memory does not bind, up to 1.49x where it does; ~40x lower p99 TTFT at saturation. Trees are "enabled and correct, not a speedup" (piecewise, not CUDA-graph-capturable). Tree (3,3,3) raises acceptance length 1.883 -> 2.786.
  - conv1d state is not addressed. vLLM fork, not upstreamed, no code.
- **Bole** (2608.01651, 2026-08-03) **[P]**:
  - Closed form `O = D_P Q S_pre + C (I + G)^-1 D_beta (V - D_P K S_pre)`; `(I + G)^-1` is a depth-bounded polynomial (`G` nilpotent).
  - Commit: `S_new = P_a S_pre + (D_a K_a)^T U_a`, one batched matmul over the accepted path. conv1d handled by following parent links in a fused kernel.
  - Verification kernel 3.4-7.7x; transient state 82-99x smaller. Under serial baselines the linear-attention share of verify grows 4% -> 27% as the tree grows 8 -> 64 nodes.
  - End-to-end up to 4.72x over autoregressive and 2.03x over the best tree baseline (batched); at batch 1 on GB10 only 1.11-1.14x over the best speculative baseline. Native MTP drafter, top-k 4, depth 8. About 6.2 kLoC Python/Triton on SGLang. No code.
- **SpecLA** (2607.16673, 2026-07-18) **[P]**: topology-aware chain and tree kernels, compact factors to recover accepted states, confidence pruning, EAGLE-style drafter. 1.70x on GDN-1.3B on an H100.
- **"ReplaySSM"**: cited by TreeWY at 1.12-1.20x chain throughput via deferred writes. Not found on arXiv.
- **[I]** Do the chain-only factor-store first (replaces a ~150 MB state copy per step across 48 GDN layers), then trees, which pay at `a < 0.75` per the table above.

### 6. [real kernel work - the planned MoE project] Expert cache with asynchronous admission, misses computed on CPU

Key design delta from `davetha/vllm-expert-cache` and `r9700-lru-expert-cache` **[P]**: those run two device kernels per MoE layer (manage: mark routed experts, pick victims by argmin over priority excluding in-use slots, rewrite the expert-to-slot table, emit a miss list; gather: copy missing experts host -> slot), then remap routing IDs to slot space and run the stock fused MoE kernel. Misses are fetched **synchronously**, which is right for a vLLM setup with no CPU expert path. Reported there: 2.8x over no cache at 50% of experts resident (83% of fully-resident speed), 1.6x at 12.5%; LFU-with-decay default; PCIe expert traffic 432 -> 86 MB/step on the R9700 stack; wide steps read through.

- **WiSP** (2606.21868, 2026-06-20; code: `nokia-applied-research/WiSP`) **[P]**:
  - RTX 3090 on PCIe 4.0, Qwen3-30B-A3B, vLLM plug-in, byte-identical outputs. LRU over per-layer expert slots via expert-map indirection. 1.58-2.01x over static offload.
  - Prefetching **lowers** single-stream decode 38-60% (cap 8: 5.47 -> 3.37 tok/s; cap 16: 6.94 -> 2.79) and raises TTFT 1.6-1.9x: the bottleneck is PCIe bandwidth, not prediction quality.
  - Only 44-48% of each layer's top-8 experts carry over between adjacent decode steps; ~1.8 GiB streamed per token.
  - Expert/KV VRAM split set by an equimarginal rule (MV-WSA); 1.07-1.19x end-to-end.
  - Transfers misses; does not compute them on CPU (named as future work).
- **FreeToken** (2608.16157, 2026-08-17; code: `FlashML-org/FreeToken`) **[P]**:
  - Misses are split between fetch and in-place CPU compute: fetch `q* ~= m * B_P / B_H`, compute the rest. PCIe DMA and CPU compute share host bandwidth (`B_R = max(B_H - B_P, 0)`).
  - Global LRU across layers with CUDA-graph-compatible device-side control.
  - 1.8x on an RTX 4060 8 GB laptop (39.3 tok/s on Qwen3.6-35B-A3B); miss rate 16-39% against llama.cpp's 62-89%. SGLang/vLLM-style engine, not llama.cpp.
- **Fiddler** (2402.07033), **HybriMoE** (2504.05897, DAC 2025: intra-layer CPU/GPU scheduling, impact-driven prefetch, score-based cache, on KTransformers), **elsa-lab** (2512.16473, ASP-DAC 2026; code: `elsa-lab/MoE-CPU-GPU-Collaborative-Inference`) **[P]**: all compute cache misses on the CPU.
- **[I]** With `B_P` ~12 GB/s and a CPU side that is compute-bound well below the DDR4 ceiling (see addendum), fetch at most a minority of misses and only in the background; never stall on a fetch.
- **[I]** Caveat from the measured curve: a speculative verify batch touches the **union** of experts (**AcceptMoE** 2608.02989, **DraftExpert** 2607.24434, **BigMoMo** 2609.14643 **[P]**). Expect the optimal N to drop on the MoE; re-measure `T_ver(N)`.

### 7. [small patch, after 6] Cache-aware routing

- **Cache-Prior / Mixture of Cache-Conditional Experts** (2412.00099, 2024-11-27, TMLR 06/2025; training-free; built on a modified llama.cpp) **[P]**:
  - `z' = z + lambda * Delta_avg * cache_mask`, used **only for re-ranking**; original logits still weight the expert outputs. `Delta_avg` is a running average of `max(z) - min(z)`.
  - Always keep the true top-J (J=1 for k=2 models, J=2 for granular ones); swapping the top-1 "severely compromises" quality.
  - Halves miss rate for +0.1-3% perplexity (Qwen-MoE +0.5%, DeepSeek-V2-Lite ~0.1%, Mixtral +2.9%) and <0.1% on MMLU/GSM8K. Beats Belady at +1% perplexity. 2x on Snapdragon phones.
  - Evaluated at caches of 1/4-1/2 of experts, **not** 5-10%. LRU eviction with higher-router-weight experts evicted first.
- **AcceptMoE** (2608.02989, 2026-08-04) **[P]**: residency-conditioned expert eligibility during speculative verification. -0.27 pt mean accuracy; 1.29x fully resident and 2.06x under offload on SGLang at batch 1.
- Not lossless; no reasoning-benchmark evidence for Cache-Prior. **[I]** It is the only lever that moves hit rate much at a 5-10% cache.

### 8. [small] Eviction policy - do not over-invest

- **Reproducible Evaluation of MoE Expert Caching** (2608.07911, 2026-08-08; code: `shijiuzhang/moe-cache-eval`) **[P]**:
  - Corrected event-atomic replay on Qwen3-30B-A3B (128 experts, 40% capacity, B=8), miss rates: Belady 9.93%, LFRU 18.01, learned next-use 18.93, static diagnostic 19.10, LFU 19.14, LRU = Least-Stale 19.30.
  - Naive per-access replay inflates recency policies by 27-29%, inverting the ranking.
  - 84-97% of the gap to optimal is eviction foresight; their learned predictor recovered -11.4% of it. Static pinning is 67.4% worse than LFRU at low concurrency.
  - Trace-driven only; no tok/s. "The binding budget is bandwidth, not capacity alone."
- **SpecMD** (2602.03921, 2026-02-03) **[P]**: Least-Stale (stale/current queues, FIFO by layer position) gets 1.6-1.9% collision misses vs LRU's 4.5-12.6% at 5% capacity; 88-92% hit rate; TTFT -10.7 to -34.7% on OLMoE. Metric is TTFT at an emulated ~5 GB/s, and the paper above found Least-Stale coincides with LRU under corrected replay. No code.
- **Cacheable by Design?** (2608.18261, 2026-08-18; code plus the `llama-moe-trace` router-telemetry tool **for llama.cpp**: `Shriniwas410/cacheable-by-design`) **[P]**: on Qwen3-30B, adjacent-token expert reuse is 2.0x chance, 95% of traffic uses 52.5% of experts, and an LRU holding 13.4% of experts serves 66% of requests.
- **SAEM** (2608.21614, 2026-08-21, DAC 2026) **[P]**: CoT reasoning stages have coherent expert sets; 1.33x over caching/offloading baselines (1.54x with matched calibration).
- **Verdict:** LFU with decay or LFRU, per-layer slots. **[I]** Expect ~50-60% hit rate at 5-10% capacity. Run `llama-moe-trace` on the candidate MoE before writing a kernel.

### Below the line: further CPU ternary kernel work [real kernel work]

- **bitnet.cpp** (2502.11880, 2025-02-17; code: `microsoft/BitNet`) Table 7 **[P]**, i7-13700H, tok/s for TL2_0 / TQ2_0 / I2_S: 7B 20.72 / 19.92 / 20.62; 30B 4.99 / 5.25 / 5.70; 100B 1.69 / 1.61 / 1.65. T-MAC runs at 0.5-0.6x of TQ2_0 (7B: 12.29). TL1 = 2 weights per 4-bit index (2 bpw), TL2 = 3 weights per sign + 4-bit index (1.67 bpw). Kernels assume a per-tensor weight scale and int8 per-tensor activations.
- **Vec-LUT** (2512.06443, 2025-12-06, MobiSys 2026; code: `OpenBitSys/vlut.cpp`, llama.cpp-integrated) **[P]**: T-MAC cannot beat TQ2_0 on Intel/AMD AVX2. Vector LUT wins 1.4-3.0x only for **parallel** tokens and beats T-MAC only from N=8; the authors recommend the scalar path for single-token decode.
- **T-MAC** (2407.00088), **FairyFuse** (2604.20913: AVX-512 masked add/sub, 1.24x over Q4_K_M on a Xeon; notes that 16x compression shifts GEMV toward the compute regime on bandwidth-limited CPUs) **[P]**.
- **Answer to question D:** a LUT GEMV for group-128 on AVX2 is at parity with a good int8 `maddubs` kernel, 0-10% at best. Group-128 scales add a per-group multiply that erodes LUT gains further **[I]**. It only helps N=1 passes.

---

## 3. Nothing useful

- **Question F (KV for hybrids with attention in 1 of 4 layers):** nothing specific published. Adjacent only:
  - **DAMP** (2608.27513, 2026-08-27) **[P]**: uniform INT8/FP8 recurrent-state quantization already degrades reasoning on Qwen3.6-35B and Kimi-Linear-48B; INT4/NVFP4 collapses to near zero. Needs mixed precision at 9.9 bits. **Do not quantize GDN state.**
  - **DASC** (2608.30386, 2026-08-31) **[P]**: 2.63x compression of recurrent state checkpoints using per-head retention horizons (KDA).
  - **Tail-Replay** (2608.30310, 2026-08-31) **[P]**: drop recurrent checkpoints and rebuild state by replaying a short suffix. Approximate; relevant to context-checkpoint memory.
  - **2609.04098** (2026-09-03) **[P]**: GDN weights survive NVFP4 W4A4 on Qwen3.8-27B; the delta rule forgets a state impulse within hundreds of steps.
- **Question A, activation sparsity:** nothing for ternary or BitNet-style models without retraining.
  - **Prox** (2607.27591, 2026-07-30) **[P]**: best training-free SwiGLU method (beats TEAL and CATS). Qwen3-14B 77.9 -> 76.6 / 76.1 / 74.8 at 50 / 60 / 70% sparsity; Qwen3-8B 76.1 -> 74.6 / 73.3 / 68.6. 1.51-1.99x on an A6000 only. Needs an INT4 proxy copy (+12% weights). Tested with AWQ, FP8, W4A16, W8A8 but never 2-bit or ternary. No CPU or offload results. No code.
  - **SVD contextual sparsity predictors** (2603.14110): ReGLU models only.

---

## 4. Rejected

| Technique | Why it does not transfer |
|---|---|
| Expert prefetch predictors: SeqMoE (2609.12978), SPICE (2608.21240), APEX (2608.11688), SpecPrefetch (2607.24787) | WiSP shows prefetch hurts single-stream decode even on PCIe 4.0; most need trained predictors. |
| Trained-locality routers: StickyMoE (2607.08780); 2608.18261 | Needs pretraining; pre-registered negative result, every config failed the <=1% perplexity gate and the tax does not shrink with scale. |
| Cache-Aware Joint Router Adaptation (2609.04895), Edge0 prerouter + recovery LoRA (2609.18063), DraftExpert (2607.24434) | Need training. |
| Mixed-precision expert caches: HOBBIT (2411.01433), SliceMoE (2512.12990), FloE (2505.05950) | Premise is a cheaper low-bit copy; ternary experts have none. |
| T-MAC / TL1 / TL2 LUT port | Parity or worse on AVX2 per two independent papers. |
| FlexEE (2609.17008), CATS-2026 (2605.11186) | Need exit-layer / cascade training. |
| Cassandra (2605.26558) | Needs a hardware codec. |
| PTD (2607.10661) | Adds positions per pass with no drafter-quality gain. |
| Recurrent-state quantization | DAMP shows reasoning degrades at INT8. |
| Litespark (2605.06485) | Compared against PyTorch only. |
| Unified LUT attention with signed-digit KV (2608.03229) | Accelerator-oriented. |

---

## 5. Addendum: measurements taken after the report

### (1) MTP draft-length sweep on the real server **[M]**

| Config | -ngl | tok/s | Acceptance |
|---|---|---|---|
| no speculation | 29 | 3.2 | - |
| MTP n=2 | 24 | 4.8 | 0.75 code / 0.71 reasoning |
| MTP n=4 | 22 | 4.7 | 0.50 |
| MTP n=15 | 18 | 3.1 | 0.18 |

The single recursive MTP head's acceptance collapses with depth. This invalidates the constant-`a` assumption behind the expected-tok/s table for this drafter, and confirms **item 4** (a real draft model) as the way to exploit cheap long verification. **[I]** Note the sweep is confounded: `-ngl` fell from 29 to 18 as n grew, so deeper configs also streamed more bytes per pass; the n=15 number understates what a drafter holding `a ~0.7-0.8` at depth would achieve at a fixed `-ngl`.

### (2) Telemetry **[M]**

- No speculation: CPU 580% of 1200%, DRAM read 12.9 of ~38 GB/s, GPU 13%. The CPU side is **compute-bound, not bandwidth-bound**.
- MTP n=2: GPU 88%, PCIe -> GPU average 8.8 / peak 10.8 of ~13 GB/s. **PCIe-bound.**

**[I]** Implications:
- Speculative passes already run the link at about 68% average utilization, so pinned buffers and copy/compute overlap (item 3b) can recover at most about 1.2-1.5x on the transfer portion. **Byte reduction** (3a: more resident layers via KV quantization; 3c: ~1.69 bpw wire packing) is the larger lever.
- **[M]** `-ctk q8_0` at 4K context changed nothing (4.72 / 4.42 vs 4.79 / 4.61 tok/s; it saves 64 MiB, less than one layer). Item 3a only applies at long context (~2 GB of KV at 32K); at 4K it is a no-op.
- **[M]** Enabling MTP costs about 5 resident layers, including 187 MiB of recurrent-state rollback snapshots. **[I]** That is the memory item 5's chain-only factor-store would reclaim: roughly 2 resident layers' worth, i.e. fewer streamed bytes on every pass, on top of removing the per-step state copy.
- The CPU path being compute-bound at about one-third of DDR4 bandwidth with only ~5.8 of 12 threads busy points at threading and kernel efficiency rather than LUTs for the N=1 path, consistent with the bitnet.cpp and FairyFuse observations. It remains off the critical path once speculation is on.
