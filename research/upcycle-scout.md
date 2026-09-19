# Qwen3.8-27B → small-active MoE: scout report

Date: 2026-09-19. Scope: web / HF API / GitHub API / arXiv only. No downloads, no GPU boxes touched.
Tags: **VERIFIED** = primary source with URL/figure. **INFERRED** = my arithmetic or judgment on top of verified inputs. Community model-card numbers are VERIFIED as *claims on the card*, not independently reproduced.

## TL;DR

1. **No 3-4B-active MoE derivative of Qwen3.8-27B exists.** The only MoE-ified Qwen3.8-27B on HF is logic65's Whittle-MoE-27B-**A17.8B** (2/3 of the weights still active per token). Everything else named "Qwen3.8-…-A3B" is a Qwen3.6-35B-A3B body with Qwen3.8 distillation on top.
2. **A 3-4B-active model cannot be carved out of Qwen3.8-27B's weights by FFN-splitting at all.** From the config, the non-FFN, always-active part (attention + GDN + lm_head) is ~9-10B params. FFN-only MoE-ification bottoms out around 9-10B active. Reaching 3-4B active requires depth/width-pruning the attention stack too, which is a rebuild, not an upcycle.
3. REAP does not apply to a dense model (needs routers and experts to score). REAP on Flash-Next is real and well-documented at 50-62% kept experts, but a ≤13 GB RAM-resident build for the 1650S box needs ~75-81% pruning, far outside REAP's validated 50%, and the result is still ~6B active (about half the tok/s of an A3B).
4. Real dense→MoE conversions that hold quality cost 200B-1T tokens on ~100 A100s. Cheap variants (split + light SFT/LoRA) measurably lose 25-45 benchmark points on knowledge/instruction tasks. A documented attempt on a Qwen3.5 SwiGLU/DeltaNet hybrid failed outright.
5. **Recommendation: (iv) with one cheap experiment.** Keep Qwen3.6-35B-A3B. Try `empero-ai/Qwen3.8-35B-A3B-Distill` GGUF (same `qwen35moe` arch, drop-in). If any training job is ever worth the friend's 2×MI210, it is *distilling Qwen3.8-27B into Qwen3.6-35B-A3B* (Whittle-style KD, hours not weeks), not MoE-ifying the 27B.

---

## 0. Ground truth on the source models

| Item | Value | Tag / source |
|---|---|---|
| Qwen3.8-27B arch | `model_type: qwen3_5`, 64 layers, hidden 5120, `intermediate_size` 17408, 3 linear_attention : 1 full_attention (`full_attention_interval: 4`), vocab 248320, untied embeddings, `hidden_act: silu` (SwiGLU) | VERIFIED — https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json |
| Qwen3.8-Flash-Next arch | `model_type: qwen4_exp`, 48 layers, hidden 2560, 512 experts, 10 routed + 1 shared per token, `moe_intermediate_size` 640, vocab 248320, n-gram table `ngram_vocab_size_base` 20,000,000, MTP 1 layer | VERIFIED — https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json |
| Flash-Next params | "125B" main + "51B n-gram embedding" + "4B MTP"; "6B" active; license `qwen-community-1.0` | VERIFIED — https://huggingface.co/Qwen/Qwen3.8-Flash-Next (card body; sidebar says 180B) |
| Qwen3.8 family | 27B dense, Flash-Next, 2.4T-A95B, hosted Max. No mid-size MoE (the 3.6-era 35B-A3B slot is empty). | VERIFIED — https://qwen.ai/blog?id=qwen3.8-flash-next (via search snippet; the blog is JS-rendered and did not fetch directly) |
| Qwen staff on a small Qwen3.8 MoE | No org-member reply in either request thread. OP of #120 now says probability is "practically zero". | VERIFIED — https://huggingface.co/Qwen/Qwen3.8-27B/discussions/114 , https://huggingface.co/Qwen/Qwen3.8-27B/discussions/120 |
| llama.cpp mainline arch strings | `qwen35` (27B), `qwen35moe`, `qwen4exp` (Flash-Next), plus qwen3next/qwen3moe/etc. | VERIFIED — https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp |
| `qwen4exp` merge | PR #27742 merged 2026-08-27. N-gram table is "1 large table — it can be offloaded to RAM or Disk via mmap"; lookup is a host-side hash + "plain row gather"; "no imatrix ever covers it" | VERIFIED — https://github.com/ggml-org/llama.cpp/pull/27742 |

**The active-parameter floor of Qwen3.8-27B (INFERRED from VERIFIED config):**
- FFN per layer = 3 × 5120 × 17408 = 267M; × 64 = **17.1B** in FFN.
- Embedding + lm_head (untied) = 2 × 248320 × 5120 = **2.5B**.
- Remainder = 27 − 17.1 − 2.5 ≈ **7.4B** in attention/GDN/norms — read every token.
- lm_head (1.27B) is also read every token. So the always-active floor before any FFN is ≈ **8.7B**.
- Cross-check: Whittle-MoE routes 8192 of 17408 FFN width → 8.05B FFN active + 8.7B floor ≈ 16.8B; the card reports 17.8B active. Consistent.
- Consequence: *no* FFN-splitting scheme reaches 3-4B active from these weights. At 3B active you'd need to delete most of the attention/GDN stack as well.

## 1. Does it already exist? (searched HF API + GitHub)

HF search `Qwen3.8-27B` (100 results, sorted by downloads) contains exactly one MoE lineage: `logic65/Qwen3.8-Whittle-MoE-27B-A17.8B` (+ GGUF repo and four mirrors). Zero hits tagged REAP/upcycle/pruned on the 27B. VERIFIED — https://huggingface.co/api/models?search=Qwen3.8-27B&limit=100&sort=downloads . A second pass `search=Qwen3.8&filter=moe` (200 results) adds nothing else derived from the 27B's weights. VERIFIED — https://huggingface.co/api/models?search=Qwen3.8&filter=moe&limit=200&sort=downloads

### 1a. logic65/Qwen3.8-Whittle-MoE-27B-A17.8B — the only true derivative
- Method: partition, not rebuild. Each layer's 17408-wide FFN cut into 64 routed slivers of width 192 + one always-on shared expert of width 5120 (64×192 + 5120 = 17408, "zero new FFN weights"). Router picks 16 of 64 → 8192 of 17408 FFN width per token. Attention side untouched. **27B total / 17.8B active.** VERIFIED — https://huggingface.co/logic65/Qwen3.8-Whittle-MoE-27B-A17.8B
- Training: routers trained "with every expert frozen"; logit distillation from Qwen3.8-27B on "245 complete answers" (v2) + a stop-gate of 0.33M params (v2.2); SFT mixes (ultrachat_200k, tulu-3, CodeFeedback). Compute: "personal hardware and paid Colab time", collection says "whittled on two 8GB GPUs". Total tokens not stated. VERIFIED (card).
- Quality (self-reported, own harness): knowledge battery 27-28/39; structured asks fail "~1 in 5"; single-turn loop rate 8% (v2.1); "It is not the parent." VERIFIED (card).
- Arch: HF `qwen3_5_moe` → GGUF **`qwen35moe`** (mainline, "No fork, no patches"). GGUF sizes (v2.2.1 table): Q3_K_M 13.9 GB, DQ3_K_XL 14.7 GB, Q4_K_M 17.4 GB, Q8_0 28.7 GB. GGUFs still carry the v2.2 gate ("keep reasoning off on quantized tiers"). VERIFIED — https://huggingface.co/logic65/Qwen3.8-Whittle-MoE-27B-A17.8B-GGUF
- Fit for our box (INFERRED): 13.9 GB file fits RAM+NVMe but bytes/token = 17.8/27 = 66% of the dense read → ~1.5× the Bonsai's 5 tok/s ≈ 7-8 tok/s. Not a 40 tok/s path. Skip.

### 1b. Things named "Qwen3.8 … A3B" that are NOT Qwen3.8-27B weights
| Repo | What it actually is | Tag |
|---|---|---|
| `logic65/Whittle-Qwen-3.8-35B-A3B` (+`-GGUF`) | Body = Qwen3.6-35B-A3B (25.1B) + 10.0B hashed n-gram memory transplanted from Flash-Next's table. Qwen3.8-27B is only the **KD teacher**: 1,840 thinking traces, 2,861 steps, "3.3 h on one 96 GB Blackwell". Arch `qwen4_exp`/`qwen4exp`, "stock llama.cpp, no patches". GGUF Q3_K_M 16.7 GB … Q8_0 37.8 GB; ~10 GB of that is the table, kept in RAM via `-ot per_layer_token_embd=CPU`; "runs at 3 B-active speed". Self-reported GSM8K 44/50; "Research preview, not a finished distillation"; greedy decoding loops. 0 downloads. | VERIFIED — https://huggingface.co/logic65/Whittle-Qwen-3.8-35B-A3B , https://huggingface.co/logic65/Whittle-Qwen-3.8-35B-A3B-GGUF |
| `logic65/Whittle-Next-27B-A3B` | Same lineage, earlier (2B-row table, 27.1B total). GSM8K 86.5% (173/200) vs its own 26B predecessor 87.0%; long-context 26k-75k 4/6 vs parent Qwen3.6-35B-A3B 6/6. | VERIFIED — https://huggingface.co/logic65/Whittle-Next-27B-A3B |
| `empero-ai/Qwen3.8-35B-A3B-Distill` (+ mradermacher / MrFuzzihead GGUFs) | Qwen3.6-35B-A3B SFT-distilled from **Qwen3.8-2.4T-A95B and Flash-Next** (not the 27B). Arch `qwen3_5_moe`. lm-eval bf16 zero-shot: MMLU 0.834 vs base 0.838 (noise); ARC-C 0.582 vs 0.548; ARC-E 0.830 vs 0.819. Trained on 8,192-token examples → "noticeably shorter outputs"; vision untouched. Dataset size / hardware not stated. | VERIFIED — https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill |
| `logic65/Qwen3.8-Whittle-Next-19.8B-A10B-base` | Qwen3.8-27B → layer-merge to 14.7B → "training-free MoE carve" (240 experts, k=58 + 2048 shared; 14.76B/9.03B active) + hyper-connections + 4B n-gram. wikitext PPL 86.62 (carve alone) → 33.18 with table, but on unseen science text 27.53 vs 27.96 (author: "mostly memorisation"). Needs a one-line silu-gate patch (not mainline). "RESEARCH ARTIFACT — NOT AN ASSISTANT". | VERIFIED — https://huggingface.co/logic65/Qwen3.8-Whittle-Next-19.8B-A10B-base |

Net: nothing shippable exists at 3-4B active from the 27B's weights, and the one person actively trying (logic65) has pivoted to *distilling the 27B into the 3.6-A3B body* — which is the empirically sane move (see §5).

## 2. REAP / expert pruning

**Does REAP apply to dense Qwen3.8-27B? No.** REAP = "Router-weighted Expert Activation Pruning": the criterion "considers both router gate-values and expert activation norms", applied to "an existing SMoE model". A dense model has no router and no experts to score. VERIFIED — https://arxiv.org/abs/2510.13999 . Confirmed by construction: the 27B config has no `num_experts` field (§0).

**REAP paper facts** (VERIFIED — https://arxiv.org/html/2510.13999):
- One-shot, "no additional fine-tuning after compression".
- Calibration: ≤110B models "1,024 samples … packed to 2,048 sequence length"; ≥110B "12,228 samples with a maximum sequence length of 16,384".
- Ratios evaluated: 25% and 50% of experts pruned. Nothing beyond 50% is evaluated.
- Quality: coding (non-giant models) mean −1.9% / −6.9% at 25% / 50%; math −0.1% at 25%; multiple-choice ≈ −4% / −13%; Qwen3-Coder-480B & Kimi-K2 −0.16% / −1.2%; tool-use at 50% −5.9%.
- Official repo: https://github.com/CerebrasResearch/reap — HF adapters via `MODEL_ATTRS` in `src/reap/model_util.py`; "memory-efficient layer-wise (block-wise) calibration observer for pruning large models on a single GPU" (2026-03-30). `qwen4_exp` is not in the listed supported set; no ROCm statement. VERIFIED (README).

**Flash-Next REAP GGUFs that exist today** (all `qwen4exp`, mainline ≥ 2026-08-27):

| Repo | Experts kept | Total stored | Active | File | Quality claim | Tag |
|---|---|---|---|---|---|---|
| `AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF` | 320/512 (62.5%) | "132B" | unchanged (10+1 experts) | UD-Q3_K_XL 68.96 GB; UD-Q2_K_XL 61.5 GB | HumanEval 96.3% (158/164) vs 95.7% unpruned UD-IQ1_M; fabrication probe 25% vs 0% | VERIFIED — https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF |
| `AnonimousA/Qwen3.8-Flash-Next-REAP-256-duo-GGUF` | 256/512 (50%) | "117B" | "unchanged — still top-10 experts + 1 shared" | UD-Q3_K_XL 57.69 GiB (61.9 GB) | HumanEval 89.6% vs REAP-320 96.3%; routing mass retained 89.29% (L0 79.6%); RTX 5090 `--n-cpu-moe 8` 68.4 tok/s | VERIFIED — https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-256-duo-GGUF |
| `sh0wie/Qwen3.8-Flash-Next-REAP-288-GGUF` | 288/512 | "124B" | n/s | Q4_K_M 78-83.8 GB, Q8_0 116-124 GB | HumanEval 91.5% vs 93.9% unpruned (MLX 4-bit lineage); ~686K-token calibration | VERIFIED — https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-GGUF |
| `talhasarac/Qwen3.8-Flash-Next-REAP128-86B-Unity-OpenCode-GGUF` | 128/512 (25%) | "~86.3B" (≈35B LM + n-gram) | "~6B" | UD-IQ4_XS 45.50 GiB | "No formal benchmarks"; knowledge/reasoning "may be substantially worse"; calibration = 50 prompts / 5,120 tokens | VERIFIED — https://huggingface.co/talhasarac/Qwen3.8-Flash-Next-REAP128-86B-Unity-OpenCode-GGUF |

How they were made (VERIFIED, REAP-320/256 cards): salience via `llama-imatrix` (525K-token corpus, 30% agentic / 30% code / 15% chat / 12.5% math / 12.5% writing); then "pruned by binary copy from Unsloth's quant — surviving experts keep their original bits, no requantization"; manifest `manifests/seleccion_mass_K320.json`. `expert_count` is one global GGUF scalar, so K is uniform across layers. Rig: "RTX 5090 32 GB", 95 GB RAM, Windows 11 — i.e. this is a **llama.cpp-only, no-PyTorch surgery** that a 32 GB GPU + ~100 GB RAM box can do. The exact slicing script is not published on either card (INFERRED: a short gguf-py script; the expert dim is the outermost tensor dim so each expert is a contiguous block-aligned span, per the sh0wie bf16 card).

The n-gram/PLE table: "26.8 GiB per-layer-embedding table (`per_layer_token_embd.weight`, 320M rows × 90 B)", "untouched by expert pruning and stays on the host", read via mmap "one page fault per ~90-byte row, on every token"; `--lazy-mode off` pins it for "+26.8 GiB RAM". VERIFIED (REAP-320 card). 90 B / row = Q4_0 block layout for 160 elements (INFERRED).

**Can a REAP'd Flash-Next fit the 1650S box (≤~13 GB RAM-resident, 16 GB RAM, 88 GB NVMe)?** INFERRED from VERIFIED config:
- One expert = 3 × 2560 × 640 = 4.92M params. 48 layers × 512 = 120.8B expert params (matches "125B main" minus ~4.2B of attention/GDN/shared/embeddings/lm_head).
- Expert bytes per kept expert ≈ 1.6 MB at Q2_K_XL-class (~2.6 bpw) / ≈ 2.15 MB at Q3_K_XL-class (~3.5 bpw).
- RAM-resident hot set = K × 48 × (per-expert) + non-expert ~4.2B at ~5-6 bpw (≈ 2.6-3.2 GB):
  - K=256: 19.7 GB (+3) — no.
  - K=128: 9.8 GB (+3) ≈ 12.8 GB at Q2-class — borderline; 13.2 (+3) at Q3-class — no.
  - K=96: 7.4 GB (+3) ≈ 10.4 GB — fits at Q2-class.
- The 26.8 GiB table lives on NVMe (mmap, a handful of 4 KiB page reads per token) → fine for 88 GB free, negligible bandwidth.
- So the box needs **K ≈ 96-128 of 512 (75-81% pruned)**, 1.5-1.6× beyond REAP's validated 50%, where coding is already −6.9% mean and MC −13%. The one existing 128-expert build has no benchmarks and its author expects it "substantially worse". Also the router expects softmax over 512 and the paper never re-normalizes beyond 50%.
- Speed: active stays ~6B (10+1 experts + ~4.2B dense incl. a 1.27B lm_head) — roughly 2× the bytes/token of a 3B-active model at equal bpw → INFERRED ~20-25 tok/s where Qwen3.6-35B-A3B does 46. And the QSA indexer "re-pools the whole cache every step" so decode "slows linearly with context" (llama.cpp #28734/#28012, cited in PR #27742). VERIFIED for the issue refs, INFERRED for the tok/s.

Verdict on §2: REAP-pruning Flash-Next is a proven "inference surgery" path at 50-62% kept (61-69 GB files, needs a 64-128 GB box), and the friend's box can do it in pure llama.cpp tooling. It is **not** a path to a ≤13 GB / ~40 tok/s model for the 1650S.

## 3. Sparse upcycling / FFN-splitting: what training actually costs

### 3a. Copy-FFN-into-N-experts + continue pretraining (expensive)
- **Sparse Upcycling (Komatsuzaki et al., ICLR 2023)**: experts are "identical copies of the original MLP", router random-init, "replace half of the MLP layers", 32 experts. Gains appear when the extra budget is "+10% to +60% of the cost to train the original (dense) network"; T5-Large/Base beat dense continuation by "1.5-2 absolute points on SuperGLUE using 46% and 55% extra training"; "A central challenge in model upcycling is overcoming the initial performance decrease". From-scratch MoE catches up "after about 120% of the original dense checkpoint computation budget". VERIFIED — https://arxiv.org/pdf/2212.05055 (pp. 1-8). For a 27B trained on an unpublished (INFERRED: tens of T) token count, "10-60%" is trillions of tokens.
- **NVIDIA "Upcycling LLMs into MoE"**: Nemotron-4 15B upcycled on **1T tokens**; upcycled 67.6% MMLU vs 65.3% for dense continued on the same tokens. VERIFIED — https://arxiv.org/abs/2410.07524
- **Drop-Upcycling** (re-init r=0.5 of FFN dims to force specialization): MoE trained on **500B tokens**; 8×3.7B (5.9B active / 18B total) scores 44.4 avg vs 13B dense 44.5 (805B tokens); "naïve Upcycling exhibited the slowest loss reduction". VERIFIED — https://arxiv.org/html/2502.19261
- **Expert Upcycling (2026)**: sparse (dense→MoE) upcycling at low activation ratios "fails to match even the Fixed-E baseline in every setting"; loss gap vs the MoE→MoE method widens "from 0.026 at 25% to 0.241 at 3.13%" activation. VERIFIED — https://arxiv.org/html/2604.19835v1 . Our target (3-4B of 27B ≈ 11-15% activation) sits in the bad end of that curve. (INFERRED)

### 3b. Split existing FFN neurons into experts + light training (cheap, lossy)
- **LLaMA-MoE v1**: LLaMA-2-7B FFN split into 16 experts top-4 = 3.5B active; continual pretraining **200B tokens on 112 A100-80G** (global batch 15M tokens). Avg 57.7 — beats Sheared-LLaMA-2.7B (56.4) but "converge[s] to higher training loss than LLaMA-2 7B". VERIFIED — https://arxiv.org/html/2406.16554
- **LLaMA-MoE v2**: LLaMA-3-8B → 8.0B total / **3.8B active** (8 experts top-2 or 1+7 top-1), **7B tokens** of two-stage SFT on 32 A100s. Deltas vs LLaMA-3-8B: MMLU 67.22 → 37.41 / 40.89; GSM8K 76.50 → 53.07 / 55.04; HumanEval 71.38 → 53.53 / 51.21; IFEval 76.53 → 32.72 / 36.04; HellaSwag 78.79 → 58.95 / 53.67. VERIFIED — https://arxiv.org/html/2411.15708 . That is the closest published analog to "split a modern dense LLM to ~half active with a light SFT" and it loses 20-45 points.
- **CMoE (training-free split + 1 h LoRA on 2,048 samples)**: Llama-2-7B WikiText PPL 5.27 → 62.30 (25% activation, training-free) → 12.73 (fine-tuned); at 75% activation 7.02 → 5.69. Llama-3-8B: 6.14 → 143.38 → 21.01 at 25%. Downstream at 25%: BoolQ 55.04 vs 82.04, PIQA 57.12 vs 78.78. Conversion "under five minutes". VERIFIED — https://arxiv.org/html/2502.04416v2
- **MoEfication (2021)**: ">95% original performance" using "10% to 30% of FFN parameters" — on ReLU-era models. VERIFIED — https://arxiv.org/abs/2110.01786 . Not transferable to SwiGLU (next bullet).
- **Documented failure on the Qwen3.5 family**: CMoE + D2DMoE applied to Qwen3.5-9B (SwiGLU, DeltaNet hybrid — same family as Qwen3.8-27B). Structural conversion succeeded on all 32 layers; router-only and expert+router fine-tuning both failed to recover ("fluent but no factual knowledge retained"); L1 sparsification plateaued at 23.4%. Author's conclusion: "Post-hoc MoEfication of SwiGLU models without prior sparsification does not work"; "The same issues would apply to any SwiGLU model"; ReLU-fication (ProSparse) "required 134B tokens for a 13B model". Total spend ~24.5 h / ~$13. VERIFIED — https://github.com/iamboosted/Qwen3.5-9B-Dense-To-Moe/blob/main/README.md
- **Whittle-MoE-27B-A17.8B** (§1a) is the in-family datapoint: router-only training on tiny hardware, 47% FFN activation, and it already shows knowledge regressions and ~1-in-5 structured-output failures.

### 3c. Does anything turn a ~27-30B dense into a 3-4B-active MoE?
No published method does. Every dense→MoE conversion above keeps total params ≈ the dense total and lands at 40-50% activation (LLaMA-MoE 3.5B/7B, v2 3.8B/8B, Drop-Upcycling 5.9B/18B). For Qwen3.8-27B specifically, the ~9B always-active floor (§0) makes 3-4B active unreachable by FFN work alone. VERIFIED (no hits) + INFERRED (arithmetic).

**Token budget summary:** quality-preserving dense→MoE = 200B-1T tokens (VERIFIED: LLaMA-MoE 200B, Drop-Upcycling 500B, NVIDIA 1T; Sparse Upcycling 10-60% of original pretraining). "Split + light SFT/LoRA" = 7B tokens or 2k samples, with 20-45-point losses (VERIFIED: LLaMA-MoE v2, CMoE).

## 4. What the friend's box can actually run

Box: 2× MI210 (gfx90a, 64 GB) + 2× R9700 (gfx1201, 32 GB), +2 R9700 soon. 192 → 256 GB total, mixed CDNA2 + RDNA4. Host RAM unknown.

**Memory math (VERIFIED formulae, INFERRED application):**
- Mixed-precision Adam: fp16 weights 2Ψ + fp16 grads 2Ψ + fp32 master/m/v 12Ψ = **16Ψ bytes** (K=12). ZeRO-1 → ≈4Ψ+12Ψ/N, ZeRO-2 → ≈2Ψ+14Ψ/N, ZeRO-3 → 16Ψ/N. VERIFIED — https://arxiv.org/html/1910.02054v3
- Full FT of 27B: 16 × 27e9 = **432 GB** of model state before activations. ZeRO-3 across the two MI210s = 216 GB each (no); across 4 devices the R9700s' 32 GB cap it (no); ZeRO-Offload needs ≥432 GB host RAM (unknown, unlikely). **Full FT does not fit on 192 or 256 GB.** INFERRED.
- Frozen bf16 27B = 54 GB. Fits one MI210 with ~10 GB headroom, or splits over two. QLoRA: "finetune a 65B parameter model on a single 48GB GPU" → a 27B NF4 base is ~14 GB + adapters/optimizer. VERIFIED — https://arxiv.org/abs/2305.14314
- Router-only / gate training on a split model needs one frozen 27B forward (54 GB bf16) plus, for KD, a teacher forward (another 54 GB) = exactly the 2×MI210 pair. Whittle's KD run into the 3.6-A3B body was "3.3 h on one 96 GB Blackwell" (VERIFIED, §1b) → fits 2×MI210 (128 GB). INFERRED.
- Continued pretraining at 200B+ tokens (§3) is ~1,000-10,000× more compute than that KD run; not a 2-GPU job. INFERRED.

**Mixed-gfx reality (VERIFIED):**
- RCCL maintainer (huanrwan-amd, 2026-01-12) on mixing two RDNA4 cards: "GPUs from the same generation/architecture … generally work better together" and "We do not have such a setup to verify the correctness". No official support statement either way. https://api.github.com/repos/ROCm/rccl/issues/2138/comments
- Open RCCL #2189 (2026-08-31, ROCm 10.0 / 7.14, RCCL 2.30.4): even **gfx1200 + gfx1201** (both RDNA4) abort on the first `ncclAllReduce` with "invalid device function"; only works by spoofing both to one arch via `HSA_OVERRIDE_GFX_VERSION=12.0.1`. https://api.github.com/repos/ROCm/rccl/issues/2189 . That spoof cannot bridge gfx90a↔gfx1201 (different ISAs, CDNA vs RDNA). INFERRED.
- Homogeneous dual-R9700 TP=2 deadlocks in vLLM (issue #40980 open since 2026-04-27, 26 comments). Root cause per AMD (tcgu-amd, 2026-08-20): RCCL LL protocol "never received the gfx12 memory-ordering fixes"; workaround `NCCL_PROTO=Simple`; patch rccl PR #2187, merge not confirmed. https://api.github.com/repos/vllm-project/vllm/issues/40980 , https://api.github.com/repos/ROCm/rocm-systems/issues/5480/comments
- Build side: ROCm 7.2.x PyTorch supports gfx1201; multi-target builds are `PYTORCH_ROCM_ARCH="gfx90a;gfx1201"`. Frameworks (DeepSpeed/FSDP/axolotl/torchtune) assume homogeneous devices; no source shows a working mixed-CDNA/RDNA collective. VERIFIED (docs) + INFERRED (no counter-evidence found).
- llama.cpp HIP: `GPU_TARGETS` omitted "will build the code for all GPUs in the current system"; multi-GPU layer split does not use RCCL unless `-DGGML_HIP_RCCL=ON` (a 2026-09-13 comment on #5480 confirms standard builds have it off). So **inference-side surgery (llama-imatrix + binary-copy REAP) can use all four cards via plain layer split**, while **any torch.distributed training job should be pinned to the 2×MI210 pair only** (128 GB, homogeneous gfx90a). VERIFIED (docs/comment) + INFERRED (recommendation).

**Flag:** treat "192/256 GB" as **128 GB (training) + 64-128 GB (separate inference/eval group)**, not a pool.

## 5. Bottom line, ranked for us

Baseline to beat: Qwen3.6-35B-A3B IQ2_M at 46 tok/s, quality ≥ Bonsai-27B.

1. **(iv) Don't MoE-ify Qwen3.8-27B. Recommended.** Three independent kills: (a) the always-active floor is ~9B, so 3-4B active is arithmetically unreachable by FFN splitting; (b) the only method class cheap enough for a 2×MI210 box (split + light SFT/LoRA) loses 20-45 points on the closest published analogs and failed outright on a Qwen3.5 SwiGLU/DeltaNet model; (c) the method class that keeps quality costs 200B-1T tokens on ~100 A100s.
2. **(i)-lite: try the drop-in Qwen3.8-flavored A3B GGUFs (zero effort).** `empero-ai/Qwen3.8-35B-A3B-Distill` (+ `mradermacher/…-i1-GGUF`, `MrFuzzihead/…-APEX-GGUF`) is `qwen35moe`, identical arch to the daily driver, same expert-cache/MTP path, MMLU flat vs base, ARC up. Worth one eval run. `logic65/Whittle-Qwen-3.8-35B-A3B` Q3_K_M (16.7 GB, of which ~10 GB is a row-gather table you can mmap from NVMe) is the other candidate, but it is a self-declared unfinished research preview.
3. **(v) If a training job on the friend's box is ever justified: distill Qwen3.8-27B into Qwen3.6-35B-A3B** (LoRA KD, teacher + student bf16 = ~108 GB → 2×MI210). That is what logic65 converged on after failing the direct route, it took hours not weeks on 96 GB, and it inherits the A3B speed we already have. Same vocab (248320) on both sides (VERIFIED for 27B; INFERRED for 3.6-A3B from the logit-KD working).
4. **(ii) REAP Flash-Next: valid surgery, wrong target.** Proven at 50-62% kept (61-69 GB files, HumanEval within 1-4 pts). For the 1650S it would need 75-81% pruning (unvalidated, one existing build with no benchmarks) and still lands at ~6B active ≈ half the A3B's tok/s. Only worth doing if a ≥64 GB inference box enters the picture.
5. **(iii) FFN-split upcycle of the 27B: no.** See #1.

## Open items I could not verify
- Qwen3.8-27B pretraining token count (not published) — needed to price "10-60% of original compute".
- Friend's box host RAM and PCIe/XGMI topology (decides ZeRO-Offload and MI210 pair bandwidth).
- Exact quant type of the Flash-Next PLE table in Unsloth's GGUFs (90 B/row is consistent with Q4_0; not stated on any card).
- Whether RCCL PR #2187 (gfx12 LL fix) has shipped in a ROCm release.
