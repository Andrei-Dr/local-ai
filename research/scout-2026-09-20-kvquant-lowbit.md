# Scout digest 2026-09-20: TurboQuant / rotated KV, and 1-2 bit weights (3 Sonnet scouts; claims are THEIRS unless marked [V])

[V] = verified by us on the box or in our tree. Everything else: scout-reported with a source, NOT independently checked. Verify
before building on it. Operator rule in force: accuracy first (SPEC 8.x).

## A. Rotated KV cache
- [V] Our tree = mainline b11056 (2026-09-19) + 6 patches. `attn-rot` (ggml-org/llama.cpp PR #21038: Hadamard rotation of Q/K/V
  before caching, attention in rotated space) is IN our build and ACTIVE: ctx1 server log `attn_rot_k = 1, attn_rot_v = 1`.
  So every q4_0 / q8_0 KV number we have already includes the rotation = the part of TurboQuant that carries the value.
- [V] Our build has `GGML_CUDA_FA_ALL_QUANTS=ON` (CMakeCache) => mixed K/V cache types stay on the fused CUDA FA path.
- Scouts: QJL residual (the paper's 2nd stage) is harmful at head_dim 128-256, every serious port dropped it; TurboQuant_mse is a
  special case of EDEN (2021) (arXiv 2604.18555); RaBitQ priority dispute (arXiv 2604.19528). Mainline REJECTED the tbq3_0/tbq4_0
  types (PR #21089, closed 2026-06-02): tbq4_0 ~= q4_0 quality (KLD 0.0096 vs 0.0091) at -47% CPU gen speed.
- Sub-4-bit KV is out for us: vLLM study (2026-05-11) reports up to 20-pt drops on AIME25 / LiveCodeBench at 3-bit and ~30% on
  256k retrieval; a reproduction saw collapse at 2.5-bit. FP8 / q8-class is their recommended default.
- K is more fragile than V (softmax amplifies K noise). llama.cpp Discussion #23470 (Qwen3.8-27B, 65k): q8_0/q8_0 KLD 0.0064,
  q8_0/q5_1 0.0076 (recommended), q8_0/q4_0 0.0123. Qwen3.6-27B at q4_0 K+V: 2 of 500 answers changed (robust); Qwen2.5-7B collapses.
  Anbeeld (Qwen3.6-27B dense, 64k, PPL): q4_0 KV < 0.01 PPL delta. Nothing measured on a GDN hybrid at >= 128k => lq1 is still new data.
- Recurrent (GDN) state is not covered by any of this (62.8 MiB stays).
- Leads, unverified: TheTom/llama-cpp-turboquant (turbo3/turbo4 KV, sm_75 in build targets); AtomicBot-ai fork (Qwen3.6-35B-A3B +
  NextN/MTP + turbo3 KV, claims 1M ctx); a PR-thread claim of "Qwen at full context on a 4 GB GTX 1650 at 50 t/s".
- => lq1 arms: f16 | q8_0 | q8_0-K + q5_1-V | q4_0 | q4_0 with LLAMA_ATTN_ROT_DISABLE=1 (what the rotation buys on OUR model at depth).

## B. 1-2 bit weights
- Pattern reported by several sources: post-training ~2-bit looks fine on knowledge sets and collapses on hard reasoning / code
  (vendor example: IQ2_XXS 27B = 88.9 MMLU-Redux but 57.5 AIME / 56.4 LiveCodeBench; CAT-Q authors concede math/code loss).
  OUR GATE (GSM8K / HumanEval / MMLU-Pro, thinking off, GSM8K saturated at 96-100%) CANNOT SEE THIS. Need a hard published set,
  thinking on, paired (AIME 24/25, LiveCodeBench, GPQA) + a non-2-bit reference (Dave's box, or local Q4/Q5 after a RAM upgrade).
- Ternary-Bonsai-2-27B (PrismML, 2026-09): QAT ternary of Qwen3.8-27B; PQ2_0 = 2.13 bpw packing, PTQ1_0 = 1.75 bpw; needs their
  llama.cpp fork. Vendor claims: 95% of full precision over 15 benchmarks, AIME 87+, within 0.4 pt of UD-Q4_K_XL at 1/3 the size.
  We have the abliterated PQ2_0 file (cold storage, symlinked); S3 measured 5.15 tok/s and DROPPED the quality run. Under the
  accuracy-first rule that verdict needs a redo: quality run + a real speed attempt (PTQ1_0 packing, more layers on GPU, its MTP head).
- Unsloth Dynamic 3.0 (2026-08): per-layer bit allocation + strong imatrix, mainline types, claims best KLD at 21/22 sizes, MoE-safe.
  = our QX2 already done by someone else: read the per-tensor types from their Qwen3.6-35B-A3B GGUF header and replay the recipe on
  our uncensored Q6_K_P source with llama-quantize --tensor-type. Zero C++.
- ik_llama IQK types reportedly merging into mainline (PR #19726, CPU first) — relevant to the CPU miss path; check our tree.
- QAT methods (ParetoQ, LC-QAT, CAT-Q, ScaleQ-1.58) and MoE-specific 2-bit (BitsMoE, TileQ, GEMQ, MC-MoE): paper-only, no usable
  checkpoints. No natively-trained ternary MoE exists (it would be the ideal model for this box).
