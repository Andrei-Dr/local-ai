# llama.cpp PR / issue mining for decode tok/s on the GTX 1650 SUPER box

**Date:** 2026-09-19

**What was searched**

- `ggml-org/llama.cpp` (mainline): pull requests and issues, via authenticated `gh search prs|issues`, `gh pr view`, `gh issue view`, `gh pr diff`, and the GraphQL API for discussion #24528.
- `PrismML-Eng/llama.cpp` (fork, branch `prism`): every PR (205), every issue (204), releases, discussions.
- `PrismML-Eng/Bonsai-demo`: issues, PRs, `SPECULATIVE.md`, `AGENTS.md`, `README.md`.
- PR and issue bodies, benchmark tables, and comment threads were read, not only titles.

**How presence in `prism@9a9394a` was checked**

- Files were fetched from `https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/9a9394a895b96003ca842a6041cb28ac49a108f7/<path>` and grepped for the PR's symbols, flags, or code lines. File and line references below are to that commit.
- Branch relationship: `gh api repos/ggml-org/llama.cpp/compare/master...PrismML-Eng:llama.cpp:prism` returns `status: diverged, ahead_by: 93, behind_by: 431, merge_base: 5ea87ddad (2026-08-25T05:08:06Z, "webgpu : fix handling of infinity values during ARGSORT and TOP_K (#27538)")`.
- **Consequence:** the fork README says it "tracks current mainline", but it is about 3.5 weeks and 431 commits behind. Any mainline PR merged after 2026-08-25 is absent unless the fork re-implemented it. Where a PR is called "present by date" below, that means it merged before the merge base and was not individually grepped.

**Labels**

- "Verified" = read in the thread or seen in the code at `9a9394a`.
- "(inferred)" = my reasoning, not stated by a source.
- Effort tags: [flag/env only], [cherry-pick], [port needed], [idea only].

**The box (for relevance judgments)**

GTX 1650 SUPER 4 GB (TU116, cc 7.5, no tensor cores, no bf16), i5-10400F 6c/12t (AVX2+FMA+F16C+BMI2, no VNNI/AVX-512), 15 GB DDR4-2667, PCIe 3.0 x16, CUDA 12.0, driver 580, Ubuntu 24.04, gcc 13. Model A: Ternary-Bonsai-2-27B PQ2_0 (`qwen35` hybrid) with grafted MTP head, ~22-29 of 64 layers resident. Models B/C: Gemma4-26B-A4B IQ3_M with `gemma4-assistant` drafter, Qwen3.6-35B-A3B IQ2_M, experts on CPU.

**Measurements supplied by the team lead mid-task (used for re-ranking)**

- MTP n=2: 4.8 tok/s vs 3.2 without speculation. Acceptance 0.75 code / 0.71 reasoning; n=4 -> 0.50; n=15 -> 0.18.
- In spec mode GPU utilization is 88% and PCIe->GPU averages 8.8 GB/s, peaks 10.8 of ~13. The spec-mode path is PCIe-bound.
- Enabling MTP costs ~5 resident layers (rs cache 187 MiB + draft context). n=8 at `-ngl 22` loaded, then died at runtime in `cublasCreate` (VRAM).
- The box already runs `--load-mode none` (the fork's replacement for `--no-mmap`).

**An open mainline issue describes this exact box.** ggml-org#24670 (https://github.com/ggml-org/llama.cpp/issues/24670), OPEN, created 2026-06-15: GTX 1650 SUPER, Ubuntu 24.04, Qwen3.6-35B-A3B-UD-IQ1_M, `-ngl 99 --n-cpu-moe 38 --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75`. `draft-mtp` "initializes without error but never actually activates"; the reporter noted main context `n_rs_seq = 2`, draft MTP context `n_rs_seq = 0`. A commenter (2026-07-27): "Draft-token generation only works when --spec-draft-p-min is set to 0.0."

---

## Section 0. Answers to the three flagged questions

### (a) Reducing rs-cache / draft-context VRAM

- **Snapshot count equals n-max.** Verified at `common/common.h:387`: `need_n_rs_seq()` returns `draft.n_max` for DRAFT_SIMPLE / DRAFT_MTP / DRAFT_EAGLE3 / DRAFT_DFLASH / DRAFT_DSPARK, and `0` for every ngram type. Recurrent-state snapshot memory is therefore linear in `--spec-draft-n-max`. Given the measured acceptance decay (0.75 at n=2, 0.50 at n=4, 0.18 at n=15), n=2 is both the throughput optimum and the VRAM optimum. n=8 needs roughly 4x the snapshots of n=2 for no gain.
- **Snapshot dtype is fixed at F32.** Verified at `src/llama-model.cpp:2774-2775`: `recurrent_type_r = GGML_TYPE_F32`, `recurrent_type_s = GGML_TYPE_F32`. No env variable or CLI flag changes it in prism (grepped `common/arg.cpp`, `llama-context.cpp`, `llama-memory-recurrent.cpp`).
  - Fork issue #143 (https://github.com/PrismML-Eng/llama.cpp/issues/143, OPEN, 2026-09-01) references `GGML_RECURRENT_STATE_F16=1` on bri-prism's unmerged branch `megakernel/rmsnorm-qmv-fuse`.
  - On CUDA/HIP it prevents the existing GDN write-fusion (`ggml_cuda_try_gdn_cache_fusion`) from matching, because the match requires `dst->type == F32`: "instrumented: 144 hits -> 0". Measured "-2.6% — inverts on HIP" (Q1_0 65.0 -> 63.2, PQ2_0 51.97 -> 50.70).
  - [port needed]. (inferred) At 187 MiB that would return roughly 90 MiB, about 2 resident layers.
- **Drafts longer than the snapshot count force a full host checkpoint.** Verified at `tools/server/server-context.cpp:2956-2957`: when the target is rollback-capable (`COMMON_CONTEXT_SEQ_RM_TYPE_RS`) and `draft.size() > llama_n_rs_seq(ctx_tgt)`, the round falls back to a full host checkpoint save/restore.
  - Cost quotes: mainline #28118, "~600 ms of each ~825 ms round — a constant ~73% overhead", 32.4 -> 6.2 t/s; fork #173, "74-77% of each round — about 0.5 s per drafted token".
  - So a combined `--spec-type ngram-*,draft-mtp` must never produce an ngram draft longer than n-max.
- **VRAM flags that exist in prism.** Verified in `common/arg.cpp`: `--spec-draft-type-k`, `--spec-draft-type-v` (quantize the draft context's KV), `--spec-draft-ngl`, `--spec-draft-device`, `--spec-draft-override-tensor`, `--spec-draft-n-min`, `--spec-draft-p-min`.
- **A tight-VRAM recipe from an 8 GB Turing card.** Bonsai-demo PR #171 (https://github.com/PrismML-Eng/Bonsai-demo/pull/171, OPEN; RTX 2080 8 GB, sm_75, Bonsai 2 27B PTQ1_0):
  - "16k context failed to allocate (the ~150 MiB rs-cache buffer had no room left). `-ctk q4_0 -ctv q4_0 -ub 128 -b 256 -np 1 -c 32768` fits; 64k does not."
  - "The server's default `-np 4` multiplies the rs cache and OOMs at 32k."
  - (inferred) A smaller `-ub` shrinks the compute buffers that your `cublasCreate` failure at n=8 was competing with.
- **Capping the draft batch is unsafe for MTP as written.** Mainline #27569 (https://github.com/ggml-org/llama.cpp/pull/27569, OPEN, 2026-08-22, 1 file) caps the draft context's `n_batch`/`n_ubatch` instead of inheriting the target's: "a 7.2 GB Q2_K drafter held 11-13 GB resident." Absent in prism. Reviewer NovNovikov: "This looks unsafe for MTP in particular. `common_speculative_impl_draft_mtp::process()` can pass the full target prefill batch to `llama_decode(ctx_dft, batch)` without the DFlash/DSpark-style chunking." Usable only after adding that chunking.
- **An env override for the snapshot count exists upstream.** Mainline #26499 (https://github.com/ggml-org/llama.cpp/pull/26499, OPEN, 2026-08-03, 1 file, `common/common.h`) adds env `LLAMA_N_RS_SEQ`, decoupling rs slots from the spec type.
  - Example from the PR: `LLAMA_N_RS_SEQ=7 GGML_OP_OFFLOAD_MIN_BATCH=8 llama-server ... --spec-type ngram-mod --spec-ngram-mod-n-min 7 --spec-ngram-mod-n-max 7 --spec-ngram-mod-n-match 24`.
  - Cost on qwen35moe at 168k context: `RS buffer size = 502.50 MiB (1 cells, 40 layers, 1 seqs 7 rs_seq), R (f32): 22.50 MiB, S (f32): 480.00 MiB` — "which is a lot on a 8gig card". Absent in prism (grepped).

### (b) Pinned-memory / async-upload paths with `--load-mode none`

- **`--load-mode none` does disable mmap.** Verified at `src/llama-model-loader.cpp:562`: `use_mmap = load_mode == MMAP || MMAP_MLOCK || AUTO`. The downgrade at line 1224 (`if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev))`, which swaps a pinned host buffer for a plain CPU buffer) therefore does not fire. CPU-resident weights are placed in `CUDA_Host` (pinned) when the allocation succeeds.
- **The pageable fallback is silent.** Per mainline #28223: "If the pinned allocation fails, the buffer falls back to pageable memory." On a 15 GB host, confirm the load log shows `CUDA_Host model buffer size = ...`. (inferred) Your 8.8 GB/s average suggests the weights are pinned; pageable transfers usually run well below that.
- **What pinning is worth.** Mainline #28223 (https://github.com/ggml-org/llama.cpp/pull/28223, CLOSED/parked 2026-09-09): "Prefill of a 26k prompt went from 166 t/s to 330 t/s with a cold page cache, and to 379 t/s once the per-layer embedding rows were cached", on 2x RTX 3090 with 40 expert layers on the host.
- **Caveat for a future repack port.** `CPU_REPACK` is an extra buffer type and is not pinned. PQ2_0 repack is not selected on this CPU today (see item 6), so this is latent. If an AVX2 repack path is added, repacked weights would stream to the GPU from pageable memory for batch >= 2.
- **Overlapped upload exists only as an upstream PoC.** Mainline #21067 (https://github.com/ggml-org/llama.cpp/pull/21067, am17an, OPEN PoC, last updated 2026-08-15; flag `--prefetch-weights`, llama-bench `-pw`) uploads layer n+1's overridden weights through a second backend instance (separate CUDA stream) while layer n computes. Absent in prism (flag grepped). [port needed]: it touches the scheduler in `ggml-backend.cpp`.
  - Body: "Note that `--no-mmap` is necessary for this work, otherwise the operations are implicitly serialized due to the weights not being pinned." "If C_n > T_n+1 we can 'hide' the transfer latency."
  - Dense test in the body: `llama-bench -m Qwen3.5-27B-Q4_K_M.gguf -fa 1 -p 2048 -ub 512,1024,2048 -d 0..50000 -n 0 -ot "ffn_(gate|up|down).*=CPU" -pw 0,1 --mmap 0` (results are an image, no table).
  - noonghunna (MoE, 2x 3090): prefill 10K 902.5 -> 1035.3 t/s (+14.7%), 90K 734.2 -> 835.7 (+13.8%), but short-prompt TTFT 1085 -> 1604 ms.
  - EJainDev (dense): "As for the dense model, I'm not noticing a measurable speed up. With upstream llama.cpp and this PR on Qwen3.8 27B I am achieving 4tps with remarkably low GPU utilization. I also do have PCIe 5.0x16."
- **Reducing bytes streamed is the only lever.** (inferred, re-ranks the list) In spec mode every forward pass streams all non-resident weight bytes, so tok/s is roughly proportional to 1 / (model bytes - resident bytes). Layer placement does not change that volume; total bytes do.
  - The PTQ1_0 file is 1.75 bpw versus 2.13 bpw for PQ2_0. Fork PR #148: "PTQ1_0, ternary at group 128 (1.75 bpw, lossless vs PQ2_0)". Demo PR #171: "Weights are 5.53 GiB".
  - That is ~18% fewer bytes over PCIe per pass, plus more resident layers for the same VRAM.
  - Verified: CUDA runs the PTQ1_0 MMQ path at every batch size (fork #164 and #160, both merged 2026-09-08, present).
  - Verified downside: the x86 CPU `vec_dot` for PTQ1_0 is scalar-only in prism. Fork #181, OPEN, adds AVX-VNNI only; its thread notes that "under clang the generic auto-vectorizes to near-parity" (4.58 tg128 generic vs 4.65 with the kernel on a Core Ultra 9 275HX). With `GGML_OP_OFFLOAD_MIN_BATCH=2` and MTP always on, the CPU matmul path is barely exercised.
  - [flag only / model swap]. It requires re-grafting the MTP head onto the PTQ1_0 file.

### (c) A separate small draft model for qwen35-family targets

- **It works on hybrids now.** Verified present in prism: fork #173 (https://github.com/PrismML-Eng/llama.cpp/pull/173, MERGED 2026-09-16) adds `DRAFT_SIMPLE` to `need_n_rs_seq()` and replaces the hardcoded `cparams.n_rs_seq = 0` on the draft context, so plain drafter+target on hybrids no longer falls back to checkpoints. `common.h:389` confirms DRAFT_SIMPLE is in the list.
- **Measured results are poor.**

  | Source | Setup | Result |
  |---|---|---|
  | Fork #173 (Arc B390, Vulkan, greedy) | 2B-PTQ1_0 drafting 27B-PTQ1_0 vs 4.43 t/s baseline | k=1 "1.137x / 1.356x (two prompts)"; k=2 0.482x -> 1.085x; k=4 0.535x -> 0.727x |
  | Fork #203 (M4 Max, Metal) | `draft-simple` + Qwen3.5-4B-Q4_K_M | 5.7 tok/s at 9% acceptance (zh), 12.2 at 52% (code), vs 31.7 baseline |
  | Fork PR #205 | `-md` sidecar vs in-file MTP | "A separate `-md` sidecar duplicates the vocabulary (~92% of its bytes), pushing the per-token draft cost ratio to rho ~ 0.43, which turns even 40% acceptance into a net loss (measured: 5.16 t/s vs 16.6 t/s no-spec). Sharing the vocabulary drops rho to ~0.06." |

- (inferred) A hybrid drafter also needs its own rs slots plus KV in VRAM. On 4 GB it loses to the in-file MTP head. Not recommended.
- **The DSpark drafter is not a fit either.**
  - `SPECULATIVE.md`: "`--drop-shared-tensors` omits the embedding and lm head (the runtime borrows the target's), shrinking the drafter to about 0.6 GB with unchanged acceptance"; L40S "1.8-2.4x for the ternary 27B (2.06x blended)"; "`--spec-draft-n-max` must equal the drafter's block size (4 for the current drafters); a smaller value crashes at the first draft round"; "Cross-request prompt-cache reuse is disabled"; single slot.
  - Bonsai-demo issue #105 (A5000, temp 0.7, `-c 32768`): depth ~0 1.11x (accept 0.449), ~8k 0.89x (0.335), ~16k 0.67x (0.234), ~24k 0.37x (0.009). It silently falls back to autoregressive decode above 32k context ("no implementations specified for speculative decoding").
  - Fork issue #109 (CLOSED 2026-08-28): DSpark drafter load crashed with "invalid vector subscript" when it did not fit in VRAM (< 9 GB GPU).
  - (inferred) 0.6 GB of VRAM costs more resident layers than MTP does.

---

## Section 1. Ranked actionable items (most tok/s per unit effort first)

### 1. [flag only] Attention/GDN on GPU for all 64 layers; only FFN weights on CPU

- Link: https://github.com/ggml-org/llama.cpp/pull/26622 — "llama : add --n-cpu-ffn option", MERGED 2026-08-27 (John-194), 3 files (`common/arg.cpp`, `common/common.h`, `tools/llama-bench/llama-bench.cpp`).
- Mechanism: convenience wrapper equivalent to `-ot` on the FFN tensors of the first N layers. Every layer's attention/GDN projections, KV, and recurrent state stay on the GPU; the CPU runs only the three FFN matmuls per layer.
- Quoted: "Putting dense model layers on the CPU via `--ngl` causes major slowdowns… I achieved a tolerable speed of 15-25 t/s tg using Q4_K_M model with a large context of over 90k… using only 16 GB of VRAM." (RTX 4070 Ti SUPER, i5-13600KF, DDR5-5800; the comparison chart is an image.) "An optimization is possible by prioritizing the largest FFN layers first… an extra 10% tg speed."
- In prism@9a9394a: ABSENT (grepped `"--n-cpu-ffn"` in `common/arg.cpp`). Equivalent today: `-ngl 99 -ot "blk\.(N|…|63)\.ffn_(gate|up|down)\.weight=CPU"`.
- (inferred) FFN is ~17.1B of 26.9B parameters (3 x 5120 x 17408 x 64). Non-FFN weights at 2.13 bpw are roughly 2.4 GiB, so all attention/GDN weights plus KV, rs cache and compute buffers may just fit in 4 GB with zero FFN layers resident.
- Risks: about two graph splits per layer (~128 split boundaries). VRAM is tight and MTP reduces it further. In spec mode this does not change the PCIe volume (see 0(b)), so its benefit is mainly for non-spec decode. Measure before adopting.

### 2. [flag only / model swap] Reduce bytes streamed per pass; keep CPU-resident weights pinned

Covered in Section 0(b).

- Links: https://github.com/ggml-org/llama.cpp/pull/28223 (closed/parked 2026-09-09), https://github.com/ggml-org/llama.cpp/pull/21067 (open PoC), https://github.com/PrismML-Eng/llama.cpp/pull/148 (PTQ1_0, merged 2026-09-04), https://github.com/PrismML-Eng/llama.cpp/pull/164 and https://github.com/PrismML-Eng/llama.cpp/pull/160 (PTQ1_0 MMQ at every batch, merged 2026-09-08).
- Related env: mainline #18535 (https://github.com/ggml-org/llama.cpp/pull/18535, MERGED 2026-01-08) added `GGML_OP_OFFLOAD_MIN_BATCH`. PRESENT in prism: `ggml/src/ggml-cuda/ggml-cuda.cu:5811`, default 32. The PR says the optimal value "is best determined empirically".

### 3. [cherry-pick] Fix MTP shared-KV misdetection on the qwen35 family

- Link: https://github.com/ggml-org/llama.cpp/pull/27781 — OPEN, 2026-08-27 (Gwonk1), 7 files (`common/speculative.cpp`, `include/llama.h`, `src/llama-context.cpp`, `src/llama-kv-cache-iswa.cpp/.h`, `src/llama-kv-cache.h`, `src/llama-memory.h`).
- Mechanism: `is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;` is true for every MTP context. Gemma4-assistant really shares the target's cells; qwen35 gets its own `llama_kv_cache` filtered to `il >= n_layer()`.
  - "The catch-up decode in `process()` is skipped, so the draft KV never advances past whatever the prompt left there. And the drafting loop takes the shared-memory branch, which places every draft token at `dp.n_past` instead of `dp.n_past + i + 1`."
  - The fix exposes `llama_memory_has_shared_cells()`.
- Quoted (Qwen3.8-27B UD-Q4_K_M, embedded MTP, n-max 2, CPU backend, 16 threads):

  | Prompt | master | this PR |
  |---|---|---|
  | extraction, acceptance | 1.000 (266/266) | 1.000 (266/266) |
  | prose, acceptance | 0.597 (108/181), mean len 2.19 | 0.675 (114/169), mean len 2.34 |
  | prose, decode t/s | 2.93 | 3.23 |

- In prism@9a9394a: BUG PRESENT. `common/speculative.cpp:2129` has the exact line; `has_shared_cells` does not exist.
- Risk: unreviewed upstream (only a bot comment), and it adds a public API function. The mechanism is directly applicable to your grafted head.

### 4. Your 14-line Hadamard MTP fix is not merged by PrismML

- Link: https://github.com/PrismML-Eng/llama.cpp/pull/205 — OPEN, 2026-09-19 (zhaoyilun), 1 file `src/models/qwen35.cpp`.
- Mechanism: "`qwen35::graph_mtp` does its own `ggml_get_rows()` on the token embedding table but never restores the primal basis… Apply the same `rot` + `signs` inverse the trunk uses, keyed on the table actually looked up (`layer.nextn.embed_tokens` when present, otherwise `model.tok_embd`). 14 lines, no new API."
- Error without the fix: `Hadamard-latent table 'token_embd.weight' is read without the inverse transform`.
- In prism@9a9394a: ABSENT. `qwen35.cpp:634-640` does `get_rows` followed by `cb(tok_embd, "mtp_tok_embd", il)` with no inverse transform.
- Quoted numbers:
  - "draft acceptance is identical to the known-good community-patched build on the same three probes — 49/90, 59/71, 57/75 (54.4% / 83.1% / 76.0%)".
  - "12-prompt A/B vs `--spec-type none` on RTX 4080 SUPER 16 GB: median 1.338x decode, aggregate acceptance 68.1% (1.23-1.70x; code/format 1.5-1.7x, free reasoning prose 1.25-1.3x)".
  - Repro material: https://github.com/zhaoyilun/bonsai2-27b-mtp-repro.
  - The same patch ships as `runtime/bonsai-mtp-embedding.patch` in `ProCreations/Ternary-Bonsai-2-27B-MTP`.
- Fork issue #203 Metal sweep (M4 Max, same grafted head, baseline 31.7 tok/s):

  | n-max | zh tok/s (acceptance) | code tok/s (acceptance) |
  |---|---|---|
  | 3 | 14.9 (37%) | 22.8 (75%) |
  | 5 | 13.4 (38%) | 23.3 (73%) |
  | 7 | 9.6 (21%) | 21.6 (70%) |

  This matches your measured decay with depth.
- Action: support or co-sign #205 rather than opening a duplicate.

### 5. [flag only] Speculation sizing around the rs cache

Covered in Section 0(a). Two additions:

- ggml-org#24670 (your exact GPU): a commenter reports drafts are generated only with `--spec-draft-p-min 0.0`. Your MTP works, but re-check `timings.draft_n` if you raise p-min.
- `SPECULATIVE.md`: "Each API response's `timings` object includes `draft_n` and `draft_n_accepted`. If `draft_n` is missing or zero, speculation is not active."

### 6. [port needed] AVX2 repack GEMV/GEMM for PQ2_0 (re-ranked down for spec mode)

- Link: https://github.com/PrismML-Eng/llama.cpp/pull/86 — "ggml-cpu: x86 AVX512-VNNI repack GEMV/GEMM for Q1_0 and Q2_0", MERGED 2026-07-18 (bri-prism). After the August rebase it became `pq2_0_4x8`.
- In prism@9a9394a:
  - `ggml/src/ggml-cpu/repack.cpp:5352-5357` selects `pq2_0_4x8_q8_0` only `if (ggml_cpu_has_avx512() && ggml_cpu_has_avx512_vnni())`.
  - Kernel bodies `ggml_gemv_pq2_0_4x8_q8_0` / `ggml_gemm_pq2_0_4x8_q8_0` in `arch/x86/repack.cpp:6586` and `6643` are guarded by `__AVX512F__ && __AVX512BW__ && __AVX512DQ__ && __AVX512VNNI__`, with a generic fallback otherwise.
  - Your CPU therefore gets no repack for PQ2_0.
  - Also verified: `ggml_vec_dot_pq2_0_q8_0` (`arch/x86/quants.c:561`) has only `(AVX512VNNI && AVX512VL) || AVXVNNI` and scalar paths, so your AVX2 kernel is new to the fork.
- Quoted from #86 (EPYC Zen 4).

  Kernel, single thread, m=4096 k=14336:

  | | n=1 | n=8 | n=512 |
  |---|---|---|---|
  | q2_0 vec_dot | 17.7 GF | 17.7 GF | 17.7 GF |
  | q2_0 repack | 52.6 GF (2.97x) | 87.4 GF (4.94x) | 87.5 GF (4.94x) |

  End to end (t=8): 27B Q2_0 pp512 2.86 -> 13.3 t/s (4.7x), pp8 2.80 -> 12.3 (4.4x), tg128 2.42 -> 5.85 (2.4x).

  Structural wins named: "one `sum(qy)` dpbusd per activation sub-block reused across all 4 columns, per-lane fp32 partials with a single horizontal reduction per output tile."

  Numerics: "27B Q2_0 diverges at ~token 50 due to fp32 accumulation-order differences… mean KLD 2.08e-4… top-token agreement 99.17%, PPL ratio 1.0013 +/- 0.0011."

  Known limit: "Both formats saturate ~88 GF/thread." "`test-backend-ops` does not allocate repack buffers in this tree, so repack kernels have no unit coverage."
- Related prior art: https://github.com/PrismML-Eng/llama.cpp/pull/29 — "Bit-interleaved Q1_0 8x32 repack kernels for x86 AVX2", OPEN since 2026-05-02 (pl752). Abandoned by the author: "feel free to borrow/fork/continue from where I left it". It conflicts with current prism (39 files).

  Quoted (mobile Zen 4, LPDDR5-6400, `-t 6`):

  | flow | run | dot | repack | delta |
  |---|---|---|---|---|
  | AVX2 | pp512 | 139.80 | 190.98 | +36.61% |
  | AVX2 | tg128 | 91.70 | 115.17 | +25.59% |

  A later revision with 16x4 gemm / 32x1 gemv tiles: AVX2 pp512 213.79 -> 285.04 (+33.32%), tg unchanged.

  A second tester on Intel 165U with DDR5-5600: "pp512 70.96 (your branch) vs 61.51, tg128 42.71 vs 38.97". pl752 attributes the smaller gain to memory bandwidth and AVX2's 16 ymm registers spilling.

  Useful method note: "benchmarks are run with `-t 6` instead of `-t 10` as SMT threads don't help significantly with performance anymore, but increase memory pressure."
- (inferred) Your AVX2 vec_dot (38.5 -> 5.6 cycles per 32 weights) already captured most of what the 2.97x figure measures. The remaining gain is reusing the per-block `sum(qy)` correction across rows, plus batched rows when verify batches run on the CPU. With MTP on and `GGML_OP_OFFLOAD_MIN_BATCH=2`, matmuls run on the GPU, so this matters only for non-spec decode.
- Blockers and risks: enabling PQ2_0 repack on AVX2 exposes the fork #180/#204 segfault (Known bugs 2), and repacked weights are not pinned (Section 0(b)).

### 7. [port needed] Device-side expert cache for models B/C — substantial prior art

**7.1 Best base: mainline #27861** — https://github.com/ggml-org/llama.cpp/pull/27861, "llama: GPU-resident LRU cache for host-offloaded MoE expert weights", OPEN draft, 2026-08-28 (csantiago78). 12 files, including new `src/llama-moecache.cpp/.h`. Flags `--moe-expert-cache N` (slots per host-resident expert layer) and `--moe-expert-cache-inserts`.

- Mechanism (quoted):
  - "Per cached layer: companion tensors `[ne0, ne1, K+1]` for up/gate/down in the device buffer of that layer's router. Slot `K` is permanently zero."
  - "An I32 `expert id -> slot` table per layer, two copies: device copy: `ggml_get_rows` remaps `selected_experts` into slot ids for a second `mul_mat_id` chain over the cache tensors. Uncached ids map to the zero slot… host copy: passed as `src[3]` to the CPU `mul_mat_id`, which skips cached ids and zeroes their dst rows."
  - "The two down-projection outputs are summed."
  - "Decode-only (`n_tokens == 1`)."
  - "Updates are throttled and asynchronous: evictions are published at a decode-boundary sync point, slices are copied by a worker thread via `ggml_backend_tensor_set`, and the new mapping is only published at a later sync point after the copy completed… (Synchronous uploads were measured to eat the entire win.)"
- Routing measurement (54k-record workload, Qwen3.8-Flash-Next, 512 experts / 10 routed): "No exploitable static skew: a top-32 'hot expert' list learned on half the workload covers only ~10% of the other half (uniform = 6.2%). Static pinning of experts is a dead end. Strong temporal locality: a per-layer LRU-64 would hit ~67%, LRU-128 ~81%."
- Numbers: "18.4 -> 24.2 tok/s (+31%) with 48 slots/layer (~4.1 GiB VRAM) at 2 uploads/layer/step" (2x RTX 3090). Runtime re-check: 22.7 vs ~18-19 tok/s.
- Third-party results in the thread:
  - **Syugakubusei** (Ryzen 5900X, RX 7600 8 GB, DDR4-3200, Vulkan): GigaChat-20B-A3B Q4_K_M 17.1 -> 20.4-20.6 tok/s (26 slots, hit 60.4%, 4622 MiB); Qwen3-30B-A3B Q4_K_M 14.4 -> 16.5 (32 slots, hit 74.7%, 4316 MiB).
  - **Syugakubusei, MTP extension**: changes the guard from `n_tokens == 1` to `n_tokens > 0 && n_tokens <= 4`, with `ggml_cont` + flatten to `[n_expert_used * n_tokens, 1]` for `ggml_get_rows`, then reshape back. Qwen3.6-35B-A3B: "plain decode ~17.1 tok/s; MTP (n-max 2) without expert cache ~23-24; MTP2 + expert cache ~30-32".
  - **Syugakubusei, Gemma4-26B-A4B**: fused path `gate_up_g = ggml_mul_mat_id(cache_gate_up, input, slot_ids); act_g = ggml_geglu_split(...); down_g = ggml_mul_mat_id(cache_down, act_g, slot_ids); experts = ggml_add(cpu_experts, down_g);`. With per-expert down scales (`down_exps_s`), `build_lora_mm_id()` wraps the MMID in a `MUL`, so the cache table must be attached to the inner node: `experts_mmid = experts->src[0]; experts_mmid->src[3] = mcache->host_table; experts_mmid->op_params[0] = mcache->n_slots;`. At 64K context:

    | Mode | Cache | Gen t/s | Hit rate | Free VRAM |
    |---|---|---|---|---|
    | MTP off | OFF | 16.40 | — | — |
    | MTP off | C24 | 18.33 (+11.7%) | 63.3% | — |
    | MTP2 | OFF | 22.88 | — | 4397 MiB |
    | MTP2 | C20 | 26.48 (+15.7%) | 54.3% | 763 MiB |
    | MTP2 | C24 | 27.35 (+19.6%) | 59.8% | 321 MiB |
    | MTP2 | C28 | 28.65 (+25.3%) | 65.1% | 0 MiB |

    Capacity sweep: C8 21.93 (28.0%), C12 22.87 (39.6%), C16 24.33 (47.9%), C20 24.93 (54.3%), C24 26.20 (60.1%), C28 27.07 (64.7%). Over-allocating past available VRAM "produced a large performance drop".
  - **momcilovicrobert-momc** (RX 9070 XT 16 GB, Vulkan, Qwen3.6-35B-A3B UD-Q4_K_XL, `-ncmoe 24`): cache off 30.4-32.2; 32 slots 37.3-38.2 (53% hit); 48 slots 39.9-40.7; inserts 4 showed no gain over 2; prefill unchanged at ~1340 t/s.
  - **mtrx93** (GLM-5.3-Flash IQ4_XS, single 3090, PCIe 3.0 x16, upload-limited VM): cache off 6.78; inserts 1 7.47 (hit 25.7-27.8%); inserts 2 6.03 (hit 2.5%); inserts 8 6.15 (hit 0.2%). Uploads saturating PCIe collapse the hit rate.
  - **pjsgsy** (RTX 3060 12 GB + i7-8700, the closest CPU class to yours):
    - "Main finding: unconditional admission churns, and the gate is the real win. Ungated, 384 decode steps produced 74,482 uploads against 73,074 evictions. In-flight slots are excluded from victim selection, so once the upload worker saturates, effective capacity collapses."
    - Policy comparison: ungated 37-43% hit, 55k-74k uploads, yield 1.4-1.7x; "window 16, admit 3" 48-51% hit, 15k-16k uploads, yield 6.6-7.0x.
    - "End-to-end on the flash model… 5.6 -> 10.0 t/s decode (+78%), prefill unchanged."
    - "Eviction policy, by contrast, is not where the win is: all-time LFU measured -16% yield vs LRU… 16-step half-life gave +164% yield where counting all time gave +38%."
    - "Report uploads/evictions/bytes uploaded/bytes served/yield, not just hit rate." A routing-trace dump (`LLAMA_MOE_CACHE_TRACE`) allows offline policy evaluation.
    - Caveat: with `--fit` promoting whole layers the cache can lose; forcing `--cpu-moe` then 28.3 -> 38.5 t/s (+36%) at 96 slots / admit 2, prefill 328 -> 250.
  - **Traps reported by pjsgsy** (each "a hard failure"):
    - `flockfile`/`funlockfile` do not link under MSVC.
    - "The cache silently disabled itself: eligibility requires the layer's router on a device buffer, but routers are often bf16, and sm_86 can't run bf16 ops on CUDA, so fit puts them on CUDA_Host. Allocating the cache buffers on the GPU device directly fixes it." Turing has no bf16 either, so this will hit you.
    - "Duplicate dummy slots break the batched CUDA mul_mat_id kernels (MMQ / *_switch_ids / the host sorting fallback) — illegal memory access. Only mmvq/mmvf are duplicate-safe, so a multi-token cache must stay in their window: n_tokens <= get_mmvq_mmid_max_batch(), i.e. 6 (IQ3_S), 7 (IQ3_XXS/IQ2_S), 8 default on sm_86; F16/BF16 experts need 1."
    - "Slots have to be reserved via --fit-target… without room, WDDM pages the model buffer and prefill collapses (328 -> 41.7 t/s)."
    - "`llama_moe_cache_init` must run before `sched_reserve()`, so the cache chain is in the reserved worst-case graph."
  - **mobilinkd**: deterministic garbage on SYCL, because the in-place weight reorder conflicts with the upload worker.
  - **feal87**: asks for perplexity data; outputs are not greedy-identical because CPU and GPU round differently.
  - **dany-on-demand**: persistent degenerate output across requests.

**7.2 Closest hardware analog, and it regressed:** discussion #24528 (https://github.com/ggml-org/llama.cpp/discussions/24528) and PR #24524 (https://github.com/ggml-org/llama.cpp/pull/24524, CLOSED 2026-06-12, leloch).

- Design: "MUL_MAT_ID stays on the CPU. Inside the CPU kernel, thread 0 dispatches one batched matvec over the cached (hit) rows on the GPU while the other threads compute the miss rows." Decode-only fill; hot-set persistence on disk; bail-out if slower than CPU.
- Quoted: GLM-5.1 754B IQ2_M 13.96 -> 17.49 (+25%); Qwen3.5 397B Q3_K_XL 28.18 -> 30.25 (+7%) on 4x 3090 / EPYC 48c. "10% of experts account for 80-81% of cache hits, and the top 30% account for 95-96%."
- The RFC also records why earlier attempts failed: "@batot1 measured ~3x decode regression from forced decode offload without residency (#20757), and the Metal slot-pool experiment in the same thread was 2x slower than vanilla even at 97-99% hit rate, purely from per-layer sync points."
- Closed by the bot and maintainers for AI content and "just too large for any maintainer to review"; am17an suggested an RFC.
- **batot1, GTX 1080 Ti 11 GB, Qwen3.6-35B-A3B Q8_K_XL MTP, `--cpu-moe`, `-t 6`:** hard-off 19.32 tok/s. With cache budget 4096 MB 13.25; 1024 MB 14.38; 512 MB 17.26; 256 MB 17.19; 128 MB 18.64; 64 MB 18.71; 32 MB 18.72.
  - "Larger cache budgets caused larger regressions."
  - Bail-out log: "cache-engaged nodes average 285us vs 266us pure-CPU".
  - leloch's reply: "on a relatively slow GPU without tensor cores (like 1080), and not much spare VRAM after placing all dense layers on a GPU, for MoE it could be just faster to compute expert activation on CPU + RAM… Ideally you want enough VRAM spare for cache… around 20%-30% of MoE expert weights."
  - This is a different design from #27861, but budget for a negative result on a 1650 SUPER.
- **noonghunna** (2x 3090, 118B and 284B MoE): coverage 11.7% -> ~50% hit, 17.2 t/s; 18.7% -> ~55%, ~17.7; 25.5% -> ~63%, 19.2; 30.6% -> ~70%, ~21.9. "Coverage -> hits -> speed stayed essentially linear."
- **xashr**, comparing the forks (DeepSeek V4, 155 GB model): decode llama.cpp 16.5 / Lidenburg 19.6 / leloch 21.7 / miltos22 22.7; prompt processing 160 / 345 / 55 / 49. "Great TG, very low PP" for the two caching forks.

**7.3 Other attempts**

- #26563 (https://github.com/ggml-org/llama.cpp/pull/26563, CLOSED 2026-08-09) and successor #26824 (https://github.com/ggml-org/llama.cpp/pull/26824, CLOSED 2026-08-10, 42 files), miltos22, `-ehs N`: heat map with decay, hottest experts cached on GPU, cold path on CPU. "Measured on Qwen3.6-35B-A3B Q2_M / Q5_K_P with 8 GB VRAM: ~1.7-2.1x steady-state decode speedup over stock (56 vs 33 tok/s, 36 vs 17 tok/s)." "Tier only engages at n_tokens == 1." Multi-GPU load failures were reported; prompt processing was "glitched" per the author.
- #23170 (https://github.com/ggml-org/llama.cpp/pull/23170, CLOSED 2026-05-25, 1 file `ggml-backend.cpp`): treats the scheduler staging tensor as an expert cache via a residency bitset. Initial +5.4% total tok/s, but perplexity went to `nan`; after scoping the bitset to the compute epoch, reviewer ernestuz-b: "the epoch invalidation clears the state before it can be used, so you lose any speed gain." The author closed it: "I still missed the invalidation cost." Lesson: the scheduler staging buffer is not a usable cache; a persistent buffer with explicit expert-to-slot bookkeeping is required.
- #21609 / #21614 / #21620 (CLOSED 2026-04-08, e1n00r): N-slot LFRU cache with prefetch, withdrawn after the AI-content bot and am17an: "Please stop submitting such large PRs. No one will review them unless you demonstrate you can understand and maintain the code."
- #28414 (https://github.com/ggml-org/llama.cpp/pull/28414, OPEN, 2026-09-04, 8 files) `--prefetch-experts-slots`: 1-deep lookahead H2D on a second CUDA stream into rotating staging buffers. Prefill-only (fires only when `ids->ne[0]*ids->ne[1] >= 2*n_expert`); "30% is a real gain" per the author. 1jeffchristensen reproduced silent wrong output on multi-GPU (runs of `/` after a few hundred tokens). pwilkin: "I'm not considering any PRs of this sort unless someone can clearly show me that this beats purely using `--mmap` with the page cache for performance."
- #25294 (https://github.com/ggml-org/llama.cpp/pull/25294, OPEN, 2026-07-04, 17 files): streams routed experts from SSD with a device-side slot cache, a CPU id-remap custom op, and wave-partitioned prefill. GLM-5.2 UD-Q2_K_XL on GB10: 64 slots ~1.83 tok/s decode at 73% hit; 90 slots ~2.20 at 79%. darksideX1 retracted a token-parity claim: "by ~96 tokens only about 1 in 15 does." An auto-sizing abort was reported for `n_expert_used >= 6`.
- #17044 (OPEN since 2025-11-06): empty body, no discussion.
- Issue #20757 (CLOSED 2026-09-09): "Two-tier GPU+RAM expert cache for MoE offload (pluggable eviction policy)" is the original RFC thread referenced above.

**7.4 Related merged and open MoE items**

- #28739 (MERGED 2026-09-11, ABSENT in prism by date): "skip 0-sized ids tensor when offloading selected experts". Fixes an out-of-bounds read when always offloading (`GGML_OP_OFFLOAD_MIN_BATCH=0`), reported on `qwen4exp`.
- #25952 (MERGED 2026-09-01, absent by date): fused MoE weighted expert reduction on CUDA, prefill +3.6% to +7.1%.
- #26631 (OPEN): `MUL_MAT_ID` with a `-1` index to skip computation. A cleaner primitive than zero slots for a future cache.
- #26079 (MERGED 2026-08-27, absent by date): per-hardware, per-quant switch points for the MMVQ -> MMQ decode crossover.
- #28781 (OPEN): fixes an MTP crash on pre-Pascal GPUs from a degenerate `get_rows`. Not your GPU, but its command line is the model-C pattern (`--n-cpu-moe 999 … --spec-type draft-mtp --spec-draft-n-max 2`).

**7.5 Maintainer stance.** pwilkin is skeptical (quoted above). am17an wants small PRs or an RFC and points to his own #21067. The AI-content policy is enforced by bot. Plan on carrying an expert cache out-of-tree.

**7.6 Gemma4 MTP drafter support.** `gemma4-assistant` arch PRESENT in prism (`src/llama-arch.cpp:60`). Mainline history: #23398 "llama : add Gemma4 MTP" (merged 2026-08-25), #24282 (E2B/E4B assistants, merged 2026-06-09), #28183 "model : fix gemma4-assistant" (MERGED 2026-09-01, ABSENT by date), #28630 "fix MTP context kv cache allocation" (MERGED 2026-09-11, absent by date).

### 8. [cherry-pick, low value here] CUDA graph reuse for the MTP draft

- Link: https://github.com/ggml-org/llama.cpp/pull/28549 — MERGED 2026-09-16 (gaugarg-nv), 2 files (`src/llama-context.cpp/.h`).
- Mechanism: "MTP alternates between output-producing draft batches, and no-output prefill and catch-up batches. Previously both shapes reused one `llm_graph_result`, so they shared the same CUDA graph cache key and repeatedly replaced each other's captured graph." The PR adds a second graph-result arena.
- Quoted: "RTX 5090… Qwen3.6-35B-A3B-UD-Q4_K_M MTP3 shows a gain of 4-5%" (279.04 -> 291.33 avg pred t/s, accept 0.7061 in both).
- In prism: ABSENT (only `gf_res_prev` and `gf_res_reserve` exist in `llama-context.cpp`).
- (inferred) Small for a partially offloaded model with many splits, but no longer zero now that the GPU is 88% busy.
- Related: #25749 "Enable CUDA graphs on Volta+Turing" (MERGED 2026-07-16; present by date). `GGML_CUDA_DISABLE_GRAPHS=1` reverts it. ggerganov noted a Volta issue report, #26119.

### 9. [cherry-pick, medium] Adaptive MTP depth; rejection sampling

- #27210 (https://github.com/ggml-org/llama.cpp/pull/27210, OPEN, 2026-08-17, 15 files, 88 comments): `--spec-type draft-mtp-adaptive --spec-draft-n-max 12`. A counting state machine with a floor and cold start of 3.
  - "It was experimentally determined that a draft MTP depth of 2 or 3 is close to optimal for reasoning and prose."
  - "It is not recommended to set `spec-draft-n-max` to values lower than 7… If you find yourself in a situation where insufficient VRAM prevents you from setting a depth of at least 7… it's generally going to be best to just stick with regular MTP."
  - That needs 7+ rs snapshots, which conflicts with your VRAM budget. Skip.
- #25726 (OPEN, 2026-07-15, 3 files): a simpler adaptive heuristic (`--spec-draft-adaptive-length-threshold N`, "N=11 worked well"). No numbers in the thread.
- #27694 (https://github.com/ggml-org/llama.cpp/pull/27694, OPEN, 2026-08-25, 7 files): `--spec-draft-sampling greedy|probabilistic`. Standard rejection sampling (`min(1, p/q)`, residual `norm(max(0, p - q))`) for draft-simple and draft-mtp. Relevant only when sampling at temperature > 0: "acceptance falls away as temperature rises" under the current exact-match rule. Detailed numbers are in a collapsed section I did not expand.
- #28473 (OPEN): stops MTP drafting when the per-sequence limit is reached ("reduces draft decodes from 8 to 1" when `n_predict=3`). Minor.
- #24025 "qwen35: use post-norm hidden state for MTP" (MERGED 2026-06-03, present by date): aggregate acceptance 0.6859 -> 0.7241. Worth confirming your grafted head was exported with the same convention (not checked).

### 10. [env/flag] OpenMP vs the built-in threadpool — no benchmark numbers found

- The only evidence is #28785 (https://github.com/ggml-org/llama.cpp/pull/28785, OPEN, 2026-09-11, "skip threadpool for graphs with no CPU work"). After the author saw overhead on an L4 box, max-krasnyansky: "Built with OMP? Can you please try disabling OMP and running the same bench." angt replied: "Without OMP, and so everything looks fine."
- Verified at `ggml/src/ggml-cpu/ggml-cpu.c:3887-3898`: prism sets only `KMP_BLOCKTIME=200`. The `OMP_WAIT_POLICY=active` lines are commented out. libgomp ignores `KMP_BLOCKTIME`, so your 20% in the libgomp barrier is governed by libgomp defaults.
- Cheap experiments: `GOMP_SPINCOUNT`, `OMP_WAIT_POLICY=passive`, or a `-DGGML_OPENMP=OFF` build with `--poll` and `--prio`.
- Related: #27138 (MERGED 2026-08-18, present by date) shares thread pools when `n_threads` differ; #27143 (OPEN, draft) shares thread pools between target and drafter.
- (inferred from `ggml/src/ggml-cpu/ops.cpp:11890-11965`) The CPU FWHT splits work by rows (`rows_per_thread = (nr + nth - 1) / nth`). At batch 1 there is one row, so only thread 0 works while the rest wait at the barrier. The `ggml_compute_forward_mul` in your profile is probably the Hadamard sign multiply, which fork #165 fused into FWHT only on Metal and CUDA.
- Low priority now that spec mode is GPU- and PCIe-bound.

### 11. Gated delta net / qwen35 — nothing actionable for decode on this box

- Fork #165 (MERGED 2026-09-08, present): six graph fusions, "-25% ops/token"; Metal 2B +7.7%, 27B +2.6%, CUDA 9B +2.5%. Includes GDN raw gates on CPU/Metal/CUDA (`GGML_GDN_RAW_GATES_DISABLE=1` restores separate ops), `ggml_swiglu_split` gated norm, and one `l2_norm` over q and k.
- Fork #163 (MERGED 2026-09-08, present): `cudaDeviceGetAttribute` instead of `cudaGetDeviceProperties`. "409 `cudaGetDeviceProperties` calls / 337 ms per llama-bench run… about 0.3 ms per decoded token". Decode +8.2% on 9B PQ2_0, +3.7% on 27B PQ2_0.
- Fork issue #143 (OPEN): in-place GDN state read plus fused beta/alpha glue, "+6.7% / +7.2%" tg128 on gfx1201. GPU-side, unmerged, HIP-measured. It also found two bugs: `ggml_cuda_op_scale` hard-asserts F32, and CUDA `supports_op` for `GATED_DELTA_NET` rejects any op with `src[6]` set, which causes a silent CPU fallback.
- Mainline #26001 (OPEN): chunked GDN CUDA prefill kernel; requires "NVIDIA Ampere+ with BF16 tensor core", so not Turing. Mainline #22587 (OPEN): autoregressive GDN kernel rewrite, tg flat (-0.5% to +0.4%), prefill +3.5% to +7%. Mainline #23940 (MERGED 2026-07-03, present by date): removes redundant CUDA copies after GDN.
- Mainline #28068 (MERGED 2026-09-06): GDN q/k normalization changed from `x / max(sqrt(sum x^2), eps)` to `x * rsqrt(sum x^2 + eps)`. A quality fix (mean KLD 0.001769 -> 0.001750 on Qwen3.8-27B Q4_K_M), not a speed fix. Whether prism has it is not confirmed: `src/models/qwen35.cpp:538` still calls `ggml_l2_norm`; mainline switched to a `build_gdn_l2_norm` helper; I did not read the op body.
- Context checkpoints: `--ctx-checkpoints` present in prism. Mainline #28302 (MERGED 2026-09-10, absent by date) fixes checkpoint eviction on short prompts. Bonsai-demo PR #161 (OPEN) documents that hybrid models still reuse context checkpoints even though `--cache-reuse` chunk shifting is unsupported.
- Mainline issue #19864 (CLOSED 2026-02-24, fixed by #19866, present by date): qwen35 graph splits placed between layers with RS cache caused severe prompt-processing loss on multi-GPU. Single GPU, so not relevant here.

---

## Section 2. Search areas that turned up nothing

- **x86 AVX2 kernels upstream for TQ1_0 / TQ2_0 / Q2_0 / BitNet I2_S:** none.
  - Mainline #26348 (https://github.com/ggml-org/llama.cpp/pull/26348, OPEN, 2026-07-30) is VNNI-only: "3x speed improvement for VNNI-compatible CPUs".
  - Mainline #29077 (https://github.com/ggml-org/llama.cpp/pull/29077, OPEN, 2026-09-18, third-party attempt to upstream PQ2_0/PTQ1_0 with a reference vec_dot at "~0.4 tok/s for the 27B on 16 threads"): CISC said "Please leave this for PrismML to submit themselves", and khosravipasha replied "PTQ1_0 and PQ2_0 might stay in fork only since adding new types is increased maintenance work".
  - #28452 (TQ2_0 llamafile_sgemm, CLOSED same day): prefill only, ~1.10x.
  - #27851 (OPEN, tiled mul_mat for k-quants): "3-7x" for tall batches, "80% of stock performance at pure GEMV". k-quants only, not IQ types; the author says the IQ types are probably feasible.
- T-MAC / LUT integrations: none.
- `nchunk` / mul_mat chunk tuning: none.
- Hyperthreading guidance: only pl752's note in fork #29 (item 6).
- `GGML_SCHED_MAX_COPIES`, per-split synchronization reduction, CUDA graphs with partial offload: none beyond #21067 and #28414.
- MoE CPU kernel speedups for IQ2/IQ3 on AVX2: none.
- CUDA 12.0 vs newer toolkit performance on Turing: none. Demo PR #171: prebuilt binaries carry only PTX for sm_75, so they JIT on first run; "`cuobjdump --list-ptx` is the check".
- Flash-attention slowness on cc 7.5 without tensor cores: none found.
- The "61-virtual + FORCE_MMQ" advice: not found in prism `docs/build.md`, `README.md`, or Bonsai-demo `AGENTS.md`/`README.md`. `docs/build.md:300` has only the generic `GGML_CUDA_FORCE_MMQ` row. Note that Known bugs 1 argues against `61-virtual`.
- Turing data point: demo PR #171 (RTX 2080 8 GB, PTQ1_0 27B, fully offloaded): "PP256 270-296 t/s, TG64 25.3-25.7 t/s"; "`-DCMAKE_CUDA_ARCHITECTURES=75-real` (145 sm_75 cubins, same speed within noise)"; CUDA 25.7 t/s vs Vulkan 2.6 t/s on the same card.

## Section 3. Does PrismML accept outside PRs?

Yes.

- Merged external PRs: #72 (thtro, AVX-VNNI Q2_0, opened 2026-07-15, merged 2026-07-17), #88 (thadreber-web), #81 (cnndabbler), #33/#34 (pl752, ARM NEON), #32 (Vort3xed, Vulkan Q2_0).
- Maintainer on #95: "happy to take similar PRs if find optimizations."
- On #76: PRs written before the August rebase onto mainline (`prism-v5` -> `prism-v7` -> `prism`) had to be re-applied on today's `prism`.
- Open external PRs from the last 48 hours: #181 (PTQ1_0 AVX-VNNI), #187/#188 (Vulkan PTQ1_0), #190 (repack null guard), #200 (HIP), #205 (the MTP Hadamard fix).
- khosravipasha routinely posts reproduction benchmarks on PRs.
- Nobody has submitted an AVX2 PQ2_0 `vec_dot`. #181 is the natural neighbor for yours.
- Upstream plan per khosravipasha in ggml-org#29077: "Our goal is to support new changes (hadamard + sign flips) with official Q2_0"; first piece merged as ggml-org#27779 ("ggml-cpu: add F16 input to the FWHT").

---

## Section 4. Known bugs to avoid

1. **`-DCMAKE_CUDA_ARCHITECTURES=61-virtual` + MoE `MUL_MAT_ID` on the 1650 SUPER crashes.**
   - ggml-org#24064 (https://github.com/ggml-org/llama.cpp/issues/24064), CLOSED stale 2026-08-01, not fixed.
   - Cause: "A binary compiled only for SM61 can still run on an SM75 GPU through PTX JIT. In this case, the host-side selector detects the runtime SM75 device and allows physical ubatch sizes up to 8, while the device-side `__launch_bounds__` value was computed when PTX was compiled for SM61." Reproduced on GTX 1660 Ti ("PASS: ne2 = 1..4; FAIL: ne2 = 5..8; PASS: ne2 = 9..16" for Q8_0) and GTX 1660 SUPER.
   - Verified present in prism: host `get_mmvq_mmid_max_batch(type, cc)` at `ggml/src/ggml-cuda/mmvq.cu:260` uses the runtime cc; device `get_mmvq_mmid_max_batch_for_device()` at `:386` uses compile-time `__CUDA_ARCH__`.
   - Batch limits by table:

     | Type | Turing+ (host) | Pascal and older (device) |
     |---|---|---|
     | IQ2_S | 7 | 4 |
     | IQ3_S | 6 | 4 |
     | IQ3_XXS | 7 | 4 |
     | IQ2_XS | 8 (default) | 5 |
     | IQ2_XXS | 8 (default) | 5 |
     | Q2_K | 7 | 4 |
     | Q3_K | 5 | 4 |
     | Q4_K | 8 (default) | 5 |
     | Q6_K | 8 (default) | 4 |
     | Q8_0 | 8 (default) | 4 |
     | MXFP4 | 7 | 4 |

     IQ1_S and IQ1_M fall through to the Turing default of 8 against a Pascal limit of 6.
   - Symptom: `ggml_cuda_compute_forward: MUL_MAT_ID failed / CUDA error: invalid argument` for GPU-side expert batches between the two limits. With `GGML_OP_OFFLOAD_MIN_BATCH=2`, that is exactly IQ2_M/IQ3_M MTP verify batches of 5-7, and any prompt whose last physical ubatch has 5-8 tokens.
   - Upstream fix attempts: #24547 CLOSED (JohannesGaessler: "would fix correctness for some JIT cases but would break the kernel tuning more generally"); #28578 (https://github.com/ggml-org/llama.cpp/pull/28578, OPEN, uses `ggml_cuda_highest_compiled_arch(cc)`); a fork fix at https://github.com/ut-slayer/llama.cpp/tree/dp4a-fix-mmid-max-batch.
   - Remedy: build `75-real` (or `61;75`).
2. **CPU_REPACK segfault loading PQ2_0.** Fork issues #180 (https://github.com/PrismML-Eng/llama.cpp/issues/180) and #204, fork PR #190; all OPEN.
   - Root cause (Mattwmaster58): "`tensor->extra == NULL` on a tensor named `prism.hadamard.1024`… the Hadamard setup inherits the weight's buffer type for its F32 helper tensors… The repack buffer's `init_tensor` then stores `nullptr` as `extra` (F32 has no repack type), and `set_tensor` dereferences it unconditionally." A second crash follows in `ggml_backend_tensor_copy` (null `get/cpy_tensor`).
   - Fix (baogorek, validated on CPU, Vulkan and HIP by c2p-cmd), in `src/llama-model.cpp` right after `buft` is taken from `weight->buffer`:
     ```cpp
     // Transform tables are F32, so they cannot use quantized CPU repack buffers.
     ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
     if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
         buft = ggml_backend_dev_buffer_type(dev);
     }
     ```
   - Workaround: `--no-repack` / `LLAMA_ARG_REPACK=0`.
   - Latent for you: PQ2_0 repack is only selected on AVX512-VNNI hosts, so it appears as soon as an AVX2 repack path exists.
3. **`--spec-type ngram-*` silently no-ops on Ternary-Bonsai-2.** Fork issue #203 (https://github.com/PrismML-Eng/llama.cpp/issues/203), OPEN.
   - Metal (M4 Max, build 9a9394a): "`timings.draft_n == null`, decode ~31.7 tok/s (identical to no-spec baseline)", no log lines.
   - CUDA (RTX 4080S, b10683): "72.2 / 71.8 / 72.7 t/s… versus 72.7 t/s with `--spec-type none`".
   - Root cause not identified in the thread and not determined here. Mainline #26499's `LLAMA_N_RS_SEQ` makes ngram-mod usable on qwen35moe upstream, and `need_n_rs_seq()` returning 0 for ngram types is a likely contributor (inferred).
4. **A failed drafter load silently downgrades into the checkpoint path.** Fork #173 comments: "`llama_model_load: error loading model: unknown model architecture: 'dspark'` / `speculative decoding will use checkpoints (context does not support partial sequence removal)`… The drafter failed to load, and the run did not stop." That path "`continue`s before `n_drafted`/`n_accept` are incremented… the rate read 100% by construction. Any historical 100% figure from a hybrid model is that artifact." Grep your startup log for `will use checkpoints`.
5. **Greedy output is not bit-identical under draft-model speculation on quantized targets.** ggml-org#25618 (https://github.com/ggml-org/llama.cpp/issues/25618), OPEN.
   - "Divergence starts immediately at width 2 (`--spec-draft-n-max 1`)… At width 2, 57 of 75 requests differed from the non-speculative baseline."
   - Root cause per the thread: batch-invariance violation. A batched verify row is not bit-exact to a single-token decode, so a 1-ULP difference flips argmax at near-ties. Reproduced on CUDA with IQ3_S and Q8_K_XL, and on Metal and Vulkan. ngram paths stay lossless.
   - Do not use byte-identity against non-spec output as a correctness gate.
6. **`draft-mtp` acceptance collapses to exactly 0.0 under `-np > 1`.** ggml-org#27572 (OPEN): "race between the async device->host copy of `t_h_nextn`… and any later graph that reuses the `t_h_nextn` extra buffer." "55.41% acceptance at `-np 1` collapsing to 0.00000 (0 of 136,713 draft tokens) at `-np 2`." Reproduced on CUDA (dual 4090). Proposed fix: `ggml_backend_sched_synchronize(sched.get())` in `llama_context::process_ubatch` before `res->set_inputs(&ubatch)`; superseded by open PR #27311. Stay on `-np 1`.
7. **ngram-mod combined with the Gemma4 MTP code: 40 -> ~4-5 tok/s.** ggml-org#24266, closed stale.
   - Commenter: "Using MTP below Q8 significantly reduces draft-acceptance rates — Gemma is very sensitive to MTP quality"; `--spec-draft-n-max 1` was fastest for several users.
   - Counter-example: another commenter on a Vega 8 iGPU found a Q4 MTP file faster than Q8 with no lower acceptance. Test both drafter quants for model B.
   - ggml-org#24596 (closed stale): Gemma4 MTP performance regressed after merge relative to the PR branch.
8. **`draft-mtp` CUDA vs Vulkan acceptance gap.** ggml-org#26750, OPEN: CUDA 35.8% vs Vulkan 92.2% on Qwen3.5-9B-MTP. Treat with caution: the confirming commenter later retracted their own measurements (bare `/completion` prompt without a chat template produced a repetition loop), and the original report is otherwise unconfirmed. Measure acceptance only through the chat endpoint.
9. **Multi-GPU only, listed so it is not rediscovered.** #26827 / #28252: whole-host lock during MTP prefill with tensor split. #27428: `draft-mtp` halves prompt processing on multi-GPU layer split. #28414: multi-GPU corruption.
10. **DSpark specifics.** n-max must equal block size 4 or it crashes at the first round; silent fallback above 32k context; acceptance decays with depth (demo issue #105); `--kv-mean-center` requires `LLAMA_ATTN_ROT_DISABLE=1` or context init fails with a misleading fit-params message.
11. **#28739 out-of-bounds read with a small `GGML_OP_OFFLOAD_MIN_BATCH`.** MERGED 2026-09-11, ABSENT in prism. No early exit for an empty `ids` tensor in `ggml-backend.cpp` (grepped around line 1646). Reported on `qwen4exp` only; whether `qwen35moe` or `gemma4` graphs produce zero-sized ids was not checked.
12. **Other fork reports, for awareness.**
    - #191: q5_0 KV cache ~8x slower than f16/q8_0/q4_0 on long-context decode (CUDA).
    - #189: open PR for a short-query MMA FA crash with quantized V.
    - Bonsai-demo #145: Q4 KV causes a 3x-67x prompt-processing slowdown on mainline CUDA for Ternary-Bonsai-27B.
    - #199: erratic decode with prebuilt tarballs on RTX 5090 with driver 570.

---

## Section 5. Verified vs inferred

**Verified by reading code at `prism@9a9394a`**

- Merge base 2026-08-25, 431 commits behind mainline.
- `need_n_rs_seq()` returns `draft.n_max` and includes DRAFT_SIMPLE (`common/common.h:387-392`).
- Recurrent-state dtype hardcoded F32/F32 (`src/llama-model.cpp:2774-2775`).
- Checkpoint fallback when `draft.size() > llama_n_rs_seq(ctx_tgt)` (`tools/server/server-context.cpp:2956-2957`).
- `--load-mode none` gives `use_mmap = false`, so the host-buffer downgrade does not fire (`src/llama-model-loader.cpp:562, 1224`).
- PQ2_0 repack gated on AVX512 + AVX512-VNNI (`ggml/src/ggml-cpu/repack.cpp:5352`; `arch/x86/repack.cpp:6586, 6643`).
- `ggml_vec_dot_pq2_0_q8_0` has VNNI and scalar paths only (`arch/x86/quants.c:561-578`); no x86 SIMD for PTQ1_0.
- `is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt` (`common/speculative.cpp:2129`).
- No Hadamard inverse in `graph_mtp` (`src/models/qwen35.cpp:634-640`).
- Absent flags and features: `--n-cpu-ffn`, `--prefetch-weights`, `LLAMA_N_RS_SEQ`, `llama_memory_has_shared_cells`, the second MTP graph-result arena (#28549).
- Present: `GGML_OP_OFFLOAD_MIN_BATCH` (`ggml-cuda.cu:5811`), `--no-op-offload`, `--cpu-strict`, `--poll`, `--prio`, `--ctx-checkpoints`, `--no-repack`, `--no-host`, `--fit-target`, `--direct-io`, `--load-mode`, the full `--spec-draft-*` and `--spec-ngram-*` flag set, spec types `draft-mtp`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`, the `gemma4-assistant` arch.
- `llm_arch_supports_rs_rollback` covers QWEN35 and QWEN35MOE (`src/llama-arch.cpp:1098`).
- MMVQ host vs device table mismatch (`ggml/src/ggml-cuda/mmvq.cu:125-163, 260, 386`).
- Only `KMP_BLOCKTIME` is set for OpenMP (`ggml-cpu.c:3892`).
- The CPU FWHT parallelizes over rows (`ops.cpp:11890`).

**Verified by reading threads:** every quoted number and quotation above.

**Inferred (my reasoning, not stated by a source)**

- The PTQ1_0 swap yields ~18% less PCIe volume per pass and a roughly proportional spec-mode gain.
- Pinned status deduced from 8.8 GB/s.
- F16 rs state is worth about 2 resident layers.
- A smaller `-ub` relieves the `cublasCreate` OOM.
- Item 1's VRAM arithmetic, and that it helps mainly non-spec decode.
- AVX2 repack is de-prioritized in spec mode.
- The single-thread FWHT and the sign-multiply attribution for `ggml_compute_forward_mul`.
- A separate drafter or DSpark loses on 4 GB.
- #28549 is small for this box.
- `need_n_rs_seq() == 0` contributes to fork #203.
- "Present by date" for mainline PRs merged before 2026-08-25 that were not individually grepped (#25749, #24025, #23940, #27138, #19866).

**Not checked**

- Whether #28068's normalization is effectively in prism.
- The root cause of fork #203.
- Whether `qwen35moe` / `gemma4` graphs produce zero-sized `ids` (relevance of #28739).
- Whether your grafted MTP head uses the post-norm hidden-state convention of #24025.
- The collapsed benchmark sections of #27210 and #27694.

**Raw thread dumps (local scratch, not durable):** `/tmp/lcpp/` — `moe.txt`, `moe27861.txt`, `disc24528.txt`, `spec.txt`, `cpu_off.txt`, `issues1.txt`, `issues2.txt`, `fork_issues.txt`, `fork_prs2.txt`, `demo_prs.txt`, `spec.md`, and the `prism_*` / `p2_*` source snapshots.
