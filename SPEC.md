# local-ai program spec: n-gram x MTP x expert-cache co-design for MoE decode on a 4 GB GPU

Version 1, 2026-09-19. Owner: Andrei. Compute partner: Dave (davetha).
This file is the PLAN. `notes.md` is the STATE (measurements, incidents, resume block). `bench/LEDGER.md` is the DATA.
If the three disagree: the ledger wins over notes, notes win over this spec, and this spec gets corrected in the same commit.

Tags used throughout: **[M]** measured by us (ledger row exists), **[V]** verified by us in source or a primary document,
**[L]** lead from a scout or paper, not yet reproduced, **[I]** inferred, **[D]** dead, with the reason and the reopen criterion.

---

## 0. How to use this document (the anti-getting-lost rules)

1. Every unit of work has an ID (`U2`, `N3`, `T1`...). Commits, ledger labels, notes entries and box scripts reference the ID.
2. Nothing gets queued on the box or started on Dave's rig without three lines written first: **hypothesis, measurement, kill criterion**.
3. Box work is queued ONLY through `/ai/bench/queue.sh add ID 'CMD'` (file-backed, survives poweroff, GPU-idle gated); no more tmux waiter chains. WIP limits: one job stream on the i5 box (it is a serial machine: 16 GB RAM, one GPU), one code stream on the Mac, at most one scout in flight. New ideas go to section 12 (parking lot), not into the queue.
4. The status board (section 1) is updated at every checkpoint. A checkpoint = ledger synced, `notes.md` updated, board updated, commit pushed.
5. Every third checkpoint: forensic re-read of this spec against what was actually built (intent vs implementation drift), findings recorded in section 11.
6. Nothing is "done" until it is measured on the box with the full telemetry row AND the quality gate (section 8) passes AND an adversarial review of the diff/result found nothing above low severity.
7. Scout output (Sonnet agents, the Qwen pi scout on Dave's rig) is a lead until checked against source or the paper. Cost rule: scouts never run on the top-tier model.
8. **Accuracy first (Andrei, 2026-09-20: "not faster at getting it wrong").** A quality-affecting lever ships only when it PROVES non-inferiority (section 8.x: paired per-question test, KL divergence vs the Q6 reference, hard sets with thinking on). Lossless levers lead. Ties go to the higher-precision file.
9. **New code preempts benchmarks (Andrei, 2026-09-21).** Build + verify of new code go to the FRONT of the queue as serialized jobs; an unrelated benchmark in flight is stopped (`systemctl stop ai-queue`) unless it is minutes from done or feeds the verification. NOTHING runs beside a benchmark (no compile, test or profile "because it is only CPU"): a row that shared the box is marked contaminated in `notes.md` and rerun.
10. The Qwen coder gets rote, tightly fenced briefs (explicit do-NOT list, 3 attempts then BLOCKED, commit per part, no /tmp debug scripts) and never has to type the literal chat special-token strings: they end its own generation. Review findings go back to it as a numbered brief.

## 1. Status board

### 1.0 In flight right now (2026-09-21) — the one table to read first
Box jobs run strictly in this order (`/ai/bench/queue.sh list`; durable, resumes after a poweroff; one job at a time, box exclusive). Every job script in `bench/box/` opens with its hypothesis and kill criterion; results land in `bench/box/ledger.jsonl` / `bench/qual/results/` and are rendered by `bench/jobreport.py`, `bench/ctxreport.py`, `bench/qual/paired.py` (and `bench/qreport.py` once Qwen's brief 23 lands).

| # | box job | work ID | question it answers | decision it feeds |
|---|---|---|---|---|
| 1 | `cpu1bench` (3 min) | CPU1 | does ggml's CPU backend thread the expert FFN (real shapes: Q2_K 2048->512 x2, Q3_K 512->2048) well, and what would a task-parallel layout (a thread owns whole experts, one barrier per layer) buy | write the CPU1 kernel or not |
| 2 | `whittle1` (~3 h + 13 GB download) | S4 / DM1 | first contact with the dense-27B-distilled-to-A3B model: fit, t/s, kq1h quality row | whether it gets an `hq1` arm |
| 3 | `hq1_iq2m` (last ~11 AIME items; AIME capped at 15) | HQ1 | so far: **MATH-L5 57.5% (23/40), 16 of 40 chains cut at the card's own 32k budget, none a loop; 23 of 24 finished are right; EvalPlus 85% (35/41)** | everything about the quant |
| 4 | `dl_stock` | HQ1 | stock Qwen3.6 bartowski IQ2_M (12.96 GB) = the control; moves 2 closed-decision models to `/mnt/md0/models-cold` first | - |
| 5 | `dense1` (~15 min) | DM1 | FFN neuron-energy concentration of dense Qwen3.8-27B | static hot / cold neuron split: dead or probe further |
| 6 | `hq1_stock` (~16 h) | HQ1 | same items / sampler / seeds on stock weights: is the overthinking the 2.5-bit quant or the fine-tune | which model file we serve |
| 7 | `qx3` (~4 h) | QX | KLD of all-Q4_K experts (ceiling = price tag of RAM1) and of hot-25% / hot-50% mixes vs K2 0.218 | runtime hot / cold precision split: GO needs >= 25% lower mean KLD |
| 8-11 | `lq1`, `hq1_k2`, `bonsai1_easy`, `bonsai1_hard` | LQ1 / HQ1 / BON1 | KV type at depth; K2 on the hard sets; Bonsai accuracy | - |
| next to write | `lq2` = retrieval A/B with `GGML_CUDA_FA_VEC_KROW` 0/1; an `hq1` arm for Whittle if it passes; "race" decoding (3 seeded chains in one batch, first finisher) as an hq1 variant; CPU1 kernel if `cpu1bench` says GO | FA4 / S4 / HQ1 / CPU1 | | |

Done since the last board (2026-09-21, numbers in `notes.md`): `fa1` run 2, `arch1`, `kld1`, `ctx1c`, `ctx1d`, `prof2`, `prof3` (blind: nsys 2022.4 cannot see kernels inside CUDA graphs), `kq1graft`, `prof4`, `fa3` (dead), `fa4`. Box fixes: the queue never autostarted because of a **systemd ordering cycle** (ai-queue After ai-perf-tweaks After multi-user.target; fixed, unproven until the next boot); power-profiles-daemon kept resetting the governor (the tweak unit now sets the `performance` profile through it); `systemctl stop` no longer marks the running job `failed`.

Qwen coder (worktree `~/dev/local-ai-qwen`, branch `qwen/work`): queues 1-7 merged (slotclient + presave/extend, ctxreport, longctx, slotlib, paired, gguf_types); **queue 8 in flight**: 23 `qreport` -> 24 `expert_mix` (hot/cold emulation, gates QX) -> 22 `ctxproxy2` (token-exact proxy). Brief 16's `ctxproxy.py` stays unmerged (6/11 tests, superseded by 22).

Verdicts since the last board (details in `notes.md`, reports in `bench/reports/`): `lat1` -ub is the prefill lever (2.4x), rest inert. `mtp1` in-model head = -520 MiB, +11% at c46. `kq1` K2 experts +15.6% decode, `kq1h` + paired: **K2 vs IQ2_M at n=521: wins 26 / loses 22, +0.8 [-1.8, +3.4] = NON-INFERIOR** on the (saturated) standard sets. `r1` / `r1b` / `r1c`: routing bias b1.0 clean at c48 but GSM8K 98 -> 92 at the shipped c30 + MTP, b2.0 -10 => **R1 OFF**. `mainqual` in band => mainline is the tree. `h2qual`: Gemma IQ3_M stays (Q2_K_P -1.7 [-4.4, +0.9], unproven). `orf1` 100 / 100 / 48 (stock contrast). `ctx1`: 32k prefill 75.7 / 107.6 / 135.4 t/s at ub 512 / 1024 / 2048; decode 27.8 @12k, 28.1 @27k, 11.5 @120k; `-nkvo` dead; its restore rows tested the wrong thing. `ctx1b`: **restore + token-exact extension WORKS: 32k in 3.1 s vs 201 s, 131k in 6.6 s vs 35 min (320x), cross-config (prefill server -> decode server), coherent replies.**

Done tonight, verdicts in the rows below: S2, S3, U1, N1, N2, P1/G4, P2, `p4c1` (P4c flat, N2-lite +4.5% at n=1, V1 ~+2%). Mac-side backlog, one at a time, in this order: **M1** harness noise (3+ repeats, longer GEN, drop first run; 3-5% spread today is as large as the levers) -> repeat N2-lite -> P4c default off / cheaper observation -> **C1** long context (`ctx1`: decode at depth 32k-128k, KV on GPU vs `-nkvo`, slot save/restore to NVMe, TTFT on a repeated prompt; needs L1's prefill answer) -> add `mtp1`/`r1`/`ctx1` to the `jobreport.py` registry -> Phase 1 decisions from the ledger -> forensic re-read (rule 5). Helpers: the Qwen pi agent codes bounded Python with tests (tasks 1-7 done, reports in `research/code*-report.md`); Sonnet agents review diffs and scout papers. Rule 3's WIP limit was bent tonight: six measurement ideas went straight into the queue instead of the parking lot. They are all zero-or-small-code, each has its three lines, and the box is a serial machine either way, but nothing else gets queued until this table drains to `kq1`.


| ID | Item | State | Gate / next action |
|---|---|---|---|
| H1 | Harness: preflight, mon, ledger, quality, death-aware waiters | done [M] | keep |
| H2 | Workload set W1-W4 + larger-n quality set | **done [M]**: W1-W4, `data_h2` (GSM8K-200 + MMLU-Pro-280), `h2qual` + `kq1h` run | superseded as a GATE by section 8.x: the standard sets are saturated (GSM8K 95-100%), see HQ1 / KLD1 |
| Q-box | Box queue (`/ai/bench/queue.tsv` + `ai-queue.service`, enabled at boot, `Restart=no`) | running; order in 1.0 | Stop a job ONLY with `systemctl stop ai-queue` (never pkill). Reorder = rewrite the pending rows under `queue.lock` with a row-count check. Waiters: `bench/box/watch.sh queue|job` (drain / stall / failure / stale-log aware), one-shot, never per-row streams |
| S1 | Gemma default file | **IQ3_M stays [M]** (accuracy-first): paired n=521 Q2_K_P -1.7 [-4.4, +0.9], Q3_K_M -0.8 [-3.0, +1.5] = not proven non-inferior; speeds 25.1 / 31.1 / 38.5 t/s | reopen with KLD vs a Gemma reference + a hard-set arm |
| S2 | Qwen3.8-35B-A3B-Distill vs Qwen3.6-35B-A3B | **done [M]: KEEP Qwen3.6** | distill regresses HumanEval -12 / GSM8K -8 (> 1 sigma), has no standalone MTP head, 39.1 vs 46.3 tok/s. Reopen only if the clean Qwen3.6 MMLU-Pro (`mmlu2k`) lands far below the distill's 72.9 |
| S3 | Bonsai PQ2_0 vs stock Qwen3.8-27B IQ3_M (dense reference) | **done [M]: dense is dead here** | stock 27B = 1.38 tok/s (`-ngl 16`); Bonsai 5.15 tok/s, quality run dropped by Andrei. No quality rows; fix the `-ot` FFN regex before any future dense bench |
| S4 | **Whittle-Qwen-3.8-35B-A3B** = dense Qwen3.8-27B DISTILLED into an A3B MoE body (8 of 180 routed experts + shared, ~3B active) + 10B hashed n-gram memory; `qwen4exp` (in our mainline tree [V]). The only existing form of "dense model run as MoE" for the Qwen3 family (DM1) | **queued: `whittle1`** — mradermacher i1-Q2_K, 12,968,342,080 bytes [V, HF API] = the only quant that fits 16 GB (Q3_K_M 16.7 GB needs RAM1). Load / placement (big non-expert tensors to the CPU by name), decode t/s at cache 0 / 16, then the kq1h row (GSM8K-200 + MMLU-Pro-280) vs Qwen3.6 IQ2_M 96.0 / 71.4 and K2 95.5 / 73.6 | kill: > 2 sigma under those rows or < 30 t/s; pass => an `hq1` arm. Caveat: plain Q2_K is a blunter quant than IQ2_M / K2, a weak row may be the quant. Self-reported GSM8K 88 / MATH 77 are one author's [L] |
| DM1 | **Dense model "as MoE"** (Andrei 2026-09-21): contextually select FFN neurons of a dense model, hot on the GPU, cold on the CPU | **do-it-ourselves routes: dead / parked [L, scout `research/scout-2026-09-21-dense-as-moe.md`]**: training-free activation sparsity on SiLU gates (TEAL / WINA / R-Sparse) = 40-50%, lossy, < 2x on the FFN only, and every published speedup assumes the whole model on one GPU; nobody ships a hot-GPU / cold-CPU neuron offloader for SiLU models; PowerInfer needs ReLU-fied models (none for Qwen3; team moved on); a STATIC hot / cold split loses ~40% vs an oracle (about half of the hot neurons change per token); no dense-to-MoE checkpoints exist for the Qwen3 family; ExpertWeaver (training-free dense-to-MoE) has no code. Our A3B MoE is the trained form of the idea (11x sparsity learned end to end). **Alive: S4 (someone already distilled the dense 27B into an A3B)** and one zero-C++ probe: `dense1` (llama-imatrix on Qwen3.8-27B, `bench/imx_heat.py` reads per-neuron energy from the ffn_down input statistics) | FLAT verdict (top 20% of neurons < 60% of energy) closes it; CONCENTRATED earns a per-token probe. Sparsity fine-tunes (ReLUfication) = Dave's box, behind T0 |
| U1 | Mainline build of `moe-cache` on the box | **done [M]** | linked clean at CUDA arch 75 (`/ai/src/llama.cpp-mainline`) |
| U2 | Fork vs mainline A/B | **DECIDED 2026-09-20 (Andrei): mainline + our patches is the default tree** (mainqual in band of the fork rows). **G2 PASS on speed [M]** (+5-7%, VRAM flat); Qwen text bit-identical, Gemma code prompt flips a greedy tie after ~10 tokens | `mainqual` quality rows within 1 sigma of the fork rows => mainline is the default tree |
| U3 | Split the cumulative mainline diff into a reviewable series | todo | after U2 |
| U4 | Upstream candidates | todo | needs Andrei's explicit go (a fork of a public repo is public) |
| N1 | Hybrid `ngram-mod,draft-mtp` bench | **done [M]** | G3 PASS (workload-shaped): keep MTP n=2, n-gram ON, cap Qwen<=2 / Gemma long; +3.8% Qwen edit, ~free Gemma, neutral code/reason |
| N2 | Decouple `n_rs_seq` from the model drafter for n-gram drafts | **long form dead on 4 GB [M]; N2-lite (cap 3) optional** | each `n_rs_seq` unit = **62.81 MiB VRAM** on Qwen3.6 [M, box logs: 62.81 / 188.44 / 251.25 MiB at 0 / 2 / 3]; 16 units = +880 MiB = ~22 cache slots. Only `n_rs_seq` 3 (+63 MiB, ~1.6 slots) is affordable; expected ~+1% on edit. See 6.2 |
| N3 | Persistent, offline-seeded n-gram continuation store ("rainbow table") | **demoted [M]**, design in section 6 | ceiling is the draft time it replaces: 5.4 ms of a 58 ms Qwen round (9%), and only in rounds where the table drafts (70 of 321 on the edit workload). Build only after KQ1/P2, and only if copy-heavy traffic is where we live |
| KQ1 | **K-quant experts for Qwen3.6** (HauhauCS Q6_K_P -> gate/up Q2_K, down Q3_K, 12.9 GB) | **speed [M], n=1**: decode ALL 46.1 -> 53.3 (+15.6%) best-config, no-cache +21%. **Quality: paired n=521 vs IQ2_M = wins 26 / loses 22, +0.8 [-1.8, +3.4] => NON-INFERIOR** on the standard sets (GSM8K-200 95.5 vs 96.0, MMLU-Pro-280 73.6 vs 71.4) | remaining before it can be the default: `kld1` (fidelity to Q6), `hq1 k2` (hard reasoning), Andrei's spot-check. Its imatrix came from the IQ2_M model => QX1 |
| R1 | **Cache-aware routing** (`--moe-expert-cache-bias B`) | **OFF [M] (accuracy-first)**: c48 no-MTP b1.0 +8.9% decode with GSM8K 98.0 / HE 97.6 vs 96.0 / 92.7, but at the SHIPPED footing (c30 + MTP) b1.0 = GSM8K 92.0 / HE 90.2 vs 98.0 / 92.7 (the bias bites harder on a smaller cache); b2.0 GSM8K -10. b0.5 (+17%, 100 / 92.7) unproven | no more box time unless a ~1000-question paired run is judged worth it. Flag stays in the tree, default 0 |
| M1 | Harness noise: identical repeats differ by 3-5%, as large as the levers | todo (Mac) for SPEED rows: 3+ repeats per arm, longer GEN, drop the first run | quality side solved differently: paired per-question test (`bench/qual/paired.py`) + KL divergence |
| C1 | **Long context to 262k** (native window of Qwen3.6, no rope stretch): KV 20.5 KB/token F16 (10 attention layers, 2 KV heads x 256), 62.8 MiB recurrent state | **mechanism PROVEN [M]** (`ctx1b`): TWO-PHASE = prefill server (expert cache 0, big ubatch: 135.9 t/s at 32k, 56.8 at 131k) -> slot file on NVMe (723 MiB at 131k, 0.4 s) -> decode server (cache 24, ub 128) restores and continues in 3-7 s. Rules learned: (1) a GDN hybrid only re-uses a restored slot when the new request's TOKEN IDS extend the saved ids (token-id arrays on `/completion`; re-rendered chat text forces a full re-prefill); (2) the standalone `-md` MTP head is blind after a restore (own empty KV): 17.2 vs 28.6 t/s; (3) 262k needs cache 0 on the prefill side (926 MiB compute buffer at ub 512). `attn-rot` (mainline #21038) is active in our build, `FA_ALL_QUANTS=ON` | open: 262k rung, `ctx1c`, `lq1` (KV type by accuracy), decode at depth (FA1), product layer C2 **262k verified end to end [M]** (`ctx1d`): restore of 239k tokens in 0.9 s, decode 8.7 t/s (11.0 with FA4), coherent replies. `ctx1c`: MTP acceptance is 38-44% at 27k depth even in the fresh presave run => MTP is a net loss at depth (18.5 vs 23.75 t/s), long-context decode runs WITHOUT speculation. Launch configs: prefill = `build6180`, cache 0, ub 2048-4096; decode = arch-75 FA build, `GGML_CUDA_FA_VEC_GQA=1 GGML_CUDA_FA_VEC_KROW=1`, cache 0 (262k) / 24 (131k), ub 128, q4_0 KV pending `lq1`. |
| ORF1 | Over-refusal compliance from published benign sets (OR-Bench-Hard-1K, XSTest safe) | **done [M]**: Qwen3.6 100%, Gemma Q2_K_P 100%, stock distill 48% | informational; stock distill is the contrast row |
| L1 | Latency / transfer levers from the telemetry (zero code) | **done [M]** (`lat1`): `-ub` is the prefill lever (38.9 -> 91.6 t/s at ub 512; fit T = 3.96 s + 5.4 ms/token per ubatch, asymptote ~184 t/s, measured 135 at ub 2048); costs ~6 cache slots + 3-5% decode => prefill and decode want DIFFERENT servers (C1 two-phase). CUDA graphs, pinning, OMP wait policy, `--prio`: inert or negative | - |
| V1c | **In-model MTP head** (graft blk.40 into the main GGUF, `--spec-type draft-mtp` without `-md`) | **works [M]** (`mtp1`, n=1): -520 MiB VRAM at equal slots, flat at c30, **+11% ALL at c46** (hit 53 -> 65%); temp-0 text differs from `-md` by greedy tie-breaks only, acceptance unchanged | promote = Andrei's call after `kq1graft` (K2 + in-model) and `ctx1c` (survives a restore?) `kq1graft` [M]: K2 + in-model head at equal slots (26) is 2-11% SLOWER than the separate head, but it frees 528 MiB; at 38 slots (same VRAM) it is +3-5% over the separate head (56.6 / 56.2 / 60.1 / 46.7 t/s code / reason / edit / long). Temp-0 text is not run-to-run stable at short context with the expert cache on (same config, two openings), so text identity across heads says nothing. |
| V1 | VRAM diet: MTP head experts to the CPU (`-otd exps=CPU`) | superseded by V1c [M] (~+2% in `p4c1`) | - |
| P1 | Token-n-gram -> expert predictability (`predict.py --shift`, `prefetch.py` on the trace2b files) | **done [M-sim]: G4 FAIL** | see P2 |
| P2 | Draft-driven expert prefetch | **dead [D]** | Qwen3.6, 30 slots. Code corpus: lookahead recall@30 68% vs LRU 57% (passes on recall), prose: 49% vs 65% (fails). Policy simulation with the real 2-step upload visibility is NEGATIVE on both (hit 60.2 -> 58.4-59.9 code, 60.1 -> 58.8 prose): what a token table predicts beyond the cache is single-use, and one upload (~1 MiB) costs 3-5 CPU miss pairs (~0.07 ms each). The cache earns through reuse only. **Reopen if** uploads get ~5x cheaper (pinned host experts + a dedicated copy stream) AND misses get costlier (they are about to get cheaper: KQ1), or a hidden-state predictor with multi-token horizon shows reuse |
| P3c | Cache graph beyond 4-token batches | **parked [M]** | miss cost scales with (token, expert) pairs (4.1), long drafts do not pay on Qwen (N2) and were flat on Gemma at 0.99 acceptance; reopen only if KQ1 makes misses bandwidth-bound |
| P4c | Prefill warm-up of the cache | **measured [M]: flat** (`bench/reports/p4c1.md`) | Qwen decode +0.8% inside 3.6% noise; Gemma decode -0.5%, wall -4.1% (observation node slows prefill). Default goes to off; reopen with a short-reply / topic-shift test and a cheaper observation (every Nth layer) |
| G-ACC | **Quality instruments** (section 8.x / 8.y) | built [M]: `paired.py` (McNemar + CI + non-inferiority verdict), hard sets in `qual.py` / `fetch.py --hard` (EvalPlus executor verified on the box), `longctx.py` + reuse guard, `kld1.sh`; `qreport.py` in Qwen's queue | every quality-affecting decision from here on |
| KLD1 | KL divergence vs the Q6_K_P reference (28.5 GiB, streams from NVMe; wikitext-2 test, 40 x 512); reference logits saved at `/ai/bench/kld/q6.kld` (5.07 GB) and reused by every later candidate | **done [M]**: IQ2_M mean KLD 0.2188, PPL ratio 1.180, same top token 79.8%; **K2 0.2181 / 1.165 / 79.9% => K2 is equal or better on fidelity too**. Both ~2.5-bit files are HEAVILY degraded vs Q6 (1 token in 5 changes its argmax) | needs a CODE corpus as a second text (hot sets are workload-specific); this is the screening rig for QX |
| HQ1 | Hard sets, thinking on: AIME 2024+2025 (60), MATH-500 L5 numeric (40), EvalPlus on our 41 HumanEval tasks. **Sampled per the Qwen3.6-35B-A3B card** (thinking: temp 1.0 / top-p 0.95 / top-k 20 / presence 1.5; code: temp 0.6 / presence 0), seed = hash of the item id (reproducible; both paired arms share seeds). Caps 32k (math_l5) / 8k (EvalPlus) / 41k (aime; the card allows 81k, F16 KV at `-c 49152` beside cache 24 does not). Each row keeps the reasoning tail + a repetition score | **running (`iq2m`)**; runs 1+2 void (greedy, off-card: every item hit the cap inside the reasoning block, repetition 0.00 = live chain, no loop) | read absolute AIME vs the card's full-precision score; `truncated` / `looping` say whether the cap or the model spoke; `k2` arm next |
| LQ1 | Retrieval at depth (RULER-style needles + variable tracking, one prefill per depth) x 5 KV arms to 131k, expert cache 0 | queued #9 | kill for q4_0: needle pct worse than f16 / q8 by more than 1 in 10 at any depth |
| BON1 | Ternary-Bonsai-2-27B (PrismML QAT ternary of Qwen3.8-27B, PQ2_0): accuracy redo. Speed was already tuned to the PCIe limit (5.15 t/s, FFN streamed + MTP n=2) | queued #10-11 | vendor claims (95% of full precision, AIME 87+ where IQ2_XXS gets 57.5) are unverified; paired vs the Qwen rows |
| FA1 | **GQA-aware quantized decode attention** (`bench/box/patches/fa-gqa-*.patch`): the CUDA `vec` FA kernel launched one block per QUERY head, so Qwen3.6's 16 q heads on 2 KV heads walked + dequantized every K/V row 8x per step. Head-group columns (ncols2 = 8) => one walk per KV head | **shipped, default ON [M]** (`fa1` run 2): 4079/4079 unit tests on every gate, **+10.3% decode at 120k** (12.7 -> 14.0 t/s), text identical over 64 tokens. My ~3x prediction was wrong: the kernel is compute bound, not dequant bound (see PROF) | `GGML_CUDA_FA_VEC_GQA=0` restores upstream |
| FA2 | Same kernel for 2-4 query tokens (MTP verify batches) instead of MMA_F16, which dequantizes the whole used KV to F16 every step. Plus opt-in F16 K/V on the GQA vec path (`GGML_CUDA_FA_VEC_GQA_F16=1`) | **default OFF [M]**: -8.5% pp3, -1.7% on the real MTP round (the MMA / tile kernels have better arithmetic intensity than the fine-grained vec kernel). What it does buy: no F16-KV reserve, compute buffer 698 -> 7 MiB at 262k with `-ub 4` | `GGML_CUDA_FA_VEC_GQA=2` when VRAM at 262k matters more than verify speed |
| NTC1 | **Runtime no-tensor-cores switch** (`ntc-*.patch`, `GGML_CUDA_NO_TENSOR_CORES=1`): attention DISPATCH only, through a separate predicate `turing_mma_dispatch_available()`. Run 1 also switched `turing_mma_available()` and crashed (illegal memory access in MUL_MAT): MMQ sizes its tiles from that predicate on the HOST while the DEVICE kernel is chosen at compile time by `__CUDA_ARCH__`. Rule: a runtime switch may only touch predicates that drive dispatch | **a wash [M]**, default off. Caveat: the measured build still carried a stale `mmq.cu` hunk (force MMQ under NTC) from the dropped patch; `fa1.sh` now resets that file | the real no-tensor-cores win is the BUILD (ARCH1), not a switch |
| ARCH1 | The banner's literal advice as a second build: `CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual` + `GGML_CUDA_FORCE_MMQ` (`build6180`) vs arch 75 | **done [M]: TWO BINARIES.** Pascal-path build = **prefill 2.4-3x** (pp512 336 vs 113 t/s; 32k real server 328 t/s at ub 2048, 363 at ub 4096 vs 136) but decode -8%, pp3 -24%. So: prefill server = Pascal build, decode server = arch-75 + FA patches; slot files restore ACROSS builds (`ctx1d`). The old "4 s fixed per ubatch = PCIe" reading was mostly the arch-75 MMQ tile path | open: one binary that picks Pascal-style MMQ for large batches on a flagged device (unscoped C++) |
| PROF | `prof2` (perf + nsys of the real K2 + cache 26 + MTP round), `prof3` / `prof4` (nsys at 120k from the kept slot) | **done [M]**. prof2 CPU: 48% of samples in libgomp spin, 19% q2_K dot, 13% q3_K dot. prof3 was blind (nsys 2022.4 does not trace kernels inside CUDA graphs); **prof4** (graphs off): `flash_attn_ext_vec` = 4.36 ms/layer upstream, 3.66 FA1 = **53% of GPU kernel time per token at 120k**; everything else is small. The kernel is COMPUTE bound (~1 G multiply-adds per layer per token on a ~4 TFLOPS card); the 0.4 ms bandwidth floor is not the binding one | profile with `GGML_CUDA_DISABLE_GRAPHS=1`; next profile after FA4 to split what is left |
| FA3 | V-skip: drop the V dequant + axpy of (position, head) pairs below 1e-8 of the running max | **dead [D]** (`fa3`): the data-dependent branches make the kernel 2x SLOWER (14.9 -> 7.7 t/s at 120k), and a free V walk would only be +23-26%. Its keepers: run-to-run logprob floor = exactly 0.0 on the kept slots; the K walk is ~2/3 of the attention kernel | - |
| FA4 | **Row-owning K walk** (`fa-krow.patch`, `GGML_CUDA_FA_VEC_KROW=1`): upstream splits every K.Q dot over 32 threads (8 elements each: a block-scale load, half->float and 2 float multiplies per 4-element group, a 5-step reduction per position x column). KROW: each thread owns a K row, loads it once for the 8 columns, exact integer block sums, one float multiply-add per block, one max-reduction per column per tile | **works [M]** (`fa4`): 4079/4079 unit tests; **decode 14.64 -> 17.1 t/s at 120k (+17%), 8.99 -> 10.97 at 239k (+22%)**. Logprob drift vs FA1 (mean 0.134 / 0.136, top-10, 96 tokens) = the drift of the ACCEPTED upstream->FA1 change (0.136 / 0.146): at this depth any float reordering moves a 2.5-bit model's logprobs that much, floor is 0.0 | set in the decode launch config now; flip the code default at the next build; `lq2` retrieval A/B before calling it proven at depth |
| FA5 | Bounded-error block-sparse decode attention: skip a 128-position KV tile when a per-tile upper bound proves every weight < thr of the max | **dead [D]** (`fa5probe`, real Q/K of all 10 layers at 120k, n=60): tiles under 1e-8 for a GQA group = **0.0% even for an oracle** (1e-6: 0.0%, 1e-4: 10.9%); Quest min/max, centroid+radius and un-rotated min/max bounds prove 0.0%. QK-norm compresses the logit range. Mass IS concentrated (top 1% of tiles = 63%, top 5% = 79%), but using that is lossy top-k attention | reopen only as an explicitly lossy mode with task-level proof (`lq`-style) per threshold |
| CPU1 | Task-parallel fused expert FFN on the CPU backend (a thread owns whole experts, one barrier per layer) | **dead [D]** (`cpu1bench`, real shapes / types): task-parallel t6 = ggml t6 (484 vs 489 us for 12 pairs); both stall at ~3.5x on 6 cores because the miss phase streams ~27 GB/s of expert bytes = **the DDR4-2667 ceiling**. The CPU phase is bandwidth bound at 6 threads (compute bound only at 1) | lever on the miss term = bytes per token only (hit rate, bits per expert). Open slack: ~0.3 ms per layer of CPU<->GPU handoff in the server (0.78 ms measured vs 0.49 bare) = up to ~12 ms of the 58 ms round; measure on the prof2 nsys timeline first (HAND1) |
| C2 | **Long-context product layer**: `ctxproxy2` (token-exact proxy: keeps the exact ids of every stored conversation, tokenizes only the new suffix, `/completion` with id arrays, OpenAI streaming, slot + id sidecar store with LRU disk budget) + a two-phase supervisor (prefill-config server for new long documents, decode-config server for serving) | proxy = Qwen brief 22 (queue 8); supervisor = design after the 262k + `ctx1c` numbers | brief 16's chat-text proxy would never hit on this model |
| QX | **Own quantization** ("our rainbow table": unlimited offline compute, consulted on every token). [M] routing: top 25% of experts carry 72-78% of the mass, but the hot set is workload-specific (code vs prose overlap = chance). Steps, each gated by KLD then paired: QX1 imatrix from the Q6 source with expert-balanced calibration (Dave's box); QX2 per-layer type recipe — replay Unsloth Dynamic's from its GGUF header (`bench/gguf_types.py`, merged) onto our Q6 source; QX3 hot/cold per-expert precision — FIRST emulated in one GGUF (`expert_mix.py`, Qwen brief 24) and priced by KLD, runtime C++ (split tensors + two MUL_MAT_ID with skip ids + expert-cache interplay, 2-3 sessions) only if mean KLD drops >= ~25% at equal size; QX4 ik_llama IQK types only if the FORMAT is the limit | accuracy answer for QX3: ~12-16 h after kld1 + brief 24 land | codebook / trellis formats only for GPU-resident tensors: they decode slowly on the CPU miss path |
| RAM1 | Hardware: 16 -> 32 GB DDR4 (~$50-70, board takes 128 GB) | proposal, Andrei's call | unlocks 3-4 bit expert files (16.7-20 GB do not fit today), the biggest accuracy lever per dollar if `hq1` shows a reasoning collapse; also page cache for slot files |
| MIG1 | Repo layout: flatten `bench/box/*` -> `bench/*` so the repo mirrors `/ai/bench` 1:1, then make `/ai` a git checkout (box identity + GitHub SSH already set) | parked, plan in `research/MIGRATION-flatten-bench.md` | only when the queue AND the Qwen worktree are idle |
| T0 | ROCm throughput spike on Dave's MI210 pair (1 h) | todo | gate for every T item |
| T1 | MTP head / drafter fine-tune with TV loss (self-distillation) | design in section 7 | after T0 |
| T2 | Router-only locality fine-tune (ReMoE-style) | design in section 7 | after T0 and P1 |
| T3 | Full logit KD dense 27B -> 35B-A3B | parked [L] | 15-193 days student + teacher-logit generation; see 7.4 |
| T5 | Continue the Whittle recipe (memory transfer + dependence loss + forward-KL) on Dave's box | lead [L] | after S4 shows the preview is worth continuing, and after T0; see 7.5 |
| T4 | Dense -> A3B/A4B conversion of Qwen3.8-27B | dead [D] | see 3.2 |
| F1 | Qwen3.8-Flash-Next (native n-gram table + MoE + MTP) | parked by Andrei | see 3.4 |
| W1 | Write-up / paper | outline in section 10 | after the ablation matrix is filled |

## 2. North star, constraints, metrics

**Goal (restated 2026-09-20).** Right answers first, then speed: the most accurate model + configuration this box can serve, and among configurations PROVEN non-inferior (section 8.x) the maximum decode tok/s. Long context to the native 262k. Hardware: GTX 1650 SUPER 4 GB (Turing, PCIe 3.0 x16), i5-10400F (6C/12T, AVX2), 16 GB DDR4, NVMe. "Every 5 tok/s counts."

**Where we are [M].**

| Model | Morning of 2026-09-19 | Now | Config |
|---|---|---|---|
| Qwen3.6-35B-A3B IQ2_M (served) | 28.0 | 46.3 (200-tok); 47.4 / 45.5 steady-state 600+ tok [M] | cache 30 + MTP head n=2, `-c 4096` |
| Qwen3.6 K2 experts (not served yet) | - | **53.3** ALL [M, n=1]; non-inferior on the standard sets at n=521 | cache 26 + MTP n=2 |
| Gemma4-26B-A4B IQ3_M (pick, accuracy-first) | 16.2 | 28.7 (29.7 MTP); 25.1 on quality prompts | cache 16 |
| Gemma4-26B-A4B Q3_K_M / Q2_K_P | - | 33.7 (37.2 drafter) / 40.2 (48.2 drafter; 50.5 / 48.3 steady-state) | cache 15 / 19 |
| Bonsai-27B PQ2_0 (dense, QAT ternary) | 0.71 | 5.15 (PCIe-bound); accuracy run queued | FFN streamed + MTP n=2 |
| Long context (Qwen3.6 IQ2_M) | - | decode 28.6 t/s @27k (F16 KV), 12.4 @120k (q4_0 KV); prefill 135.9 t/s @32k (ub 2048), 56.8 @131k (ub 1024); **restore + continue: 3.1 s @32k, 6.6 s @131k** | two-phase, section 1 C1 |

Steady-state (`steady1`) is higher than the 200-tok headline: the cache warms and the cold start amortizes. Every decode number above the long-context row was taken at `-c 4096` with <= 1.9k tokens in the KV.

**Hard constraints.**
- 4 GB VRAM: cache slots, MTP head, KV and compute buffers compete. Qwen + head n=2: 36 slots = 3688 MiB and a CUDA OOM during generation (not at load); 30 slots = 3454 MiB fits [M]. The speculation side (head + draft context + recurrent rollback snapshots) costs ~190 MiB [M, notes.md]; every feature that adds VRAM (larger `n_rs_seq`, longer verify batches) must report its own cost.
- 16 GB RAM: CPU-side experts are ~10 GB of shmem; systemd-oomd kills the whole tmux scope on pressure; no builds or downloads next to a resident server; `--cache-ram 0`; preflight refuses to START a job below 12.5 GB available on the idle box (while a server runs, 2-4 GB available is normal) [M].
- The cache graph handles 1-4 token batches; larger verify batches leave the cache path [M].
- On GDN hybrids (Qwen3.6) a draft longer than `n_rs_seq` takes the checkpoint + replay path; `n_rs_seq = --spec-draft-n-max` only when a model drafter is listed, else 0 [V: `common/common.h` `need_n_rs_seq()`, `server-context.cpp`].
- Quality may not move: every change is either provably lossless (speculation, caching, kernels, slot restore) or PROVES non-inferiority (section 8.x). "Within 1 sigma at n=50" is retired.
- The GPU is a cc 7.5 card WITHOUT tensor cores; mainline's dispatch cannot tell (NTC1 / ARCH1). Attention at depth is kernel-bound, not bandwidth-bound (FA1 / FA2).
- GDN hybrid: recurrent state cannot roll back. A restored slot or a cached prefix is re-used only when the new request's token ids EXTEND it; the server's context checkpoints (~1 ubatch before the prompt end, in RAM only, not in the slot file) are what let several questions share one long document.
- Prefill and decode want different servers: prefill = expert cache 0 + big ubatch (the cache serves 1-4 token batches only), decode = cache + ub 128 + MTP. 262k only fits that way.
- The box is exclusive per job: nothing runs beside a benchmark (rule 9).

**Metrics, all per run, all in the ledger:** decode tok/s and wall tok/s per workload, acceptance per drafter, accepted tokens per step, cache hit rate / uploads / evictions, VRAM, GPU util and W, PCIe rx/tx, DRAM GB/s, CPU %, MemAvailable min, swap max, git commit + dirty flag + full args, quality row when applicable.

## 3. Evidence ledger: alive, lead, dead

### 3.1 Alive and measured [M]
- GPU-resident expert cache (P1-P4): numerically sound (cache on == cache off on all three benchmarks), 57-65% hit rate, simulator predicted 59-64%. P4 overlap of CPU misses and GPU hits = +17-18%.
- MTP speculation: standalone Qwen head, acceptance 0.83-0.94, n=2 is the sweet spot; Gemma drafter n=2.
- K-quant experts are the largest single Gemma lever: CPU expert matvecs are compute-bound (`ggml_vec_dot_iq*_q8_K` ~49% of the profile); Q2_K_P holds quality within noise at n=50/41/70 (GSM8K 96 vs 100 is ~1.4 sigma).
- `-t 6` beats `-t 5`; `-lv 4` logging is free.
- **Two-phase long context with slot files** (C1): token-exact restore + extension, 320x at 131k, cross-config.
- `-ub` as the prefill lever (L1); K2 experts (KQ1); in-model MTP head (V1c); mainline + our patches as the tree (U2).
- Mainline `attn-rot` (Hadamard-rotated K/V before caching = the part of TurboQuant that carries the value) is ACTIVE in our build; every q4_0 / q8_0 KV number includes it. Digest: `research/scout-2026-09-20-kvquant-lowbit.md`.

### 3.2 Dead [D]
- **T4, MoE-ifying Qwen3.8-27B to 3-4B active BY FFN-SPLITTING ITS OWN WEIGHTS.** (The goal itself is alive by a different route, see 3.3.) Always-active floor (attention + GDN + lm_head) is ~8.7B [I from V config]; FFN-splitting bottoms out at 9-10B active; the one real derivative (Whittle-MoE-27B-A17.8B, card verified) is 17.8B active and "is not the parent"; quality-preserving upcycling costs 200B-1T tokens [L]. **Reopen only if** someone publishes a <= 6B-active derivative with benchmark parity, or a depth/width-pruning + upcycle recipe under 5B tokens.
- **Retrofitting n-gram embedding modules (Engram 2601.07372, SCONE 2502.01637, PLE, Over-Tokenized 2501.16975) into an existing MoE.** All are pretraining-time architectures [L, IDs verified]. **Reopen only if** a retrofit-by-distillation result appears.
- **Uno (2609.04010) as a drafter on this box.** Draft pass = full-model forward over L tokens; on CPU experts that fans out to up to 8L experts per layer [I]. Its TV loss survives as T1.
- **`ngram-cache` (corpus lookup files) as shipped.** Hardcoded 8-token drafts, never saves the dynamic cache under the server, cross-request staleness bug (#27852) [L from the Qwen scout, file:line cited, spot-check before relying on it].

- **R1 routing bias as a default** [M]: fails at the shipped footing (see board). **`-nkvo`** [M]: decode -68%. **P2 token-table prefetch** [M-sim]: negative. **Sub-4-bit KV / TurboQuant cache types** [L, several sources]: 20-30 pt losses on reasoning / 256k retrieval at 3 bits, mainline rejected the types (#21089).
- **Same-prompt slot restore on the hybrid** [M]: forces a full re-prefill (rollback). Only extension works. **Standalone `-md` MTP head after a restore** [M]: blind, net loss.
- **"Dense is dead here" as a verdict** is WITHDRAWN for Bonsai: it was decided on speed alone (BON1).

### 3.3 The vision already exists as a research preview [V, model cards read 2026-09-19]
`logic65/Whittle-Qwen-3.8-35B-A3B` (+ `-GGUF`, Apache-2.0, released 2026-09-19): 35.1B = 25.1B body + **10.0B hashed n-gram memory** (8 hash heads x 4.88M rows x 256, bigram + trigram, injected before layer 2), ~3B active (8 of 180 routed experts + shared), 40 layers, GDN + full attention, `qwen4_exp` format, "runs on stock llama.cpp". Distilled from Qwen3.8-27B (forward-KL on top-128 teacher distributions over 1,840 complete thinking traces); memory rows transferred from Qwen3.8-Flash-Next's own 320M-row table; a **dependence loss** (`relu(0.3 - (CE_memory_off - CE_memory_on))`) forces the body to route knowledge through the table. Joint training: **3.3 h on one 96 GB Blackwell** after a 30 min table-first warm-up. Self-reported: GSM8K-50 41-44/50, math probe 44-46/60, memory gain +2.84 nats on unseen code; "research preview, not a finished distillation"; recommended sampler is NOT greedy (temp 0.7) and thinking on. GGUFs: Q3_K_M 16.73 GB ... Q8_0 37.83 GB; the memory is served with `-ot per_layer_token_embd=CPU` (one row per token per head, so it is a pageable lookup, not compute) and "must be served whole".
Why it matters: it is dense-27B -> A3B + n-gram table + cheap distillation, i.e. the model half of this program, already built by one person on rented GPU hours. Consequences: (1) **S4** = bench it here. Fit [I]: experts ~10-11 GB to RAM, the ~4.5 GB memory table mmapped from NVMe (random 256-wide row reads, page cache takes what is free), non-expert weights on the GPU; needs mainline at or after PR #27742 (2026-08-27; our mainline tree `e613ef2` has `qwen4exp`), NOT the Prism fork, so it sits behind U1/U2. **Budget [I], to be replaced by measurements:** RAM: routed experts ~10-11 GB resident + table working set (random row reads; the hot rows are the frequent bigrams, so the working set should be far below 4.5 GB, unmeasured) + OS ~1 GB on a 16 GB box where a Qwen3.6 server already leaves only 2-4 GB available; the table's page-cache churn competes with nothing else because experts are pinned shmem, but memory PRESSURE is what triggers oomd. VRAM: non-expert weights of a 40-layer hidden-2048 body at Q3_K_M ~1.5-2 GB + KV + compute buffers + cache slots, against 4 GB; slots will be fewer than Qwen3.6's 30. **Kill S4 if:** the cache-off server cannot hold MemAvailable above ~1.5 GB with swap quiet, or cache-off decode is below 20 tok/s, or GSM8K-50 (run the card's way: temp 0.7, thinking on) is below 40/50. **Cache check:** `build_moe_ffn` has to see the same tensor naming and a top-8-of-180 router; hyper-connections change the residual stream, not the expert FFN, so the cache should apply unchanged, verify with the identity check; our expert cache has to be checked against hyper-connections + shared expert + 180 experts; our quality harness runs temp 0 / thinking off, so run it both our way and the card's way. (2) **T5** = the author states the next steps are compute-limited; Dave's box could continue the recipe. (3) It makes the n-gram table a THIRD table type in this system (model-side memory), next to the draft store (N3) and the expert-prediction table (P2).

### 3.4 Parked, with reopen criteria
- **F1 Flash-Next.** It IS the architecture this program is circling: 512 experts, ~6B active, a 20M-row n-gram embedding table that mainline serves from RAM or disk via mmap (PR #27742), and an MTP layer [V]. Blockers: smallest GGUF 72.5 GB; a RAM-resident build needs 75-81% expert pruning, outside REAP's validated ~50% [L]; NVMe-streamed it lands at 3-8 tok/s [I]; ~6B active is about half the A3B speed even if it fit. **Reopen if** (a) the runtime stack here (cache + prefetch + n-gram drafting) reaches hit rates above 85%, which is exactly what makes NVMe-tier experts viable, or (b) a <= 16 GB pruned build with a quality card appears, or (c) Dave's box is free for a REAP calibration run (inference-only, R9700s, hours).
- **T3 full logit KD.** See 7.4.
- **Bonsai on MLX (Mac).** Different machine, different program.

## 4. The performance model (what every workstream is attacking)

Per decode step [I, fitted to M]:

    T_step(k) = T_fixed(k) + c_miss * m(k) + T_draft
    tok/s     = tau(k) / T_step(k)

- `k` = tokens in the verify batch (1 + draft length). `tau(k)` = expected tokens committed per step = 1 + sum of per-position acceptance.
- `T_fixed` = GPU-side attention/dense/router/cached-expert work: Qwen ~18.5 ms, Gemma ~21.7 ms at k=1 [M fit].
- **Validity: k <= 4 only.** The cache graph serves 1-4 token batches; at k >= 5 the batch leaves the cache path, every routed expert becomes a CPU miss, and on GDN models a draft longer than `n_rs_seq` additionally pays checkpoint + replay. The model is piecewise, with a cliff at k = 5, until P3c and N2 land. N1's 16-token rows measure the far side of that cliff; nothing in the target ladder assumes drafts beyond 3 tokens.
- `c_miss * m(k)` = CPU expert matvecs for the `m` distinct uncached (layer, expert) pairs the batch touches: at k=1 and no cache, Qwen ~20 ms, Gemma ~38 ms [M fit]. `m(k)` grows with k unless consecutive tokens route to the same experts.
- Sanity check: Qwen best config = 46.3 tok/s; at tau ~2.5 that is T_step ~54 ms, i.e. a 3-token verify costs about 2x a single step. **Speculation on this box is throttled by expert fan-out, not by the GPU.** That is the core coupling this program exploits.

### 4.1 Fit to the ledger, 2026-09-20 [M]
Per-round time = 1000 * tau / decode_tps, draft time from the server's `statistics draft-mtp ... dur(g)` line.

| Qwen3.6 IQ2_M | slots | hit % | tau | round ms | of which draft ms |
|---|---|---|---|---|---|
| k=1 (no spec) | 48 / 30 | 62 / 55 | 1.00 | 25.3 / 27.0 | 0 |
| k=2 (MTP n=1) | 36 | 58 | 1.9 | 50 | ~2.7 |
| k=3 (MTP n=2) | 30 | 53 | 2.70 | 58.4 | 5.4 |
| k=4 (MTP n=3) | 28 | 51 | 3.2-3.4 | 75-77 | 7.9 |

Gemma Q2_K_P: k=1 24.9 ms (c19), k=2 37.4, k=3 48.8 (draft 5.1), so ~+12 ms per extra verify token.
- **The draft is cheap (9% of a Qwen round). The verify is what costs:** a 3-token verify is ~53 ms vs 27 ms for one token. With T_fixed ~21 ms at k=3, **~31 ms of a 58 ms round is CPU expert matvecs**, 3.3x the k=1 miss cost for 3x the tokens.
- **Correction to the model above:** the miss term scales with uncached (token, expert) PAIRS, not with distinct experts. The i-quant vec_dot is compute-bound (`ggml_vec_dot_iq2_s_q8_K` = 49% of the profile, DRAM at 8-10 of 38 GB/s), and two tokens that share an uncached expert still pay two matvecs. So "sub-linear m(k)" (T2's second benefit, P3c's premise) buys nothing while misses are compute-bound; only hit rate (P2, T2, slots) and the per-pair cost (the expert quant) move the term.
- **Consequences.** (1) Biggest lever left on Qwen is the per-pair cost: Gemma's IQ3_S -> Q2_K/Q4_0 experts cut the miss cost ~2.7x (60 -> 36 ms per token with no cache). HauhauCS ships no Qwen K-quant that fits 16 GB (Q2_K_P = 13.67 GB of experts), so **KQ1** builds one: gate/up Q2_K + down Q3_K = ~11.7 GB of experts (IQ2_M: 10.41). If the pair cost drops 1.7-2.5x, the round goes 58 -> 45-40 ms = **60-68 tok/s**, and n=3 may start to pay. (2) Hit rate: every 10 points of hit rate is ~6 ms of a Qwen round (~+11%); that is P2/T2/P4c. (3) N3 is capped at the 9% draft share and only on copy-heavy rounds: demoted. (4) T1 (better head so n=3 pays) only pays after the verify tokens get cheaper, i.e. after KQ1.

Four levers, and which work attacks them:

| Lever | Term | Work |
|---|---|---|
| More tokens per step | tau up | N1-N3 (free n-gram drafts), T1 (higher MTP acceptance so n=3 pays) |
| Fewer misses | m down via hit rate h | P2 (prefetch with draft lookahead), T2 (locality-trained routers), P4c (warm-up), admission tuning |
| Fewer distinct experts per batch | m(k) sub-linear in k | T2 (consecutive tokens reuse experts), P3c (cache path for k > 4) |
| Cheaper miss | c_miss down | K-quant expert files (done), kernel work (parking lot) |

**Measured 2026-09-20 (N1):** on Qwen (GDN) an n-gram draft longer than `n_rs_seq` (=MTP n_max=2) forces the recurrent checkpoint+replay path and costs -24% on the edit workload; the identical 16-token drafts are free on Gemma (no recurrent state). So raising tau via longer drafts on GDN models is blocked on N2, and raising it via MTP n=3 is blocked on T1 (n=3 acceptance 0.77-0.80 today, a net loss). tau is cheap to raise only on Gemma until those land.

N and T1 raise tau but also raise k, which raises m(k). P2 and T2 are what keep that affordable. That interaction is the reason to build them as one system rather than four tricks, and it is the claim a write-up would stand on.

Target ladder (targets, not predictions) for Qwen3.6 / Gemma Q2_K_P: 46 / 48 now -> 55 / 56 with the runtime work (N, P) -> 65+ with the two small fine-tunes (T1, T2). Each rung has to be earned in the ledger. Honest dependency: the 55 / 56 rung leans mostly on P (hit rate). If G4 fails, the runtime-only gain on chat/code workloads is whatever P4c warm-up gives plus ~nothing from n-gram drafting (published data says neutral there), and the ladder's second rung moves to Phase 3.
Calibration note: the P4 overlap was projected at +10 / +15% and measured +17 / +18% [M]; our projections have run conservative once, which is one data point, not a license.

## 5. Workstream U: get off the 431-commit backlog

**Why.** The Prism fork is needed only for Bonsai's PQ2_0/PTQ1_0 types, and it lacks `qwen4exp`, so the most interesting model candidate (S4) and Flash-Next (F1) are mainline-only. Everything we serve daily (Qwen3.6, Gemma4, the distill) is mainline-supported. Mainline has 431 commits of fixes we are not getting, and our patches are only upstreamable from a mainline base.

- **U1 build.** `moe-cache` on mainline `e613ef2` (commit `0af8ea3`, compiles on the Mac) built on the box as a third tree. Queued.
- **U2 A/B.** Same model files, same flags, fork `build75` vs mainline build: Qwen cache 30 + MTP n=2, Gemma Q2_K_P cache 15 + drafter n=2, plus cache-off baselines. Queued.
  - **Gate G2:** mainline within -2% tok/s of the fork on all four rows, identical temp-0 text on the fixed prompts, cache stats within noise => mainline becomes the default tree; the fork stays only as the Bonsai tree. If mainline is slower, bisect the delta (it is information about a regression or a missed flag) before deciding.
- **U3 series.** Re-cut the cumulative diff into a reviewable series on mainline: (1) #27861-style cache core, (2) fused gate_up + scales + GELU experts, (3) gated admission, (4) multi-token cache graph, (5) scheduler barrier + overlap, (6) CLI. Each commit builds and passes the identity check.
- **U4 upstream candidates** (each needs Andrei's go before anything public): scheduler barrier/overlap; `n_rs_seq` for n-gram drafters (N2); `ngram-mod` persistence (N3); AVX2 PQ2_0 kernel to the Prism fork + a +1 on its PR #205.
- **U5 rebase cadence.** Once on mainline: rebase weekly, re-run the A/B pair as the regression test. **Rebase hazard [L, Qwen scout]: open PR #28391 makes `ngram-mod` a DEFAULT drafter on server/CLI — a rebase that pulls it in would silently activate two drafters; explicitly set `--spec-type` on every launch and check for it after each rebase.**
- **N2 implementation lead [L, Qwen scout, verify]:** instead of editing `need_n_rs_seq()`, open PR **#26499** adds a `LLAMA_N_RS_SEQ` override; or simply `--spec-draft-n-max 3` raises `n_rs_seq` to 3 (but that also verifies n=3 MTP, which N1 showed is a net loss today — so the override is the cleaner path). Read #26499 before building N2.

## 6. Workstreams N and P: the runtime co-design

### 6.1 N1, hybrid n-gram + MTP drafting (zero code, queued)
[V] `--spec-type` is a comma list; implementations run in a fixed order with n-gram types first; the first non-empty draft wins the round; MTP does not run that round. `ngram-mod` = one 16 MB lossy direct-index hash table (4M x int32; `add()` overwrites unconditionally, no tags, no probing, so collisions silently replace entries), key = hash of the last `n_match` (24) tokens, value = ONE next token; a draft is produced by walking the table iteratively. **`n_min` is a cliff, not a floor** [V `speculative.cpp` draft loop]: if the walk dies before `n_min` tokens the whole draft is discarded, so `n_min = n_max = 2` throws away valid 1-token hits. Defaults 48/64 are for long verbatim copies.
- Bench `ngram1.sh`: MTP n=2 baseline; hybrid capped at 2 with `n_min` 2 and with `n_min` 1 (inside the cache window and inside `n_rs_seq`); cache-28 pair at n=3; hybrid at 16 (outside both, measures the replay path); n-gram alone; Gemma pair (no recurrent state). Workloads: code, reason, edit (copy-heavy).
- **Gate G3:** hybrid >= +3% on the edit workload and no worse than -1% on code/reason => keep it on by default and proceed to N2/N3. A published data point says neutral on chat, ~4x on repetitive recall (#27210 [L]); expect the gain to be workload-shaped.

### 6.2 N2, make long n-gram drafts legal on GDN models
Problem [V]: `need_n_rs_seq()` returns `draft.n_max` only when a model drafter is listed, so n-gram drafts longer than the MTP cap, and every draft in an n-gram-only chain, take checkpoint + replay on Qwen3.6.
Change: `n_rs_seq = max over enabled implementations of their n_max` (bounded; each extra snapshot costs recurrent-state memory, to be measured), plus per-implementation draft caps so MTP stays at 2 while n-gram goes longer.
- **Kill:** if N1's 16-token rows show the replay path already pays, or never pays even on the edit workload, N2 is unnecessary.
- **Verdict 2026-09-20 [M + V, Qwen scout hunt 2 checked against source]:** the change itself is ~6 lines (`need_n_rs_seq()` takes the max with the enabled n-gram types' ceilings, mirroring `common_speculative_n_max()`, `common/speculative.cpp:2335`); nobody upstream owns it (PR #26499 is only a `LLAMA_N_RS_SEQ` getenv, open, and it also reroutes non-speculative runs onto the RS path, #28425). But `n_rs_seq` is not free: `llama-memory-recurrent.cpp:101` allocates `mem_size * (1 + n_rs_seq)` rows per GDN layer ON THE GPU, measured **62.81 MiB per unit** for Qwen3.6 (F32 state, 40 layers). `n_rs_seq` 16 = 1068 MiB vs 188 MiB today = +880 MiB, i.e. ~22 of our 30 cache slots (39 MiB per slot). **Long n-gram drafts on Qwen are dead on a 4 GB GPU.** Do not shrink the state to pay for it (DAMP 2608.27513: do not quantize GDN state). What survives is **N2-lite**: n-gram cap 3 with MTP at 2, `n_rs_seq` 3, +63 MiB (~1.6 slots), inside the 4-token cache graph; N1's closest row (c28 + ngmod3 + MTP3) was 49.5 vs 49.0 on edit, so expect ~+1% on copy-heavy work only. Bundle the 6-liner into the next box build and A/B it (edit/code/reason, c29 vs c30); never a build of its own. **Reopen the long form if** a bigger GPU appears, or upstream makes rollback planes lazy/host-resident (#29121 direction). Post-rebase invariant: if #28391 lands, its `ngram-mod` default `n_max` 64 would inflate `n_rs_seq` to 64 through this patch (~4 GB of planes) => the launcher ALWAYS pins `--spec-ngram-mod-n-max`. Gemma has no recurrent state: long n-gram drafts stay free there.
- Depends on P3c when drafts exceed 3 tokens (the verify batch leaves the cache graph).

### 6.3 N3, the "rainbow table": a persistent, offline-seeded continuation store
The honest mapping: a rainbow table trades space for recomputation by storing little and re-deriving the chain. `ngram-mod` already has that shape: it stores one next-token per hashed context and re-derives a multi-token draft by walking. What it lacks is memory across requests and a precomputed seed.
Design:
0. **Work items, in order:** N3a two-region table (seeded region is read-only to `reset()` and to the >25% occupancy wipe in `begin()`; live region keeps today's behavior; lookup checks live first, then seeded) -> N3b load/save + header (tokenizer hash, `n_match`, entry count) -> N3c offline seeding tool + corpus per workload -> N3d backoff tiers. N3a comes first because persistence without it is useless: the low-acceptance reset (5 bad rounds) wipes everything, and low live acceptance is exactly when the seeded tier is supposed to help.
1. **Persistence:** `--spec-ngram-mod-load FILE` / `--spec-ngram-mod-save FILE`, plus a size flag (today fixed at 4M entries = 16 MB; 64M entries = 256 MB is affordable, mmapped with `MADV_RANDOM` so it never evicts expert pages). File header carries the tokenizer hash and `n_match`.
2. **Offline seeding (self-distillation):** run the DEPLOYED model (same quant, same abliteration) at temp 0 over a representative prompt corpus per workload; feed its outputs through the same hash to build the table. A store built from the model's own outputs maximizes acceptance because it encodes this model's continuations, not a corpus author's [I; no paper compares the two directly].
3. **Backoff tiers:** try `n_match` 24, then 12, then 6; shorter context = higher coverage, lower precision; cap the draft length by tier so low-confidence tiers only propose 1-2 tokens. **Each tier needs its own table** and, below 12, a tag check (store a few key bits next to the value): the structure overwrites on collision, and short keys over a large corpus collide constantly, so tiers sharing one untagged table would trash each other. Collisions never hurt correctness (verification is lossless), only acceptance. Mainline warns below `n_match` 16 for this reason.
4. **Pruning (CREST-style [L]):** keep entries whose continuation was observed >= 2 times or came from a high-margin greedy step.
5. **Policy:** the low-acceptance reset must not wipe the seeded tier (today `mod.reset()` clears everything after 5 bad rounds [L]).
- **Measure:** acceptance and accepted-tokens-per-step per workload, cold store vs seeded store, table size sweep.
- **Kill:** seeded store adds < 5% accepted tokens per step over the live-history table on W2-W4.
- Later option, only if W3/W4 justify it: a suffix-tree drafter (SuffixDecoding 2411.04975 [L]: ~20 us per draft token, 1.8-5.3x on agentic workloads). 1-2 weeks of work; not before N3 data says copy-heavy traffic is where we live.

### 6.4 P1/P2, draft-driven expert prefetch
Correction to the scout: our drafts are TOKENS (MTP head, n-gram table), not main-model router outputs. The main model's routing for a draft token is only known during verification. So prefetch has to predict experts from token identity:
- **P1 (queued):** `predict.py` on the trace2 data (40k tokens with token-id sidecars): recall@B of a token's true experts from unigram and bigram tables per layer band, against the `recent (LRU)` and `global` baselines.
  - **Gate G4:** bigram-table recall@(slots per layer) beats the LRU baseline by >= 10 points in at least the deeper half of the layers => build P2. Otherwise the cache is already capturing what token identity can tell us, and the hit-rate lever moves to T2.
- **P2 design:** an mmapped table `(layer, hash(prev_token, token)) -> top-B expert ids` built offline from traces (the second "rainbow table"). When a draft is produced, look up its tokens and hand the predicted experts to the existing async upload worker as admission candidates that bypass the 3-misses gate but never evict an expert used within the admission window (`--moe-expert-cache-window`, 16 tokens today). **The table belongs to one model FILE**: routing shifts with the quant, the abliteration and the fine-tune, and token ids with the tokenizer, so the header carries the model file hash + tokenizer hash and the runtime refuses a mismatch. Build it only after S1/S2 settle the served files; regenerating costs one trace run (~40k tokens). Uploads stay off the critical path; misses are still computed on the CPU; nothing stalls.
  - Prior art [L, IDs verified]: MoE-SpAc 2603.09983 uses draft tokens as a lookahead sensor for the expert cache (85% prediction accuracy reported); MoE-SpeQ 2511.14102 similar. Caution from Speculating Experts 2603.19289: COMMITTING to predicted experts wrecks quality (GSM8K 0.950 -> 0.576 reported); that does not apply here because our predictions only warm the cache and every expert the router picks is still computed exactly.
  - Risks: PCIe upload saturation (ungated admission churned before [M]); mitigate with a per-step upload budget and a confidence threshold. Literature reports 90%+ next-layer recall from hidden-state predictors [L]; token-only predictors are weaker; G4 tells us by how much.
- **P4c prefill warm-up:** route the prompt, count expert use per layer, fill the slots before the first decode step. Removes the cold-start misses that every short benchmark run pays.
- **P3c cache graph for k > 4:** needed only if N says long drafts pay (skip-id path, mainline #26631).
- Eviction policy: our simulator said policy barely matters at these slot counts [M-sim]; LFRU is a parking-lot item unless P1 shows hub-expert thrashing.

## 7. Workstream T: what Dave's box is actually for

Principle: **only small-trainable-set jobs.** Full-model training of a 35B MoE on 2x MI210 with ZeRO-3 offload is estimated at 60-150 tok/s [L, stated assumptions, no direct measurement exists] = 15-39 days per 200M tokens. Two existence proofs for the cheap direction, both from the Whittle line [V cards]: the 27B-A17.8B model trained routers with every expert frozen on personal hardware, and Whittle-Qwen-3.8-35B-A3B did its joint memory + KD training in 3.3 h on one 96 GB Blackwell (1,840 teacher traces; a research preview, not a full distillation). Neither transfers to 2x MI210 under ROCm without T0.

### 7.1 T0, throughput spike (1 hour, gates everything else)
On the MI210 pair, AMD's curated ROCm image, BF16, DeepSpeed ZeRO-3 + CPU offload: load Qwen3.6-35B-A3B HF weights, freeze everything except (a) the routers, (b) the MTP head; measure forward-only tok/s and trainable-step tok/s at seq 2048. Known risks [L]: no FP8 on gfx90a, grouped-GEMM coverage, gfx90a off the CI hot path, no mixed gfx90a + gfx1201 collectives (R9700s only for separate inference jobs).
- **Output:** a measured tok/s that replaces every ETA in this section. **Re-plan threshold:** a trainable-step rate (forward + backward through the frozen body) below ~100 tok/s puts a 10M-token T2 run past a day; below that, shrink the token budget or the sequence length before giving up on T2.

### 7.2 T1, drafter fine-tune with a total-variation loss
- Trainable: the MTP head only (Qwen) / the drafter only (Gemma). Body frozen.
- Loss: TV distance between drafter and body next-token distributions at each draft position (acceptance probability = 1 - TV, so the loss is the metric); chunked implementation from ifm-ai/uno `training/losses.py` (Apache-2.0) [V]. Self-distillation data: the body's own generations on the workload corpus (pairs with N3's corpus; generate once, use twice). Related recipe: MTP self-distillation 2603.23911, +5-7% acceptance [L].
- Target: per-position acceptance high enough that n=3 beats n=2 on the box (today it does not [M]).
- Deliverable: a replacement `mtp-*.gguf`; the body file is untouched.
- **Kill:** < 2 points of acceptance gain at position 2 after the first 10M tokens.
- The identity acceptance = 1 - TV holds under standard speculative rejection sampling, which is what llama.cpp implements; at temp 0 it degenerates to argmax agreement, so also track top-1 agreement.
- **Gate Q1:** the deployed body is an abliterated IQ2_M quant; the head trains against BF16 weights. Acceptance is measured on the box against the deployed file. If the gain seen on Dave's rig shrinks by more than half on the box, retrain against the DEPLOYED body's distributions (generate targets with llama.cpp from the IQ2_M file; slower, but it is the distribution that matters).

### 7.3 T2, router-only locality fine-tune (ReMoE-style 2605.27081 [L])
- Trainable: `ffn_gate_inp` per MoE layer only. Loss: temporal-locality regularizer (consecutive tokens reuse experts) + KL trust region to the frozen router; standard load-balance loss off.
- Why it fits this box twice: it raises the cache hit rate AND makes m(k) sub-linear in k, which is exactly what throttles speculation here (section 4).
- Deliverable: a patched GGUF where only the router tensors are replaced. **Routers are F32 in both deployed files [V, gguf-py on the box 2026-09-19]:** Qwen3.6 IQ2_M has 40 x `ffn_gate_inp` [2048, 256] (~21M trainable parameters in total) plus 40 x [2048] shared-expert gates; Gemma4 Q2_K_P has 30 x [2816, 128] plus 30 x [2816]. The swap needs no requantization of anything.
- Second recipe to compare: Sticky Routing 2607.08780 [L] (L2 penalty on consecutive gate distributions; cache hit rate 0.54 -> 0.88 with perplexity improving, but only shown at 8.8-22M parameter scale). **Unreconciled [L]:** the same paper reports ReMoE doing almost nothing at that scale, and neither recipe has been shown at 35B-A3B. So the recipe choice is an experiment, not a given: first step of T2 is a short ablation of both losses on real Qwen3.6 routing (a few hundred steps each, hit rate measured with our simulator on held-out traces) before any long run.
- Evaluate BEFORE training with the simulator: replay traces through a synthetic "stickier" router to see how much hit rate a given locality gain buys.
- **Gates:** quality gate (section 8) on the patched file, hit rate +10 points at equal slots, tok/s up on the box. **Kill:** any benchmark drops more than 1 sigma, or hit rate gain < 5 points.
- Open question Q2: train on the same weights we deploy. The box serves HauhauCS abliterated variants; either obtain those HF weights, or train on base weights and have Dave re-abliterate, then re-check.

### 7.4 T3, full logit KD (parked)
Dense Qwen3.8-27B -> 35B-A3B student: 15-193 days of student training for 200M-1B tokens plus 29-58 days of teacher top-k logit generation on 4x R9700 [L, assumptions stated in the scout report]. An SFT-distilled checkpoint already exists (S2). **Reopen if** T0 measures >= 5x the assumed throughput, or S2 shows the distill is clearly better than Qwen3.6 and we want more of the same.

### 7.5 T5, continuing the Whittle recipe (lead)
Only after S4 says the preview is competitive on this box. Inputs the author publishes: weights, eval logs, the recipe on the card (`logic65/whittle-dev` exists; contents not yet read). What Dave's box would add: more teacher traces (broad-domain, from Qwen3.8-27B on the R9700s), longer joint training, on-policy distillation. Differences that T0 has to price: one 96 GB Blackwell with a mature CUDA stack vs 2x 64 GB MI210 under ROCm with weak grouped-GEMM coverage [L]. Courtesy item: talk to the author before building on their work; they ask for compute support on the card.

## 8. Experiment protocol and quality gates

- **Naming:** `<ID>_<model>_<config>`; the box script is mirrored in `bench/box/`, the log marker is `<SCRIPT>_DONE`, waiting is only ever done with `bench/watch.sh` (death-aware) or the row monitor.
- **Workloads (H2):** W1 chat/reasoning, W2 code generation, W3 code edit / copy-heavy, W4 long-context summarization/RAG. Results are reported per workload, never blended. W1-W3 exist in `specclient.py` (W3 behind `EDIT=1`); W4 is to be added.
- **Lossless changes** (cache, speculation, prefetch): temp-0 text identity against the reference config on the fixed prompts, plus one full quality pass per major change as a canary.
- **Weight changes** (quant, distill, router or head fine-tunes): quality harness at temp 0, thinking off, new caps (768/1024/1024): GSM8K-50, HumanEval-41, MMLU-Pro-70. Pass = every benchmark within 1 sigma of the reference row. That gate is loose at n=50 (1 sigma is up to ~6 points, i.e. 3 GSM8K items), so it is only the ITERATION gate; anything that is going to be called done (T1, T2, a new default file) must also pass at the larger n. For final default decisions use the larger set (GSM8K-200, MMLU-Pro-280), because at n=50 sigma is 2.8-5.9 points and cannot separate close candidates. Plus a refusal spot-check on anything that replaces an abliterated file.
- **One variable per run.** A/B pairs run back to back in the same script, same thermal and memory state, preflight enforced.
- **Record negative results.** A killed idea gets a ledger row and a line in section 3.2 with its reopen criterion.

### 8.x Quality gate = paired non-inferiority (Andrei, 2026-09-20: "not faster at getting it wrong")
A quality-affecting lever (requant, routing bias, KV quantization, smaller file) is adopted only if it PROVES it is not worse:
same question ids in both arms, count discordant pairs (base right / test wrong vs the reverse), exact McNemar p + 95% CI on the
paired difference, and the LOWER bound must clear the margin (proposed: -2 pts pooled). "Within 1 sigma at n=50" is retired: it
passes by default. About 9% of answers flip between any two ~2-3 bit quants, so proof needs ~1000 paired questions per decision
(published sets only: GSM8K test 1319, MMLU-Pro, HumanEval 164). Ties go to the higher-precision file. Lossless levers need no gate.
First paired reads (existing rows, 2026-09-20): R1 b1.0 vs b0: loses 0 / wins 3 of 91, diff +3.3 [-0.4, +7.0]; R1 b2.0: loses 7 /
wins 3, dead; K2 vs IQ2_M: 8 / 7 of 161, -0.6 [-5.3, +4.1] = undecided (kq1h adds n=480); Gemma Q2_K_P vs IQ3_M: 29 / 20 of 521,
-1.7 [-4.4, +0.9] = not proven, IQ3_M stays; Gemma Q3_K_M vs IQ3_M: 20 / 16, -0.8 [-3.0, +1.5] = not proven.

### 8.y The instruments behind 8.x (built 2026-09-20/21)
- `bench/qual/paired.py BASE.jsonl TEST.jsonl [--margin 2]`: discordant pairs, exact McNemar p, CI on the paired difference, verdict NON-INFERIOR / WORSE / UNDECIDED + the n needed. ~9% of answers flip between any two ~2-3 bit quants, so a decision needs ~900-1000 paired questions.
- KL divergence vs the Q6_K_P reference (`kld1.sh`, `llama-perplexity --kl-divergence`): every token is a sample, minutes per candidate; SCREENS candidates, the paired task test confirms finalists.
- Hard sets with thinking on (`fetch.py --hard`, `qual.py --think`): AIME 24/25, MATH-500 L5 (numeric golds, exact-fraction scoring), EvalPlus. The standard sets are saturated and cannot see the reasoning collapse reported for ~2-bit PTQ.
- `bench/qual/longctx.py` (`lq1`): retrieval at depth per KV type, with a reuse guard.
- Published items only, verbatim. No hand-written test prompts; no refusal probes unless Andrei asks.

## 9. Sequencing

(Rewritten 2026-09-21; the 2026-09-19 phases 0-1 are done, see the board.)
**A. Prove the new code (front of the queue):** `fa1` (FA1 + FA2 + NTC1: unit-test gates, text identity from the kept 131k / 262k slots, llama-bench matrix) -> `arch1`. Winners become the binary every later job uses; rerun `prof3` for the after picture.
**B. Stand up the instruments:** `kld1` -> `hq1` (IQ2_M, then K2) -> `lq1`. With `ctx1c`, `prof2`, `kq1graft` interleaved (short).
**C. Decide, one sitting, from rows (Andrei):** K2 as the Qwen default (paired OK; needs kld1 + hq1 + his spot-check); in-model head + K2 as the serving file (`kq1graft`, `ctx1c`); KV type of the long-context config (`lq1`); Bonsai (`bonsai1`); RAM1 if `hq1` shows a collapse.
**D. QX accuracy experiment:** `expert_mix` emulation + Unsloth-recipe replay + (Dave's box) Q6-source imatrix, all scored on the kld rig with a code corpus added; runtime C++ for the hot/cold split only on a >= ~25% KLD gain at equal size.
**E. Long-context product layer (C2):** `ctxproxy2` (Qwen) -> two-phase supervisor -> shipped config.
**F. Dave's box (needs Dave's OK):** QX1 imatrix, T0 spike -> T1 -> T2 as before.
**G. Compose and ablate, write-up (section 10).**

Critical path: A -> B -> C. D and E run beside B on the Mac / Qwen side because they need the box only for short serialized jobs.

## 10. The write-up (W1)

Working title: *Speculation is expert-bound: co-designing n-gram drafting, MTP and a GPU-resident expert cache for MoE decode on a 4 GB GPU.*
Claims the data would have to support:
1. With experts on the CPU, the cost of a k-token verify is governed by distinct uncached experts, so speculation and caching are one optimization problem (section 4), shown with measured T_step(k) and m(k).
2. A never-stall cache (misses computed on the CPU, async gated admission, overlap of hit and miss paths) is numerically exact and worth +X%.
3. Free n-gram drafts from a self-distilled persistent table compose with an MTP drafter, workload-dependently.
4. Draft tokens are a prefetch signal: token-n-gram -> expert tables lift hit rate by Y points at zero model cost.
5. Two small fine-tunes (drafter TV loss, router locality) move acceptance and hit rate without touching expert weights, and the router patch drops into a quantized GGUF without requantization.
Ablation matrix (per model, per workload): baseline | +cache | +MTP | +n-gram live | +n-gram seeded | +prefetch | +warm-up | +T1 head | +T2 routers | all. Every cell is a ledger row with the full telemetry and a quality reference.

## 11. Risk register and checkpoint log

| Risk | Mitigation |
|---|---|
| Box dies silently (OOM, oomd, wedge) | preflight, `--cache-ram 0`, swapfile, `watch.sh`, 30-min monitor re-arm, no builds/downloads next to a server |
| VRAM ceiling makes features mutually exclusive | every feature reports its VRAM cost; slots vs head vs batch is tuned per model, never assumed |
| PCIe upload churn from prefetch | upload budget per step, confidence threshold, keep the gated path as fallback |
| GDN replay path hides n-gram gains | N1 measures it explicitly; N2 removes it |
| Scout claims wrong | tags [L] until checked; IDs and code paths verified before work is scheduled |
| ROCm on gfx90a bites | T0 spike before any plan; curated image; small-trainable-set jobs only |
| Train/deploy mismatch (BF16 base vs abliterated IQ2_M) | Q1/Q2; acceptance and quality are measured on the box, on the deployed files |
| Public exposure by accident | no public fork or PR without Andrei's explicit go |
| Tables silently mismatched to the model (N3 store, P2 table are tokenizer- and file-specific) | header with tokenizer hash + model file hash; runtime refuses a mismatch; a mismatch would show up only as low acceptance / low hit rate, never as an error |
| Disk (64 GB free on /ai at 2026-09-19; S4 adds 16.7 GB, traces and tables are < 1 GB) | disk check before every download; delete superseded GGUFs only with Andrei's OK |
| S4 does not fit | see the S4 budget in 3.3; kill criteria there |
| Losing the thread | section 0 rules, the status board, WIP limits, the parking lot |
| A benchmark that shared the box | rule 9: builds / tests / profiles are queue jobs; a contaminated row is marked in `notes.md` and rerun (open: `ctx1b` 262k prefill t/s) |
| Saturated or underpowered quality gate waves a bad lever through | 8.x / 8.y: paired test, KLD, hard sets; ties to the higher-precision file |
| Token-exactness on the hybrid (proxy, templates, tokenizer junctions) | token-id arrays only; id sidecars next to slot files; a miss shows up as `prompt_n` ~ full in the log, never as an error |
| The Qwen coder stalls (multi-hour debug turn, or its own stop token) | fenced small briefs, 3-attempts rule, waiters that fire on delivery / long turn / idle-at-prompt; recover with Escape + a delete-don't-debug directive |
| Disk (40 GB free on / at 2026-09-21; kept slots 2 GiB, kld logits 3-5 GB, QX intermediates 20-36 GB) | intermediates go to `/mnt/md0` (563 GB free), cold models already there behind symlinks; deletions only with Andrei's OK |
| Waiters lie (stale log markers, count-based exits) | `watch.sh` ignores a log not written since it started; queue waiter keys on `queue.tsv` state, not on a job list |

Checkpoint log (append: date, phase, what was verified, review findings):
- 2026-09-21: **forensic re-read (rule 5), intent vs implementation drift found:** (1) the quality gate was underpowered AND saturated: "within 1 sigma at n=50" passed by default, and the sets cannot see hard-reasoning damage => 8.x / 8.y; (2) `ctx1`'s restore rows tested a rollback, not a restore, and produced a false "dead" verdict for a day => `ctx1b`; (3) `r1b`'s speed rows were designed to OOM (cache 48 + MTP head); (4) `watch.sh` was cited in memory for a day while absent on the box, then fired on a stale log => fixed twice; (5) I ran a build and a GPU test beside the 262k prefill => rule 9, row marked contaminated; (6) repo `bench/box/` vs box flat layout drift => MIG1; (7) the "dense is dead" Bonsai verdict was speed-only => BON1; (8) board rows KQ1 / R1 / C1 / V1c / L1 / S1 were stale by a day => this rewrite. Verified this checkpoint: ledger + results synced, 106+ tests green, FA patch built and unit-tested, scouts' key claims checked against our own tree (`attn-rot`, `FA_ALL_QUANTS`, the tensor-core banner).
- 2026-09-19: adversarial review of v1 (Sonnet reviewer, 25 findings: 2 critical, 8 high, 10 medium, 5 low). Fixed: performance model now piecewise with the k=5 cliff; `ngram-mod` described as a lossy overwrite hash, tiers get their own tagged tables; N3 decomposed with reset-protection first; `n_min` cliff documented and an `n_min 1` row added to `ngram1.sh`; S4 got RAM/VRAM budgets and kill criteria; router tensors verified F32 on the box ([I] -> [V]); ReMoE vs Sticky Routing conflict made explicit, recipe ablation added to T2; Q1 promoted to a gate; 46.3 tagged as a 200-token number and `steady1` queued; P2/N3 tables made file- and tokenizer-specific; W defined; quality gate split into iteration vs final; Mac-side Phase 0 sequenced; T0 threshold restated. Not changed: the 8.7B vs ~9.7B active-floor discrepancy (finding 16) does not move any decision, the scout's "9-10B" is what 3.2 already says; PLE has no arXiv ID (Google Gemma docs).
- 2026-09-19: v1 amended the same hour: the n-gram scout's second pass surfaced Whittle-Qwen-3.8-35B-A3B; both model cards read first-hand, GGUF sizes from the HF API; added S4, T5, section 3.4; corrected the earlier "3.3 h claim cannot be sourced" statement (it is on the card).
- 2026-09-19: spec v1 written. Inputs: `notes.md`, the five scout reports, first-hand reads of ifm-ai/uno and of mainline `common/speculative.cpp`, `common/common.h`, `common/arg.cpp`.

## 12. Parking lot (ideas that are NOT in the queue)
- LFRU / learned eviction; hidden-state expert predictors (SpecPrefetch 2607.24787, ProMoE 2410.22134) if token-only prediction (P1) falls short.
- `GGML_CUDA_NO_PINNED=1` headroom test; moving the cache singleton into `llama_model`.
- Faster CPU expert kernels (AVX2 IQ2_S/IQ3_S), only if we are forced back onto IQ-quant experts.
- `draft-dflash` / `draft-dspark` spec types exist in both trees [V]; no drafter weights for our models; revisit if any appear.
- Suffix-tree drafter; NVMe-tier experts for Flash-Next (F1); REAP calibration run on Dave's R9700s.
- Scout davetha's current repos for RDNA4 tricks (offered, not confirmed).
- Leads, unverified: `TheTom/llama-cpp-turboquant`, the AtomicBot fork (Qwen3.6-35B-A3B + MTP + sub-4-bit KV, claims 1M context), a PR-thread claim of "Qwen at full context on a 4 GB GTX 1650 at 50 t/s". All sub-4-bit KV => leads only under rule 8.
- A natively-trained ternary MoE (none exists as of 2026-09); QAT methods (ParetoQ, LC-QAT, CAT-Q, ScaleQ-1.58) and MoE 2-bit papers (BitsMoE, TileQ, GEMQ) have no usable checkpoints.
- Upstreaming FA1 / FA2 / NTC1 (public => Andrei's explicit go). Slot files that carry the server's context checkpoints (same-prompt restore). Single-purpose specialists (Postgres DBA: harness + execution-verified traces + size sweep 0.27B-4B) — discussed 2026-09-20, not started.

## 13. References
Reports: `research/ngram-codesign-scout.md`, `research/ngram-llamacpp-scout.md`, `research/upcycle-scout.md`, `research/lit-scout-arxiv-survey.md`, `research/pr-scout-llamacpp-prs.md`, `research/moe-scout-abliterated-moe.md`, `research/qwen-mtp-scout.md`. Patches: `research/patches/`. Simulator and predictor: `research/scripts/moe-cache-sim/`.
Papers (arXiv IDs resolved 2026-09-19): SuffixDecoding 2411.04975, REST 2311.08252, CREST 2408.04678, Engram 2601.07372, SCONE 2502.01637, ReMoE 2605.27081, SpecPrefetch 2607.24787, MoE-SpeQ 2511.14102, SeqMoE 2609.12978, MoE cache evaluation 2608.07911, MTP self-distillation 2603.23911, Uno 2609.04010, MoE-SpAc 2603.09983, Speculating Experts 2603.19289, Sticky Routing 2607.08780, local routing consistency 2505.16056.
