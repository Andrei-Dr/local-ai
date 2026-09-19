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

## 1. Status board

| ID | Item | State | Gate / next action |
|---|---|---|---|
| H1 | Harness: preflight, mon, ledger, quality, death-aware waiters | done [M] | keep |
| H2 | Workload set W1-W4 + larger-n quality set | todo | before any model-default decision is called final |
| Q-box | Box queue. Done: q38, distill, build_mainline, mainline_ab, trace2, ngram1, steady1 (qual3/Bonsai skipped). Remaining: `mmlu2k` (clean MMLU-Pro at cap 2048) -> `mainqual` (G2 quality confirm) -> `trace2b` (`.tok` sidecars for P1) | running, unattended; **durable: `/ai/bench/queue.tsv` + `ai-queue.service`** | Stop a job ONLY with `systemctl stop ai-queue` (never pkill). Enabled at boot (resumes by itself after a poweroff; `Restart=no`, so never after a crash); `systemctl start|stop|disable ai-queue`; status with `/ai/bench/queue.sh list` |
| S1 | Gemma default file (Q3_K_M vs Q2_K_P) | data in [M], provisional: Q2_K_P | H2 larger-n pass to confirm |
| S2 | Qwen3.8-35B-A3B-Distill vs Qwen3.6-35B-A3B | **done [M]: KEEP Qwen3.6** | distill regresses HumanEval -12 / GSM8K -8 (> 1 sigma), has no standalone MTP head, 39.1 vs 46.3 tok/s. Reopen only if the clean Qwen3.6 MMLU-Pro (`mmlu2k`) lands far below the distill's 72.9 |
| S3 | Bonsai PQ2_0 vs stock Qwen3.8-27B IQ3_M (dense reference) | **done [M]: dense is dead here** | stock 27B = 1.38 tok/s (`-ngl 16`); Bonsai 5.15 tok/s, quality run dropped by Andrei. No quality rows; fix the `-ot` FFN regex before any future dense bench |
| S4 | **Whittle-Qwen-3.8-35B-A3B** (A3B body + 10B hashed n-gram memory, distilled from Qwen3.8-27B; `qwen4exp`, mainline only) | todo, unblocked once `mainqual` confirms G2 | see 3.3; download Q3_K_M 16.7 GB after the queue drains, disk check first |
| U1 | Mainline build of `moe-cache` on the box | **done [M]** | linked clean at CUDA arch 75 (`/ai/src/llama.cpp-mainline`) |
| U2 | Fork vs mainline A/B | **G2 PASS on speed [M]** (+5-7%, VRAM flat); Qwen text bit-identical, Gemma code prompt flips a greedy tie after ~10 tokens | `mainqual` quality rows within 1 sigma of the fork rows => mainline is the default tree |
| U3 | Split the cumulative mainline diff into a reviewable series | todo | after U2 |
| U4 | Upstream candidates | todo | needs Andrei's explicit go (a fork of a public repo is public) |
| N1 | Hybrid `ngram-mod,draft-mtp` bench | **done [M]** | G3 PASS (workload-shaped): keep MTP n=2, n-gram ON, cap Qwen<=2 / Gemma long; +3.8% Qwen edit, ~free Gemma, neutral code/reason |
| N2 | Decouple `n_rs_seq` from the model drafter for n-gram drafts | **long form dead on 4 GB [M]; N2-lite (cap 3) optional** | each `n_rs_seq` unit = **62.81 MiB VRAM** on Qwen3.6 [M, box logs: 62.81 / 188.44 / 251.25 MiB at 0 / 2 / 3]; 16 units = +880 MiB = ~22 cache slots. Only `n_rs_seq` 3 (+63 MiB, ~1.6 slots) is affordable; expected ~+1% on edit. See 6.2 |
| N3 | Persistent, offline-seeded n-gram continuation store ("rainbow table") | **demoted [M]**, design in section 6 | ceiling is the draft time it replaces: 5.4 ms of a 58 ms Qwen round (9%), and only in rounds where the table drafts (70 of 321 on the edit workload). Build only after KQ1/P2, and only if copy-heavy traffic is where we live |
| KQ1 | **K-quant experts for Qwen3.6** (requantize HauhauCS Q6_K_P: gate/up Q2_K, down Q3_K, own imatrix) | queued (`kq1.sh`) | section 4.1. Gate: quality within 1 sigma of IQ2_M, MemAvailable min >= 1.5 GB, swap quiet. Kill: miss cost improves < 1.3x |
| P1 | Token-n-gram -> expert predictability (`predict.py` on trace2) | blocked on `trace2b` (first trace run wrote no `.tok` sidecars: `trace_tok.py` guard bug, fixed `1509285`) | G4 |
| P2 | Draft-driven expert prefetch | blocked on P1 | G4 |
| P3c | Cache graph beyond 4-token batches | todo | needed if N1 shows long drafts pay |
| P4c | Prefill warm-up of the cache | todo | cheap, independent |
| T0 | ROCm throughput spike on Dave's MI210 pair (1 h) | todo | gate for every T item |
| T1 | MTP head / drafter fine-tune with TV loss (self-distillation) | design in section 7 | after T0 |
| T2 | Router-only locality fine-tune (ReMoE-style) | design in section 7 | after T0 and P1 |
| T3 | Full logit KD dense 27B -> 35B-A3B | parked [L] | 15-193 days student + teacher-logit generation; see 7.4 |
| T5 | Continue the Whittle recipe (memory transfer + dependence loss + forward-KL) on Dave's box | lead [L] | after S4 shows the preview is worth continuing, and after T0; see 7.5 |
| T4 | Dense -> A3B/A4B conversion of Qwen3.8-27B | dead [D] | see 3.2 |
| F1 | Qwen3.8-Flash-Next (native n-gram table + MoE + MTP) | parked by Andrei | see 3.4 |
| W1 | Write-up / paper | outline in section 10 | after the ablation matrix is filled |

## 2. North star, constraints, metrics

**Goal.** Maximum decode tok/s at unchanged quality on: GTX 1650 SUPER 4 GB (Turing, PCIe 3.0 x16), i5-10400F (6C/12T, AVX2), 16 GB DDR4, NVMe. "Every 5 tok/s counts."

**Where we are [M].**

| Model | Morning of 2026-09-19 | Now | Config |
|---|---|---|---|
| Qwen3.6-35B-A3B IQ2_M | 28.0 | 46.3 (200-tok); **47.4/45.5 steady-state 600+ tok [M]** | cache 30 + MTP head n=2 |
| Gemma4-26B-A4B IQ3_M | 16.2 | 28.7 (29.7 MTP) | cache 16 |
| Gemma4-26B-A4B Q3_K_M | - | 33.7 (37.2 drafter n=2) | cache 15 / 11 |
| Gemma4-26B-A4B Q2_K_P | - | 40.2 (48.2 drafter n=2); **50.5/48.3 steady-state 800/587 tok [M]** | cache 19 / 15 |

Steady-state (`steady1`, 2026-09-20) is HIGHER than the 200-tok headline for both, not lower: the cache warms and the cold start amortizes over the longer run, so the short-run numbers were conservative. The 200-tok caveat is retired.
| Bonsai-27B PQ2_0 (dense) | 0.71 | 5.3 | - |

**Hard constraints.**
- 4 GB VRAM: cache slots, MTP head, KV and compute buffers compete. Qwen + head n=2: 36 slots = 3688 MiB and a CUDA OOM during generation (not at load); 30 slots = 3454 MiB fits [M]. The speculation side (head + draft context + recurrent rollback snapshots) costs ~190 MiB [M, notes.md]; every feature that adds VRAM (larger `n_rs_seq`, longer verify batches) must report its own cost.
- 16 GB RAM: CPU-side experts are ~10 GB of shmem; systemd-oomd kills the whole tmux scope on pressure; no builds or downloads next to a resident server; `--cache-ram 0`; preflight refuses to START a job below 12.5 GB available on the idle box (while a server runs, 2-4 GB available is normal) [M].
- The cache graph handles 1-4 token batches; larger verify batches leave the cache path [M].
- On GDN hybrids (Qwen3.6) a draft longer than `n_rs_seq` takes the checkpoint + replay path; `n_rs_seq = --spec-draft-n-max` only when a model drafter is listed, else 0 [V: `common/common.h` `need_n_rs_seq()`, `server-context.cpp`].
- Quality may not move: every change is either provably lossless (speculation, caching) or passes the quality gate (weights, quant, router changes).

**Metrics, all per run, all in the ledger:** decode tok/s and wall tok/s per workload, acceptance per drafter, accepted tokens per step, cache hit rate / uploads / evictions, VRAM, GPU util and W, PCIe rx/tx, DRAM GB/s, CPU %, MemAvailable min, swap max, git commit + dirty flag + full args, quality row when applicable.

## 3. Evidence ledger: alive, lead, dead

### 3.1 Alive and measured [M]
- GPU-resident expert cache (P1-P4): numerically sound (cache on == cache off on all three benchmarks), 57-65% hit rate, simulator predicted 59-64%. P4 overlap of CPU misses and GPU hits = +17-18%.
- MTP speculation: standalone Qwen head, acceptance 0.83-0.94, n=2 is the sweet spot; Gemma drafter n=2.
- K-quant experts are the largest single Gemma lever: CPU expert matvecs are compute-bound (`ggml_vec_dot_iq*_q8_K` ~49% of the profile); Q2_K_P holds quality within noise at n=50/41/70 (GSM8K 96 vs 100 is ~1.4 sigma).
- `-t 6` beats `-t 5`; `-lv 4` logging is free.

### 3.2 Dead [D]
- **T4, MoE-ifying Qwen3.8-27B to 3-4B active BY FFN-SPLITTING ITS OWN WEIGHTS.** (The goal itself is alive by a different route, see 3.3.) Always-active floor (attention + GDN + lm_head) is ~8.7B [I from V config]; FFN-splitting bottoms out at 9-10B active; the one real derivative (Whittle-MoE-27B-A17.8B, card verified) is 17.8B active and "is not the parent"; quality-preserving upcycling costs 200B-1T tokens [L]. **Reopen only if** someone publishes a <= 6B-active derivative with benchmark parity, or a depth/width-pruning + upcycle recipe under 5B tokens.
- **Retrofitting n-gram embedding modules (Engram 2601.07372, SCONE 2502.01637, PLE, Over-Tokenized 2501.16975) into an existing MoE.** All are pretraining-time architectures [L, IDs verified]. **Reopen only if** a retrofit-by-distillation result appears.
- **Uno (2609.04010) as a drafter on this box.** Draft pass = full-model forward over L tokens; on CPU experts that fans out to up to 8L experts per layer [I]. Its TV loss survives as T1.
- **`ngram-cache` (corpus lookup files) as shipped.** Hardcoded 8-token drafts, never saves the dynamic cache under the server, cross-request staleness bug (#27852) [L from the Qwen scout, file:line cited, spot-check before relying on it].

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

## 9. Sequencing

**Phase 0, in flight (box unattended, ~8-10 h):** Q-box. Mac side meanwhile, strictly one at a time in this order (WIP limit): W4 workload -> U3 series split -> N3a draft against mainline. `predict.py` is already written and smoke-tested.
**Phase 1, decide (one sitting, all from ledger rows):** G2 tree default, S1/S2 model defaults, G3 n-gram default, G4 prefetch go/no-go. Then S4: download Whittle Q3_K_M (disk check first, never next to a resident server), cache-off baseline, cache on, quality both ways. Checkpoint.
**Phase 2, runtime build (Mac code, box tests, one item at a time):** P4c warm-up -> N3 store -> P2 prefetch (if G4) -> N2/P3c (if N1 says long drafts pay). Each item: patch, identity check, A/B rows, commit. Checkpoint per item; forensic re-read after the third.
**Phase 3, Dave's box (needs Dave's OK and a time window):** T0 spike -> T1 -> T2, each ending in an artifact benched on the i5 box.
**Phase 4, compose and ablate:** the full matrix of section 10 on the final tree, larger-n quality pass, write-up.

Critical path: Q-box -> G2 (everything after lands on the chosen tree) -> P1/G4 and N1/G3 -> Phase 2. Phase 3 is parallel to Phase 2 once T0 is done, because it uses a different machine.

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

Checkpoint log (append: date, phase, what was verified, review findings):
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

## 13. References
Reports: `research/ngram-codesign-scout.md`, `research/ngram-llamacpp-scout.md`, `research/upcycle-scout.md`, `research/lit-scout-arxiv-survey.md`, `research/pr-scout-llamacpp-prs.md`, `research/moe-scout-abliterated-moe.md`, `research/qwen-mtp-scout.md`. Patches: `research/patches/`. Simulator and predictor: `research/scripts/moe-cache-sim/`.
Papers (arXiv IDs resolved 2026-09-19): SuffixDecoding 2411.04975, REST 2311.08252, CREST 2408.04678, Engram 2601.07372, SCONE 2502.01637, ReMoE 2605.27081, SpecPrefetch 2607.24787, MoE-SpeQ 2511.14102, SeqMoE 2609.12978, MoE cache evaluation 2608.07911, MTP self-distillation 2603.23911, Uno 2609.04010, MoE-SpAc 2603.09983, Speculating Experts 2603.19289, Sticky Routing 2607.08780, local routing consistency 2505.16056.
