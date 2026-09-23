# research/

Scout reports, design notes and the llama.cpp patch series. Index generated from the files themselves.

<!-- BEGIN GENERATED: research-index (bench/docgen.py; edit the source, not this block) -->
**Living documents** (kept current):

- [design-harmony-ledger.md](design-harmony-ledger.md) — The per-link byte ledger ("harmony") — decode after a 9.3k-token prompt, STABLE, MTP n=2 (2026-09-23)
- [patches/README.md](patches/README.md) — llama.cpp patches — what they are and what they do (2026-09-24)
- [patches/SERIES.md](patches/SERIES.md) — llama.cpp patch series — status ledger (the table is generated from series.toml) (2026-09-23)

**Dated snapshots** (true as of the date shown; never updated, newest first):

- [STATUS-2026-09-23.md](STATUS-2026-09-23.md) — Status 2026-09-23 14:18 — LEGACY -> STABLE (this morning) -> STABLE (now) (2026-09-23)
- [opt-hunt-2026-09-23.md](opt-hunt-2026-09-23.md) — Optimization hunt (2026-09-23) — small levers the big-hardware world skips (2026-09-23)
- [scout-2026-09-23-outside-box.md](scout-2026-09-23-outside-box.md) — Outside-the-box speedup scout — 2026-09-23 (2026-09-23)
- [HANDOFF-2026-09-21-queue-watch.md](HANDOFF-2026-09-21-queue-watch.md) — Handoff, 2026-09-21 evening: the box grinds for ~2 days, nothing large is left to code. For whoever watches the queue. (2026-09-21)
- [scout-2026-09-21-bounded-sparse-attention.md](scout-2026-09-21-bounded-sparse-attention.md) — Bounded Sparse Attention for Long-Context Decode (2026-09-21)
- [scout-2026-09-21-dense-as-moe.md](scout-2026-09-21-dense-as-moe.md) — Dense-as-MoE: Running Dense LLMs with Contextual Neuron Subsets (2026-09-21)
- [MIGRATION-flatten-bench.md](MIGRATION-flatten-bench.md) — Migration: flatten repo bench/box/* -> bench/*, then make /ai on the box a git checkout (2026-09-20)
- [code1-report.md](code1-report.md) — CODE1 report — H2 harness upgrade (2026-09-20)
- [code2-report.md](code2-report.md) — CODE2 report — scoreboard: truncation as a bound, not a veto (2026-09-20)
- [code3-report.md](code3-report.md) — CODE3 report — H2 larger-n quality sets (2026-09-20)
- [code4-report.md](code4-report.md) — CODE4 report — A/B report tool (2026-09-20)
- [code5-report.md](code5-report.md) — CODE5 report — top-5 leaderboards per dimension (2026-09-20)
- [code6-report.md](code6-report.md) — CODE6 report — job reports from the ledger (2026-09-20)
- [code7-report.md](code7-report.md) — CODE7 report — over-refusal set from published benchmarks (2026-09-20)
- [kv-tiering-scout.md](kv-tiering-scout.md) — KV Cache Tiering: Why It Does Not Work Like MoE Expert Caching (and What Does) (2026-09-20)
- [ngram-llamacpp-scout-2.md](ngram-llamacpp-scout-2.md) — Hunt 2: N2 implementation path (n_rs_seq override) + #28391 rebase hazard (2026-09-20)
- [scout-2026-09-20-kvquant-lowbit.md](scout-2026-09-20-kvquant-lowbit.md) — Scout digest 2026-09-20: TurboQuant / rotated KV, and 1-2 bit weights (3 Sonnet scouts; claims are THEIRS unless marked [V]) (2026-09-20)
- [lit-scout-arxiv-survey.md](lit-scout-arxiv-survey.md) — Literature Survey: Decode-Speed Techniques for Ternary-Bonsai-27B on a 4 GB Turing Box (2026-09-19)
- [moe-scout-abliterated-moe.md](moe-scout-abliterated-moe.md) — Abliterated MoE GGUF shortlist for expert-offloaded llama.cpp (2026-09-19)
- [ngram-codesign-scout.md](ngram-codesign-scout.md) — N-gram MoE Co-Design Scout Report (2026-09-19)
- [ngram-llamacpp-scout.md](ngram-llamacpp-scout.md) — Research hunt: n-gram speculative decoding in llama.cpp (mainline `src/llama.cpp-mainline`) (2026-09-19)
- [pr-scout-llamacpp-prs.md](pr-scout-llamacpp-prs.md) — llama.cpp PR / issue mining for decode tok/s on the GTX 1650 SUPER box (2026-09-19)
- [qwen-mtp-scout.md](qwen-mtp-scout.md) — Qwen3.6-35B-A3B MTP head scout (2026-09-19)
- [upcycle-scout.md](upcycle-scout.md) — Qwen3.8-27B → small-active MoE: scout report (2026-09-19)
<!-- END GENERATED: research-index -->
