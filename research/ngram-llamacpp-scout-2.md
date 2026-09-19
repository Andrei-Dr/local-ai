# Hunt 2: N2 implementation path (n_rs_seq override) + #28391 rebase hazard

Companion to `research/ngram-llamacpp-scout.md` (hunt 1). Sources: on-disk mainline tree `src/llama.cpp-mainline`, GitHub API (queried 2026-09-19). Cite-format: `file:line` relative to the mainline tree, or PR/issue URL. Everything here re-verified against the API payloads saved in `/tmp/h2_*.json` during the hunt.

## Bottom line

- My hunt-1 cite of #26499 was **right**: it is an env override for `n_rs_seq`, still **open, 1 commit, +4/-0 in `common/common.h` only** — a `getenv` early-return inside `need_n_rs_seq()` (§1).
- The **clean minimal change** is not that PR: teach `need_n_rs_seq()` itself about enabled n-gram types (mirror `common_speculative_n_max()`), ~6 lines, no other plumbing (§2). `n_rs_seq` is **not purely a sizing knob** — it reallocates recurrent planes linearly, reclassifies the context's seq_rm type (routing non-spec code paths onto the RS rollback path too), forces a ubatch-split constraint, and buys a wider *single-use* rollback window; correctness is guarded by an arch whitelist that includes the Qwen3.5/3.6-MoE/exp families, but there is a **live stale-plane hazard** documented in rejected PR #29117 for any rollback issued after intervening single-token steps, and open evidence that the RS path outside speculative decoding is crash-prone (#28425, #28019).
- #28391 (default `ngram-mod`) is **still open** (not merged as of today). If a rebase pulls it: server/CLI silently get `ngram-mod n_min=48 n_max=64` — a 64-token draft ceiling riding on top of your `--spec-type draft-mtp` — plus type-list *merging* (`--spec-type draft-mtp` ⇒ `[ngram-mod, draft-mtp]`). Exact opt-out post-rebase: `--no-spec-type ngram-mod` (env `LLAMA_ARG_NO_SPEC_TYPE`) (§3).

## 1. PR #26499 — what it actually does

State from `api.github.com/repos/ggml-org/llama.cpp/pulls/26499`: **open**, not merged, created 2026-08-03, updated 2026-09-19, 1 commit, 1 changed file, +4/-0. It is about `n_rs_seq`, and it adds **neither a CLI flag nor a need_n_rs_seq semantic change** — only an env-var preemption at the top of the function (`common/common.h` @@ -384,6 +384,10):

```cpp
     uint32_t need_n_rs_seq() const {
+        if (const char* env = std::getenv("LLAMA_N_RS_SEQ"); env != nullptr) {
+            return std::stoi(env);
+        }
+
         bool needs_rs_seq = std::any_of(types.begin(), types.end(), ...);
```

Behavioral caveats visible from the hunk alone (facts from the diff + the wiring traced in §2):

- `std::stoi` unguarded → throwing on garbage/env whitespace, no bounds check; no CLI/doc surface; invisible in `--slots`/logs except the `n_rs_seq =` boot line (src/llama-context.cpp:327).
- It fires **regardless of whether any spec type is enabled** — and `common_context_can_seq_rm()` returns `COMMON_CONTEXT_SEQ_RM_TYPE_RS` whenever `llama_n_rs_seq(ctx) > 0` (common/common.cpp:1589-1592), so the env also re-routes **non-speculative** runs into the RS rollback classification, which per issue #28425 ("server: recurrent/hybrid seq_rm partial-rollback (n_rs_seq) unreachable outside speculative decoding — crash o…", **open**) is not exercised/tested territory.
- Does **not** leak into the speculative draft context: `common_speculative_init_result` hard-zeros `cparams.n_rs_seq = 0` for `ctx_dft` (common/speculative.cpp:2554). Target context is where it lands (common/common.cpp:1723).
- Arch clamp still applies: `n_rs_seq > 0` requested on a non-whitelisted arch is silently clamped to 0 (src/llama-context.cpp:115-119); whitelist includes `QWEN35`, `QWEN35MOE`, `QWEN4EXP` (src/llama-arch.cpp:1112-1128) — Qwen3.6's GDN build is one of the former two (inferred from arch naming in #28019/#27931 reports for this family; not confirmed against your GGUF from here).

Verdict: usable as a stopgap in your patched tree (cherry-pick is trivial, or just export `LLAMA_N_RS_SEQ` and hot-patch those 4 lines yourself), but it is a process-global hammer with a spec-unrelated blast radius (§2 hazard list).

## 2. Minimal change to make 16-token n-gram drafts legal; cost of a larger n_rs_seq

**Current single source of truth** (all reads on the local tree):

- `common/common.h:394-400` — `need_n_rs_seq()` = `draft.n_max` iff a model drafter ∈ types, else 0.
- `common/common.cpp:1723` — `cparams.n_rs_seq = params.speculative.need_n_rs_seq();` (target ctx only; `common/speculative.cpp:2554` zeroes it for the draft ctx).
- `src/llama-context.cpp:115-119` — arch whitelist clamp; `src/llama-arch.cpp:1112-1128` — whitelist members KIMI_K3, QWEN35, QWEN35MOE, QWEN4EXP, DEEPSEEK4, NEMOTRON_H(_MOE), LFM2(MOE), BAILINGMOE3.
- `src/llama-memory-recurrent.cpp:101` — physical layout: each per-layer `cache_r/cache_s` (+ `cache_ple_r`) tensor gets `n_rows = mem_size * (1 + n_rs_seq)`; `mem_size` = `max(1, n_seq_max)` for recurrent-only (src/llama-model.cpp:2548-2556) and `rs_size = max(1, n_seq_max)` for hybrid (src/llama-model.cpp:2605) — i.e. **one row-block per concurrent sequence, replicated (1 + n_rs_seq) times as rollback planes**.
- Rollback enforcement: `src/llama-memory-recurrent.cpp:191-205` — partial `seq_rm` succeeds **only** if `1 ≤ rollback ≤ n_rs_seq`, the plane is addressed as `cell_id = rs_idx*size + cell` (:803, consumed/restored via the index-shifted state copy `llama_memory_recurrent_context::s_copy`, :1311-1327), and **no rollback is already pending** (single-use until the next ubatch consumes it).
- Split constraint: with `n_rs_seq > 0`, ubatch splitting must keep each seq's trailing `1 + n_rs_seq` tokens in one ubatch or snapshots go invalid (src/llama-memory-recurrent.cpp:443-445 `[TAG_RECURRENT_ROLLBACK_SPLITS]`; identical clause for hybrid at src/llama-memory-hybrid.cpp ~:112-119).
- Server consumption points: `tools/server/server-context.cpp:3073,3076,3928` — `draft.size()/n_rollback > llama_n_rs_seq(ctx)` ⇒ escalate to full checkpoint+replay (the 24% tax you're trying to kill; also `common/common.cpp:1589-1592` classification).

**Minimal change (recommended):** extend `need_n_rs_seq()` to take the max with the n-gram ceiling of enabled n-gram types, mirroring the existing switch in `common_speculative_n_max()` (common/speculative.cpp:2335-2369):

```cpp
uint32_t need_n_rs_seq() const {
    uint32_t rs = /* existing model-drafter result (draft.n_max | 0) */;
    for (auto t : types) {
        if (t == COMMON_SPECULATIVE_TYPE_NGRAM_MOD)   rs = std::max(rs, (uint32_t) ngram_mod.n_max);
        if (t == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE) rs = std::max(rs, (uint32_t) ngram_simple.size_m);
        if (t == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K)  rs = std::max(rs, (uint32_t) ngram_map_k.size_m);
        if (t == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V)rs = std::max(rs, (uint32_t) ngram_map_k4v.size_m);
        if (t == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE)  rs = std::max(rs, 8u); // hardcoded n_draft, speculative.cpp:2211
    }
    return rs;
}
```

That's it — one function, `common/common.h:394`. Everything downstream (allocator sizing, RS classification, server thresholds, split constraint) is already driven off `llama_n_rs_seq()`; the arch clamp at `src/llama-context.cpp:116` is the safety net. With MTP at n=2 and `--spec-ngram-mod-n-max 16`, you get `n_rs_seq = 16`, no replay for n-gram rolls ≤ 16, **plus** side-effect benefits to MTP's own rollback window. Alternative B (`#26499` env) achieves the same value with worse blast radius (§1) — prefer A; you control the patch already.

**Not purely a buffer-sizing knob.** Raising `n_rs_seq` to 16:

1. **Memory, linear:** +1 unit adds exactly one full replica of the recurrent state grid — for your hybrid: `Σ over GDN layers (conv_state + ssm_state bytes per token-slot) × n_seq_max` per unit (`recurrent.cpp:101-120`). Read it directly off the boot log `RS buffer size … MiB / "size = … (… seqs … rs_seq)"` line (`recurrent.cpp:130-136`); unit cost = that buffer ÷ (1 + n_rs_seq). For a Qwen3-Next-class GDN layout that's order **tens of MiB per sequence per unit** (inferred from the state geometry, not measured here — 16 units × 1 seq ≈ low-hundreds-of-MiB, feasible at 16 GB (inferred)). Note the planes are eagerly zeroed today; PR #29121 (open, 2026-09-19) reports 12.8→8.4 GB RAM for 16 slots + spec on a laptop build *after* switching to lazy mmap-zero — evidence the zeroing cost of extra planes is real (their machine, their numbers, not transferable).
2. **Steady-state decode cost:** with `n_rs_seq > 0` the GDN state path takes the gather/copy path instead of in-place updates; #29121 measured "~27% of step time" in `get_rows`/`cpy` for GDN states on the old path (CPU, multi-seq; open PR). Your GDN layers run on CPU cores — going 2→16 could measurably tax **plain decode**, not just spec rounds (magnitude unverified for your box; benchmark before celebrating the 24% recovered).
3. **Rollback stays single-use:** widening the window doesn't allow two consecutive partial rollbacks without an intervening consuming decode (`recurrent.cpp:195-203` `pending` check) — the escalation logic is unaffected either way.
4. **Correctness:** same-round rollback (verify → rollback in the same step) is exact on whitelisted archs. But closed-unmerged PR **#29117** documents that **upstream restores plane `d` blindly**, and plane d is only the wanted state if no single-token step happened since the last multi-token ubatch wrote it — otherwise restore returns a stale state *silently* ("decoding continues from wrong state"); their index-shift fix + refusal logic was proposed 2026-09-19 and is closed unmerged. Spec-flow same-round rollback has m=0 and is fine; **any deferred/chained rollback strategy atop the widened window inherits this hazard until #29117-style provenance lands** — keep 16-token n-gram rounds rolling back in the round they verify (which the server already does). Related open hazards in this family: #28019 (multi-seq split replay corrupts recurrent state on qwen4exp with rs enabled — reason the arch behaves differently; argues for `--parallel 1` while experimenting), #27931 (crashes on qwen3_5_moe hybrid + spec), #24055/#25592 (checkpoints always invalidated on hybrid/recurrent — predates RS and feeds the same replay-tax story).
5. Batch-shape corollary (operational, not n_rs_seq): a 16-token n-gram draft verifies in a **17-token batch** — outside your patched 1-4-token expert-cache graph, and it trips the split-consistency constraint (# above) only in prefill-shaped batches (decode batch 17 ≤ n_ubatch 2048 is fine). The graph will simply fall back/offload the 17-column decode; confirm that path exists before betting on 16-token drafts (this repo's own cache graph handles 1-4 only, hunt 1 §7 note 2).

## 3. PR #28391 — ngram-mod as default drafter

From `pulls/28391`: **open**, not merged, created 2026-09-04, last updated 2026-09-08, 10 files. Not in the on-disk tree yet (verified: `common_speculative_default_config` absent, `spec_types_is_default` still compares against `{NONE}`).

If rebased in:

- Server **and CLI** bootstrap `params.speculative = common_speculative_default_config()` (arg.cpp hunk @@ -1401: `else if (ex == LLAMA_EXAMPLE_SERVER || ex == LLAMA_EXAMPLE_CLI)`), where the new `common_speculative_default_config()` (speculative.cpp hunk @@ -44) sets `types = { NGRAM_MOD }; n_match = 24; n_min = 48; n_max = 64;` — i.e. the **64-token default draft ceiling danger is now the out-of-the-box state**.
- Passing `--spec-type draft-mtp` **merges onto** the default instead of replacing it — their own `tests/test-arg-parser.cpp` assertions pin `--spec-type draft-mtp` ⇒ `[ngram-mod, draft-mtp]`. The merge keeps working with hunt 1 §3 dispatch (ngram-mod still outranks MTP in the chain).
- **Precise opt-out:** `--no-spec-type ngram-mod` (new flag, env `LLAMA_ARG_NO_SPEC_TYPE`), which removes the default type; `--spec-type none` also collapses to `{NONE}` (parser hunk @@ -4265: a lone `none` replaces everything). Detection of "user configured explicitly" moves to a `spec_type_specified` bool on `common_params_speculative` (common.h hunk @@ -369). Presets are updated accordingly (`docs/preset.md` switches `spec-default = 1` → `no-spec-type = ngram-mod`).
- `--spec-default` is deprecated to a no-op marker.

Post-rebase launch contract for your rig: keep the explicit `--spec-type draft-mtp[,ngram-mod…]` AND add `--spec-ngram-mod-n-max 3 --spec-ngram-mod-n-min 3` (or `--no-spec-type ngram-mod` if you want MTP-only), because defaults 48/64 ride in unnoticed otherwise — combined with §2 they'd also silently inflate `need_n_rs_seq` to 64 (≈64 replicas of the GDN state grid) once the Q2 change is in. That coupling makes "cap the ngram knobs in the launcher" a permanent invariant, not a suggestion.

## 4. Other PRs/items 2026 touching n_rs_seq configurability or the n-gram↔recurrent-replay interaction

| # | type | state (2026-09-19) | created | one-line | our relevance |
|---|------|--------------------|---------|----------|---------------|
| [26499](https://github.com/ggml-org/llama.cpp/pull/26499) | PR | **open** | 2026-08-03 | `LLAMA_N_RS_SEQ` env override in `need_n_rs_seq()` (+4/-0) | stopgap; §1 blast radius |
| [24320](https://github.com/ggml-org/llama.cpp/issues/24320) | issue | closed 2026-07-24 | 2026-06-08 | FR: runtime/per-slot n_rs_seq API (named MTP, Qwen 3.6) | origin of the configurability demand; no code landed from it |
| [28425](https://github.com/ggml-org/llama.cpp/issues/28425) | issue | **open** | 2026-09-05 | n_rs_seq/RS partial rollback unreachable **outside** spec decoding + crash on that path | why env-route is risky; keep spec enabled |
| [28019](https://github.com/ggml-org/llama.cpp/issues/28019) | issue | **open** | 2026-08-30 | qwen4exp multi-seq split replay corrupts recurrent state when rs rollback enabled | argues `--parallel 1` during RS experiments |
| [29117](https://github.com/ggml-org/llama.cpp/pull/29117) | PR | closed, **unmerged** | 2026-09-19 | exact snapshot rollback (index-shift + provenance + refuse-impossible); documents silent stale-plane restore upstream | correctness landmine for any deferred rollback strategy at wide windows; revisit if resurrected |
| [29121](https://github.com/ggml-org/llama.cpp/pull/29121) | PR | **open** | 2026-09-19 | in-place GDN recurrent state update on CPU; keeps old gather path when `n_rs_seq > 0` (refs #28841) | quantifies hidden per-token cost of n_rs_seq>0 on CPU-GDN (27% get_rows/cpy on their rig) |
| [28841](https://github.com/ggml-org/llama.cpp/issues/28841) | issue | **open** | 2026-09-13 | skip redundant recurrent gather when n_rs==1 && n_seqs==1 | same cost axis, n_rs_seq≥1 penalty |
| [29085](https://github.com/ggml-org/llama.cpp/pull/29085) | PR | **open** | 2026-09-18 | reserve graph nodes for LFM2 recurrent rollback | shows rollback path still churning graph-layer bugs |
| [28466](https://github.com/ggml-org/llama.cpp/pull/28466) | PR | closed **merged** | 2026-09-06 | Kimi-K3 recurrent-state rollback support (fixes #28461) | proves whitelist expansion pattern is active — relevant precedent for our family being supported |
| [28007](https://github.com/ggml-org/llama.cpp/pull/28007) | PR | **open** | 2026-08-30 | server falls back to full re-processing when memory rollback fails | alternative to widening n_rs_seq — makes the replay tax graceful instead of eliminating it |
| [28232](https://github.com/ggml-org/llama.cpp/pull/28232) | PR | **open** | 2026-09-02 | truncate speculative results at EOG before rollback | reduces needless deep rollbacks incl. GDN rounds |
| [25592](https://github.com/ggml-org/llama.cpp/pull/25592) / [#24055](https://github.com/ggml-org/llama.cpp/issues/24055) | PR/issue | **open/open** | 2026-07-12 / 2026-06-03 | checkpoint handling broken on hybrid/recurrent (checkpoints always invalidated) | background of the replay-tax bug class on your arch |
| [27931](https://github.com/ggml-org/llama.cpp/issues/27931) | issue | **open** | 2026-08-29 | crash on hybrid recurrent (qwen3_5_moe) w/ spec + checkpoints | same family; check against our patched build's crash behavior |
| [28391](https://github.com/ggml-org/llama.cpp/pull/28391) | PR | **open** | 2026-09-04 | default spec config (ngram-mod on by default, `--no-spec-type` opt-out) | rebase hazard, §3 |

No PR found — via `n_rs_seq`, `recurrent rollback speculative`, `GDN speculative`, `ngram draft rollback` searches — that changes `need_n_rs_seq()` to account for n-gram drafters. The gap hunt 1 identified is real and still owned by nobody; the Q2 six-liner would be novel work upstream (worth PR-ing ourselves with #27210-style tables).

## Could not verify

- **GDN state bytes/token for the actual Qwen3.6-35B-A3B GGUF** (layer count, conv-width, v-head geometry) — would need the model's GGUF metadata; formulas given, "tens of MiB/unit" is an order-of-magnitude from Qwen3-Next-class shapes. *(inferred)*
- **Which exact `LLM_ARCH` enum your Qwen3.6 GGUF reports** (QWEN35 vs QWEN35MOE vs qwen4exp) — both whitelist members, so moot for legality, but #28019's corruption note is arch-specific. *(inferred)*
- **#29117 closure reason** (rejected vs superseded vs withdrawn) — body rich, comments not pulled; state verified as closed/merged=False only. *(unverified)*
- **#29121 perf transferability** (27% gather overhead, 12.8→8.4 GB) — their laptop/single-channel-RAM, 16-seq configuration; direction credible, magnitude on 1650SUPER+i5 unmeasured. *(unverified)*
- Runtime cost curve of n_rs_seq>0 on GPU-resident GDN variants (PR is CPU-kernel-scoped per its own text: "new path is for CPU only"). *(unverified)*
- #28391 review momentum (updates stalled 2026-09-08; may or may not merge before your next rebase). *(unverified)*

## Sources

- PRs: 26499 · 28391 · 29117 · 29121 · 28466 · 29085 · 28007 · 28232 · 25592 · 27210 · links in the table above
- Tree: `src/llama.cpp-mainline` @ working copy with local expert-cache commits; spec/recurrent/server files as cited.

HUNT2_DONE
