# Research hunt: n-gram speculative decoding in llama.cpp (mainline `src/llama.cpp-mainline`)

Source of truth: local mainline tree (`src/llama.cpp-mainline`), GitHub API pulls dated 2026-09-19, arXiv API. All `file:line` refs are relative to `src/llama.cpp-mainline/`.

## TL;DR

1. **Yes — stacking is native today.** `--spec-type` accepts a comma-separated list; every speculative impl (n-gram and model-based) is instantiated as an ordered chain, n-gram first, model drafters last. `--spec-type draft-mtp,ngram-mod` gives: n-gram draft attempted first each step, MTP only when the n-gram store yields nothing (`common/speculative.cpp:2634-2650`, `2807-2873`). CLI order is irrelevant — the chain order is hardcoded.
2. It is first-non-empty-wins per round, **not** a union: if n-gram drafts 1 token, MTP idles that step. Max draft per drafter = 3 via `--spec-ngram-mod-n-max 3 --spec-ngram-mod-n-min 3` (defaults 64/48 are batch-killers for your 1-4-token verify graph; `--spec-draft-n-max 3` also raises `n_rs_seq` to 3, which matters on GDN — see §7).
3. Corpus-fed store (`ngram-cache` + `-lcs/-lcd` files) exists and the **server can load it**, but is hardcoded to draft up to 8 tokens with no flag (`common/speculative.cpp:2211`), has an open cross-request staleness bug on slots (#27852), and copies its static cache per slot in RAM. For your box, `ngram-mod`/`ngram-map-k` fed from the live prompt/session history is the practical route; a prebuilt corpus store needs a ~5-line patch or the external-corpus trick in §2.
4. Real-world gain (PR #27210, strongest published dataset): MTP-fixed-3 + ngram-mod vs MTP-fixed-3 alone — reasoning 51.65 vs 51.68, prose 55.90 vs 55.89, code 68.36 vs 69.28 tok/s (**neutral**), repetitive recall **324 vs 82 tok/s (≈4×)**. Expect free draft tokens only on repetition-heavy traffic; on chat it's ~free-but-nothing.

## 1. Speculation types and the n-gram variants

`--spec-type` values (`common/speculative.cpp:34-44`, confirmed in `tools/server/README.md:271`): `none`, `draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`. Types enum: `common/common.h:171-184`. Chain priority (highest first): ngram-simple → ngram-map-k → ngram-map-k4v → ngram-mod → ngram-cache → draft-simple → draft-eagle3 → draft-mtp → draft-dflash → draft-dspark (`common/speculative.cpp:2634-2650`).

### ngram-simple (`common/speculative_impl_ngram_simple`, speculative.cpp:1770-1813; algo `common_ngram_simple_draft`, ngram-map.cpp:49-121)
- **Structure:** none — stateless brute-force backward linear scan of the current token history per draft call (O(history·n) per step; no map, no cache).
- **Key:** the trailing `size_n` tokens + last sampled token. **Value:** the up-to-`size_m` tokens literally following a located match in history (verbatim copy).
- **Proposal:** newest matching occurrence searched first (scan from `cur_len - n - 1` downward, ngram-map.cpp:79-95); draft = the m-gram after the match. Quirk: a candidate whose remaining-copy length `< n_draft_min` (= `size_n`, default 12!) is discarded (ngram-map.cpp:96-102), so with defaults it drafts either ≥12 tokens or nothing — unusable at batch ≤ 4 unless `--spec-ngram-simple-size-n` is also lowered to ≤ 3.
- **Store updates:** implicit (reads prompt vector every call; no accept-feedback).
- **Memory:** ~zero persistent.
- **Flags:** `--spec-ngram-simple-size-n` (default 12), `--spec-ngram-simple-size-m` (default 48) (`common/arg.cpp:4304-4323`; defaults from `common_params_speculative_ngram_map`, `common/common.h:359-363`). `--spec-ngram-simple-min-hits` exists (arg.cpp:4325-4334) but **is dead code** — `common_ngram_simple_config` only carries `size_ngram`/`size_mgram` (`common/ngram-map.h:24-28`); init drops `min_hits` (`common/speculative.cpp:2669-2689`).

### ngram-map-k / ngram-map-k4v (speculative.cpp:1815-1867; algo ngram-map.cpp:133-536)
- **Structure:** per-seq `common_ngram_map` (ngram-map.h:61-96): a `key_map` hash array of 262144 × uint32 (1 MB/seq, `COMMON_NGRAM_HASH_MAP_SIZE`, ngram-map.h:33) mapping n-gram LCG hash (`hash = h*2654435761 + tok`, ngram-map.cpp:14-20) → index in history, plus a `keys` vector of `common_ngram_map_key {key_idx, stat_idx, key_num, values[4]}` (~64 B each; values record `value_idx`, occurrence count `value_num`, and learned `n_accepted`). History indices are invalidated on context regeneration/shrink in `common_ngram_map_begin` (ngram-map.cpp:136-243).
- **Key/value:** key = `size_n`(12)-gram + sampled token; value = the `size_m`(48)-gram(s) following historical occurrences. **-k** uses the single-following-m-gram view (one value slot, no hit-count gating — `key_only` path returns before the `min_hits` check, ngram-map.cpp:382-398, so `--spec-ngram-map-k-min-hits` is effectively inert for -k). **-k4v** enumerates up to 4 distinct m-gram values per key, drafts only the majority value and only if it dominates: abstains when `max_occur < 2*sum_occur` (ngram-map.cpp:495-498), gates on `key_num >= min_hits` (ngram-map.cpp:401-406).
- **Draft length adaptivity (both):** draft = `min(size_m, value.n_accepted)` — after each verify, the accepted count for that value is stored (ngram-map.cpp:520-536) and reused as the next draft-length cap. Requires history ≥ `2*size_n + size_m` before drafting anything (ngram-map.cpp:259-262).
- **Store updates:** lazily — new n-grams enter `key_map` during each `draft()` (ngram-map.cpp:337-363), `keys`/value stats grow incrementally, cleaned up only in `begin()`.
- **Memory:** 1 MB hash map × n_parallel + `keys` ∝ recurring n-grams (bounded by cleanup on regen).
- **Flags:** `--spec-ngram-map-k-{size-n,size-m,min-hits}` and `--spec-ngram-map-k4v-{size-n,size-m,min-hits}` (arg.cpp:4355-4402; defaults 12/48/1 for both types). Introduced by merged PR **#18471** (2025-12-29 → merged 2026-01-28).

### ngram-mod (speculative.cpp:1869-2042; store `common_ngram_mod`, ngram-mod.h/cpp)
- **Structure:** direct-mapped open-addressing hash table: `int32 entries[4 Mi]` = **fixed 16 MB** (allocated `mod(n_match, 4*1024*1024)`, speculative.cpp:1896), keyed by `idx = hash(tokens[0..n)) % size` with LCG multiplier `6364136223846793005` (`common/ngram-mod.cpp:14-23`); **one table shared across all sequences** (comment speculative.cpp:1872) — cross-talk at `--parallel > 1`.
- **Key/value:** key = `n_match`(24) tokens; value = **single next token** (collisions silently overwrite — `add()` clobbers, ngram-mod.cpp:25-31). Proposals are iterative: predicted token appended to the rolling n-gram window, rehashed, repeated until a hole or `n_max`.
- **Proposal path:** reconstructs the n-token prefix + `id_last`, walks forward up to `n_max` tokens; if the walk dies before `n_min` tokens it **throws the whole draft away**; otherwise keeps what it walked (speculative.cpp:1957-2010 — result-build loop; note the walk continues through *holes only up to `n_max`*, truncates at first miss if `≥ n_min` reached).
- **Store updates:** prompt ingested wholesale in `begin()` (minus trailing n), then new n-grams in ≥32-token chunks per draft call (speculative.cpp:1952-1958). **Resets**: full-table wipe if occupancy > 25% at `begin()` (speculative.cpp:1932-1940) or after 5 consecutive rounds with acceptance fraction < 0.25 (speculative.cpp:2015-2040, `LLAMA_TRACE` logs "low acceptance streak - resetting ngram_mod"). Ref PR **#19164** (merged 2026-01-30).
- **Flags:** `--spec-ngram-mod-n-match` (24), `--spec-ngram-mod-n-max` (64), `--spec-ngram-mod-n-min` (48) (arg.cpp:4274-4303). Warns if `n_match < 16` citing #19164 (speculative.cpp:1905-1909).

### ngram-cache (speculative.cpp:2044-2182; `common/ngram-cache.{h,cpp}`)
- **Structure:** three-level n-gram → histogram store per sequence: `unordered_map<common_ngram(4-slot token array), unordered_map<token,int32>>` for (a) context (current request, ingested lazily from prompt+deltas in `draft_one`, speculative.cpp:2120-2141), (b) dynamic (prior generations, loaded from file), (c) static (corpus, loaded from file). N-gram orders 1..4 (`LLAMA_NGRAM_MIN/MAX` ngram-cache.h:9-11); static level searched at order 2 (`LLAMA_NGRAM_STATIC`).
- **Proposal:** token-by-token greedy up to **hardcoded `n_draft = 8`** (speculative.cpp:2211 — "TODO get from config?"); each step tries context cache with lax thresholds {sample≥2 or n≥1; ≥50-66% dominance}, then dynamic cache with strict thresholds {4/3/2/2 samples; 66-75%}, then static-only with lax thresholds; static counts weight any candidate by ×100 (ngram-cache.cpp:59-63,120-135,146-178). Stops at first failed step or at 8.
- **Updates:** context cache from ingested tokens (ngram-cache.cpp:12-52); **server never saves** the dynamic cache — `save_static/save_dynamic` hardcoded `false` (speculative.cpp:2213-2216); `begin()` is a **no-op** (speculative.cpp:2112-2114) → known staleness bug #27852 (§7).
- **Memory:** whole static+dynamic maps are **deep-copied into every seq** at init (speculative.cpp:2096-2110) — RAM ≈ (distinct n-grams × ~150 B unordered_map overhead) × n_parallel. Big corpus files + parallel > 1 will hurt on 16 GB.
- **Flags:** only `-lcs/--lookup-cache-static`, `-lcd/--lookup-cache-dynamic` (arg.cpp:1621-1635, valid for `SERVER` + `LOOKUP` examples). Nothing auto-enables the type from those flags — you must pass `--spec-type ...,ngram-cache` explicitly (grep confirms sole writers of `params.speculative.types` are the `--spec-type` parser arg.cpp:4264-4273 and HF/GGUF auto-detection for model drafters arg.cpp:544-569).

## 2. Static vs dynamic lookup caches (corpus stores)

- **File format** (`common_ngram_cache_save/load`, ngram-cache.cpp:200-248): raw binary stream, no header/version: repeated records of `common_ngram` (4× int32 tokens = 16 B, unused slots = `LLAMA_TOKEN_NULL`) + int32 pair-count + pair-count × (int32 token, int32 count). Endian/host-order dependent, no vocab tag — the file is meaningless unless produced with the same tokenizer (implicit assumption stated nowhere; breakage is silent garbage drafts, low acceptance).
- **Building one:** `llama-lookup-create` (targets `llama-lookup{,-create,-merge,-stats}`, `examples/lookup/CMakeLists.txt:1-19`) tokenizes **`-p` prompt** with the loaded GGUF's tokenizer and writes only order-2 grams (`LLAMA_NGRAM_STATIC`..`STATIC`, lookup-create.cpp:34-41) to the path from `-lcs`. There is no corpus-file input flag — for a big corpus, feed chunks through repeated `llama-lookup-create -p <chunk>` runs and merge with `llama-lookup-merge part1.bin part2.bin ... merged.bin` (positional args, `lookup-merge.cpp:14-52`; merge sums counts). Prompt-size limits in practice force the chunking workflow (per-chunk tokenization via `llama-lookup-create` requires the text to survive `-p` on the command line; >few-MB chunks are awkward (inferred limitation, not a documented one)).
- **Dynamic cache:** `llama-lookup` (the standalone example) loads `-lcd`, and on exit merges context-cache → dynamic and saves it back (lookup.cpp:72-74, 220-222). Under `llama-server` the dynamic cache path is **loaded but never written** (hardcoded `save_dynamic=false`, speculative.cpp:2211-2216); PR **#22055** (open, 2026-04-17) proposes persisting it at shutdown and removing the hardcoded flags.
- **Can llama-server load them?** Yes — `-lcs/-lcd` are registered for `LLAMA_EXAMPLE_SERVER`, consumed by the `ngram-cache` impl at construction (speculative.cpp:2711-2718 → `create_state_ngram_cache` reads `params.ngram_cache.lookup_cache_*`). Requires `--spec-type` to include `ngram-cache`; a nonexistent/unreadable file is a hard `GGML_ABORT` (speculative.cpp:2101, 2112).
- **Size per million tokens:** no authoritative figure found. From the format: on-disk, one record per **distinct bigram**, minimum 28 B each (16+4+8) → for typical English/code corpora, hundreds of thousands of distinct bigrams per million tokens ⇒ ~10-20 MB/Mtok files (inferred arithmetic from ngram-cache.cpp:200-219, not measured). Runtime × n_parallel deep copies make RAM the binding constraint (unordered_map node overhead ≫ on-disk bytes; rough 5-10× file size (inferred)). Listed under "could not verify" too.

## 3. Combining two speculation sources in one run

**Supported, built into the framework — as a fallback chain, exactly "n-gram first, model drafter when the n-gram store has nothing".** Dispatch code:

- Chains across impls are declared in the draft-param contract: "`this flag is used to chain the drafts through all the available implementations — after the first successful draft from an implementation, we set it to false to prevent further drafts for that sequence`" (`common/speculative.h:54-58`).
- Init instantiates **every** enabled type as its own `impl` in fixed priority order, n-gram impls strictly ahead of model impls (`common/speculative.cpp:2634-2650`).
- `common_speculative_draft()` (`speculative.cpp:2807-2884`): loops impls; each fills results only for seqs still flagged `dp.drafting`; a non-empty result sets `dp.drafting=false`, records `impl_last[seq_id]`, truncates to `dp.n_max` (:2850-2857), and **breaks the chain for that seq** (per-round winner). Later impls are skipped for that seq — MTP's forward pass simply doesn't run that step.
- `common_speculative_accept()` (:2890-2926) routes acceptance stats to the winning impl and mirrors `accept(..., is_other=true)` to the losers so stateful n-gram stores can learn/clean (ngram_map tracks cross-verified acceptance via `common_ngram_map_accept` regardless of winner — speculative.cpp:1858-1866).
- Per-impl stats (`#mean acc len`, acc-rate-per-position) are printed per implementation with `LLAMA_TRACE` (`speculative.cpp:2946-2993`) — handy for measuring exactly how often n-gram preempts MTP on your traffic.

So: `llama-server ... --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 --spec-ngram-mod-n-max 3 --spec-ngram-mod-n-min 3` works today (and #28391 proposes making ngram-mod a default companion — §6). Your `--spec-type draft-mtp,ngram-mod` combo is confirmed working by reporter evidence in issue #27839 ("`--spec-type draft-mtp,ngram-mod` works on it" against a target with its own MTP layers).

Limitations / where a *different* hybrid would go:
- **No complement-union**: nothing pads an n-gram draft of 2 with MTP's continuation of positions 3-4 (would go in the per-seq loop of `common_speculative_draft` around :2854, plus accept-routing per producing impl — nobody's written it).
- **"Model-first, n-gram-fallback" is not flag-selectable**: chain order is the hardcoded `add_config_if_enabled` list (:2636-2650); reorder there to flip priority.
- **No round-robin alternation** tied to acceptance: closest existing thing is ngram-mod's internal reset-on-low-acceptance (:2015-2040) — after a reset its store is empty for a while, so MTP naturally takes those rounds. Effectively "alternate on quality" for free.
- Crashes when mixing `draft-mtp` with an **external** `-md` draft of another family: `cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP` leaks into the external draft-model context creation (`common_speculative_init_result` ctor, speculative.cpp:2533-2585 — `spec_mtp` flips ctx_type before deciding target-vs-draft model). Issues #27850 (closed with analysis; MTP ctx should be made against target model) and #27839 (**still open**, same root cause with dflash+mtp+ngram-mod). Irrelevant to you if MTP stays target-native and you only add ngram types (confirmed working combo), but **do not** bolt `-md` into a mixed list until #27839 lands.

## 4. Draft-length and no-draft controls (batch ≤ 4 ⇒ drafts ≤ 3)

Global cap: the server recomputes `dp.n_max` per round from remaining context only (`server-context.cpp:483-501`, injected at :3028-3036; chain-wide truncation `speculative.cpp:2850-2857`) — **no** batch/graph-derived clamp exists. So the 3-token ceiling is yours to enforce via per-type knobs; `common_speculative_n_max()` returns the **max over all enabled types** (speculative.cpp:2335-2369, :2371-2385), and verify-batch/output sizing uses `1 + n_draft` (`common_speculative_get_output_limits`, :2608-2617) — patched graph must therefore handle n-gram-produced batches too, not just MTP's.

Per type:
- **ngram-simple:** drafts exactly `min(size_m, room_after_match)` and *refuses* if that's `< size_n` (ngram-map.cpp:96-102); refuses outright with no historical match or history ≤ `n+m+1` (:67-71, 90-92). Config for you: `--spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 3`.
- **ngram-map-k/-k4v:** cap = `size_m`; natural no-draft gates: history `< 2n+m`, no key match, and (k4v only) insufficient hits `< min_hits` (:401) / ambiguous successor distribution `max < 2·others` (:495); draft length self-shrinks to the value's last accepted count (:386, :503) — a slow-learning confidence control. Set `--spec-ngram-map-k4v-size-m 3` (+ optionally `min-hits 2`).
- **ngram-mod:** cap `--spec-ngram-mod-n-max 3`; **`--spec-ngram-mod-n-min` is a cliff, not a floor** — a walk that stalls before `n_min` tokens is discarded entirely (speculative.cpp:1986-1993): leave `n_min = 3` to permit 1-3-token drafts, higher values suppress short drafts *by throwing them away*. No-confidence = hash miss; default 64/48 will regularly hand you 16-64-token batches ⇒ violates your graph; never launch with defaults on your rig.
- **ngram-cache:** hardcoded 8 (speculative.cpp:2211) — **no flag** (legacy `--draft*/--spec-ngram-*` aliases now `throw "removed"` pointing at the per-type flags, arg.cpp:4410-4460). Confidence gates = per-order sample-size and dominance thresholds (ngram-cache.cpp:59-63 and the ×100 static-weight heuristic :124) that abort the greedy walk early, so realized drafts are usually ≤8 and often <8, but the *ceiling* is uncappable without a one-line edit at :2211 (or patch `create_state_ngram_cache` to honor a param). Treat as unusable at batch ≤ 4 until that patch.
- MTP sibling stays at your current `--spec-draft-n-max 3` (was 2; 3 is exactly batch 4).

## 5. Measured numbers (from PR text/comments only)

- **PR #18471** (self-spec / ngram-simple+k4v, merged 2026-01-28): translation-favorable case gpt-oss-120b — acceptance **0.768 (2397/3120)**; Qwen3-235B heavily offloaded — **0.863 (2382/2760)**, "**from ~12 to ~21 tokens/s** ... **occurs only in favorable cases with large repeated sections**" — author-flagged, repetitive/code-edit workload. Explicit warning that generic benefit is situational. (https://github.com/ggml-org/llama.cpp/pull/18471)
- **PR #19164** (ngram-mod, merged 2026-01-30): bimodal behavior documented in comments — verbatim-code-repeat runs swing acceptance **0.92** (1416/1536) between runs with **0.0** when the low-acceptance-reset fires mid-task (bfroemel 2026-01-31); agentic/chat-with-thinking reports **0.05-0.63**, commenter "not sure it's not placebo" (jacekpoplawski 2026-02-01); CRLF-vs-LF tokenization mismatch pins acceptance at **0.11** until prompt normalized → **0.83** (EndeavoringOrb 2026-02-11). Repetitive/code-favored, weak-to-nil on ordinary chat. (https://github.com/ggml-org/llama.cpp/pull/19164)
- **PR #27210** (adaptive MTP, open 2026-08-17) — the one controlled ngram+MTP matrix (same-target combos; tok/s = predicted_per_second, columns reasoning/prose/code/recall; baseline none ≈ 30.0 across):
  | config | reasoning | prose | code | recall |
  |---|---|---|---|---|
  | C1 draft-mtp fixed 3 | **51.68** | 55.89 | 69.28 | 81.94 |
  | C5 ngram-mod alone | 30.03 | 30.10 | 29.98 | 292.88 |
  | C7 fixed 3 **+ ngram-mod** | 51.65 | 55.90 | 68.36 | **324.10** |
  | C3 adaptive 3..12 | 51.42 | **56.64** | **78.03** | 147.46 |
  | C6 adaptive **+ ngram-mod** | 51.48 | 56.61 | 73.06 | 321.25 |
  Takeaway: ngram-mod layered on MTP is **performance-neutral on chat/reasoning/prose/code** (±1%) and **+300% on memorized-recall replay**; n-gram alone does essentially nothing except recall. Author also notes deep/adaptive drafting regresses on old hardware — consistent with staying shallow (your n=3) on a 1650 SUPER. (https://github.com/ggml-org/llama.cpp/pull/27210)
- **Issue #27852** (open, 2026-08-28): ngram-cache on server degraded acceptance **86% → 11%** across successive requests on the same slot (staleness bug — §7), net slower than no speculation. Negative datapoint for `ngram-cache` specifically, not n-gram drafting in general.
- Chat-workload verdict across sources: n-gram drafting earns nothing and costs one hash-probe; repetitive/code/copy tasks are where the "free draft tokens" thesis holds.

## 6. 2026 PRs/issues touching n-gram drafting, suffix drafting, drafter combination

State/dates from GitHub API (queried 2026-09-19):

| # | title | type/state | created | summary |
|---|-------|-----------|---------|---------|
| 18471 | Add self-speculative decoding (no draft model required) | PR **merged** 2026-01-28 | 2025-12-29 | introduces ngram-simple / ngram-map-k / ngram-map-k4v |
| 19164 | spec : add ngram-mod | PR **merged** 2026-01-30 | 2026-01-28 | 16 MB direct-mapped hash n-gram drafter + reset heuristics |
| 22055 | spec : save the dynamic ngram cache file | PR open | 2026-04-17 | persists ngram-cache dynamic store at teardown; kills hardcoded save flags |
| 26283 | Suffix decode | PR open | 2026-07-29 | suffix-decoding (suffix-tree/self-drafting, https://suffix-decoding.github.io/) — per-request online tree only, no global corpus tree yet |
| 26499 | common : add env to override n_rs_seq | PR open | 2026-08-03 | `LLAMA_N_RS_SEQ=` escape hatch; motivating config is literally `--spec-type ngram-mod --spec-ngram-mod-n-min 7 --spec-ngram-mod-n-max 7` on a hybrid to dodge full-rollout checkpoints |
| 27210 | spec : add adaptive MTP draft depth (draft-mtp-adaptive) | PR open | 2026-08-17 | adaptive MTP depth 3..12; contains the C0-C8 MTP±ngram-mod benchmark table (§5) |
| 28391 | common : enable default speculative config | PR open | 2026-09-04 | server/CLI get ngram-mod **on by default**; `--spec-type draft-mtp` ⇒ `[ngram-mod, draft-mtp]`; opt-out `--no-spec-type ngram-mod` |
| 27839 | Combined --spec-type draft-dflash,draft-mtp,ngram-mod with -md fails at init | issue open | 2026-08-28 | MTP ctx_type leaks onto external -md model context → init abort |
| 27850 | combined --spec-type draft-mtp + external draft (-md) crashes at init | issue closed | 2026-08-28 | same root cause incl. memory-feasibility note for stacked draft contexts |
| 27852 | ngram-cache keeps per-slot context cache across requests (begin() no-op) — 86%→11% | issue open | 2026-08-28 | staleness bug in ngram-cache under server slot reuse |
| 28425 | recurrent/hybrid seq_rm partial-rollback (n_rs_seq) unreachable outside speculative decoding | issue open | 2026-09-05 | n_rs_seq=0 unless a model drafter is configured (§7) |
| 28019 | qwen4exp: multi-seq split replay corrupts recurrent state when rs rollback enabled | issue open | 2026-08-30 | why RS rollback stays gated behind spec on GDN-family archs |
| 28860 | SYCL demands 2 GB+ scratchpad when ngram-mod enabled | issue open | 2026-09-13 | odd but notes 16 MB table interaction with SYCL batch buffers |

Pattern to watch: **28391 makes ngram-mod+anything a de-facto default** — if merged, rebases of your patched tree will suddenly run two drafters unless you add `--no-spec-type ngram-mod` (once that flag exists) or `ngram-mod` disappears from the chain; also 26283 suffix-decoding would slot into the same chain as another ngram-class impl.

## 7. Setup-specific hazards (Qwen3.6 GDN hybrid, --parallel 1, checkpoints)

1. **Partial recurrent rollback is budgeted by `n_rs_seq`, and n-gram drafters do not raise it.** GDN/conv-state memory supports partial end-of-sequence rollback **only within the last `n_rs_seq` snapshots** (`src/llama-memory-recurrent.cpp:191-205` — bigger rollback → `seq_rm` returns false). The server flags such memory `COMMON_CONTEXT_SEQ_RM_TYPE_RS` (`common/common.h:993-996`) and treats any draft longer than `n_rs_seq` (creation: server-context.cpp:3070-3076) or any rollback deeper than `n_rs_seq` (accept: :3926-3928) as requiring a **full state checkpoint + replay** — expensive re-evaluation (server-context.cpp:3931-3957). Crucially, `need_n_rs_seq()` sets `n_rs_seq = draft.n_max` **only if a model drafter (MTP/Eagle3/DFlash/DSpark) is in the types list** (`common/common.h:391-400`); an n-gram-only chain ⇒ `n_rs_seq = 0` ⇒ every draft rollback is a full replay (matches issue #28425 and #26499's rationale; #28019 documents a family where multi-seq rs-replay is actively broken, which is why it's gated this way).
   **Practical consequence for you:** with `--spec-type draft-mtp,ngram-mod`, `n_rs_seq = --spec-draft-n-max`. Today you run n=2 → any n-gram draft of 3 forces the costly full-ckpt/replay path and eats the gain. Either keep `--spec-ngram-mod-n-max ≤ n_rs_seq`, or bump `--spec-draft-n-max 3` (raises n_rs_seq to 3 *and* verifies n=3 MTP — your acceptance numbers suggest that's affordable; graph already handles 4-token batches), or take the LLAMA_N_RS_SEQ override from #26499 (watch recurrent snapshot-buffer memory growth per extra rs slot — magnitude not verified).
2. **Verify-graph exposure:** target verification of *any* drafter's output flows through normal decode batching sized `n_parallel × (1 + common_speculative_n_max)` (speculative.cpp:2371-2385, 2608-2617; batch assembly server-context.cpp:3100+ feeding `handle_last_sampled_token`). Your 1-4-token-only expert-cache graph must therefore tolerate n-gram-originated batches; leftover defaults (64/48 mod, 8 cache, 48 map) will emit 5-65-token verify batches — worst case GGML assertion, realistically broken/offloaded slow path.
3. **Slot-reuse staleness:** `ngram-cache` impl `begin()` no-op (speculative.cpp:2112) → poisoned context-store across requests (#27852; its acceptance collapse dwarfs the win). `ngram-map` does rebuild/cleanup in `begin()` correctly (speculative.cpp:1833-1838 → ngram-map.cpp:136-243); `ngram-mod` re-ingests the prompt in `begin()` and wipes on >25% occupancy (speculative.cpp:1911-1943) — sound, minus hazard 5.
4. **`--parallel 1`:** nothing breaks. Cost notes if you ever go >1: ngram-mod is a **single table shared by all seqs** (speculative.cpp:1872) → cross-session n-gram pollution (mostly benign but pollutes acceptance); ngram-map allocates its 1 MB key_map **per seq**; ngram-cache deep-copies static+dynamic **per seq** (speculative.cpp:2096-2110) — RAM multiplier on a 16 GB host.
5. **Context checkpoints / slot state save:** the server's speculative checkpoint machinery (`spec_ckpt`, update/load PARTIAL_ONLY around server-context.cpp:3012-3055, 3078-3100) is drafter-agnostic and composes with n-gram types; drafter *internal* state (eagle3 boundary) serializes via `common_speculative_get_state`, with an explicit "TODO: support more than one speculative implementation having a state" (speculative.cpp:2930-2945). No n-gram impl implements `get_state/set_state` → on session/slot-state restore the n-gram stores are reconstructed from the re-supplied prompt at next `begin()` (fresh but valid for map/mod; another strike against ngram-cache; also means ngram-mod's learned next-token table and low-accept streak counters restart per restored session).
6. **Tokenizer pairing:** `llama-lookup-create` binds a store to whatever GGUF tokenized it (lookup-create.cpp:30-36) with no vocab stamp in the file (§2) — pin stores per model in your launcher, mismatches manifest as mystery 0%-acceptance runs, not errors.
7. **Minor:** PR #19164 thread documents that `mod.reset()` after 5 bad rounds wipes *even the prompt-derived* n-grams (builder acknowledged pattern; mitigation proposals in-thread not merged) — on a memory-jailed A3B with a GDN tendency toward repetition loops (your issue-shaped "#" reports, e.g. #23577 in the wild), expect occasional store churn; and `n_match < 16` triggers a quality warning (speculative.cpp:1905-1909) while raising collision pressure with lower n — for a 3-token output cap, tune n_max/n_min, not n_match.

## Recommended launch shape (validated against the code above)

```
llama-server ... \
  --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max 3 \
  --spec-ngram-mod-n-max 3 --spec-ngram-mod-n-min 3 --spec-ngram-mod-n-match 24 \
  [-np 1] [--poll ...] ; # measure with LLAMA_TRACE=1 → per-impl "#mean acc len / #acc rate/pos" split
```
Notes: `n_match 24` stays (its floor-warning zone is <16); the `--spec-draft-n-max 3` side-effect raises `n_rs_seq` to 3 so full n-gram drafts verify-roll back natively (hazard 1); if you'd rather keep MTP at n=2, cap ngram-mod to `n-max 2`. Avoid `ngram-cache` until #27852/#22055 land and `n_draft` becomes configurable. Re-test the flag surface after any rebase that pulls #28391 in.

## Could not verify

- **Bytes per Mtok** for lookup-cache files/RAM (formula inferred from record layout, marked above; no measured chart found). *(inferred)*
- **Whether `n_rs_seq` raises recurrent snapshot memory and by how much** — only qualitative statements in #26499/#24320 discuss the tradeoff; no formula located. *(inferred)*
- **Published ngram-map-k4v/k or ngram-simple acceptance charts on chat workloads** — #18471 demos are favorable-case by the author's own admission; nothing broader found via the searches run.
- **Canonical arXiv IDs** for suffix decoding / PLD papers: code cites project sites (PR #4484 → `apoorvumang/prompt-lookup-decoding`; PR #26283 → `suffix-decoding.github.io`); arXiv API queries (incl. re-tries) surfaced no citable entries for those exact phrases — one loosely-matching cs.CL entry ("Self-Speculation for Faster Reasoning Models", arXiv 2608.20359) appeared but was not opened or assessed. *(inferred relevance)*
- **Resolution state of #27839** beyond the posted analyses (patch attached to #27850 by reporter; not confirmed merged into the local tree).
- **Any measured numbers for `suffix decode` PR #26283** (body claims no benchmarks yet; not tested here).
- Whether the exact e613ef2 revision referenced in the hunt prompt equals this working copy (local checkout carries our `0af8ea3` GPU-expert-cache lineage on top of a mainline base; all cited symbols/line numbers verified against the files as-present on disk today).

Sources:
- https://github.com/ggml-org/llama.cpp/pull/18471
- https://github.com/ggml-org/llama.cpp/pull/19164
- https://github.com/ggml-org/llama.cpp/pull/4484
- https://github.com/ggml-org/llama.cpp/pull/22055
- https://github.com/ggml-org/llama.cpp/pull/26283
- https://github.com/ggml-org/llama.cpp/pull/26499
- https://github.com/ggml-org/llama.cpp/pull/27210
- https://github.com/ggml-org/llama.cpp/pull/28391
- https://github.com/ggml-org/llama.cpp/issues/27839
- https://github.com/ggml-org/llama.cpp/issues/27850
- https://github.com/ggml-org/llama.cpp/issues/27852
- https://github.com/ggml-org/llama.cpp/issues/28425
- https://suffix-decoding.github.io/

HUNT_DONE
