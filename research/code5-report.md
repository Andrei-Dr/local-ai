# CODE5 report — top-5 leaderboards per dimension

## Files touched

- `bench/ledger2md.py` (modified): added module-level `smodel()` (the three-way name-shortening chain that was duplicated inline, now one function — main table output byte-identical); rows gained `n` per axis (needed by the score cells); new block between the legends and the best-config section generating `## Top 5 by dimension` (subsections `### math (GSM8K)`, `### code (HumanEval)`, `### knowledge (MMLU-Pro)`, `### speed (decode tok/s)`), `## Fastest config per model file`; best-config section now caps its eligible rank table at 5, dropped the `Excluded:` paragraph, and prints a `Closest misses` table (up to 5 ineligible configs having all three axes, speed desc, reasons column) in BOTH branches (eligible and no-eligible). `import re` added (MTP n parse). Speed-leaderboard plumbing: latest-by-ts completed-only specbench per label, max/mean/count over numeric prompt decode values, note = `slots S`/`no cache`, `MTP n=K` parsed from `--spec-draft-n-max` when `--spec-type` in args or `-md` is a standalone token (else `no spec`), VRAM MiB, git.commit verbatim. Sorting: `(missing, -pct, (no speed, -speed), label)` — missing axis renders `-` and sorts last, tie-break speed desc then label, per spec 4. Knowledge board = competitive tier (pct desc) ++ contaminated tier (pct desc) ++ missing rows, capped at 5 combined.
- `bench/tests/test_scoreboard.py` (extended): `sb()` specbench-record builder; `TopFiveCase` with the four specced tests (math order/cap/tie-notes-exactly-within-1SE; competitive-before-contaminated incl. bound-rendering reuse; speed board latest-per-label + completed-only + `MTP n=2`/`no spec`/`no cache`/`slots 96`; closest-misses reason text + `Excluded:` gone). Two existing PickerCase assertions adapted where the layout legitimately moved (spec (e) sanctioned this — details below).
- `bench/SCOREBOARD.md` regenerated (`LEDGER.md`/`BUILDS.md` also rewritten by the same run — content unchanged, git sees them identical).

## Two existing assertions adjusted (intent kept)

1. `ranks = md.split("## Best config",1)[1]...` — the rank-table slice now additionally splits at `Closest misses` first, because the misses table lives in the same section; the assertion's meaning ("contaminated never appears in the ELIGIBLE rank table") is unchanged and still enforced.
2. `assertIn("Excluded: ...")` became `assertIn("`sx_zerp`", misses_section)` + `assertIn("knowledge contaminated", misses_section)` — the content moved into the misses table per spec 3.
(`LatestRecordCase.assertNotIn("(<=", ...)` etc. all still pass untouched — competitor cells are `80.0 ±4.7 (n=70)`, no parens.)

## Test tail

```
$ ~/dev/.venv/bin/python -m unittest discover -s bench/tests -v  (tail)
----------------------------------------------------------------------
Ran 21 tests in 0.764s

OK
```

(21 = 4 new + 17 pre-existing; all green. py_compile clean.)

## Regenerated SCOREBOARD.md sections (real ledger)

```
## Top 5 by dimension

### math (GSM8K)

| rank | config | model | score | note |
|---|---|---|---|---|
| 1 | `q36_iq2m_cache48` | Qwen3.6-35B-A3B-IQ2_M | 100.0 ±0.0 (n=50) |  |
| 2 | `g4_q3km_cache15` | Gemma4-26B-A4B-Q3_K_M | 100.0 ±0.0 (n=50) | tie with #1 within 1 SE |
| 3 | `g4_iq3m` | Gemma4-26B-A4B-IQ3_M | 100.0 ±0.0 (n=50) | tie with #1 within 1 SE |
| 4 | `g4_iq3m_cache16` | Gemma4-26B-A4B-IQ3_M | 100.0 ±0.0 (n=50) | tie with #1 within 1 SE |
| 5 | `g4_q2kp_cache19` | Gemma4-26B-A4B-Q2_K_P | 96.0 ±2.8 (n=50) |  |

### code (HumanEval)

| rank | config | model | score | note |
|---|---|---|---|---|
| 1 | `q36_iq2m_cache48` | Qwen3.6-35B-A3B-IQ2_M | 95.1 ±3.4 (n=41) |  |
| 2 | `g4_q2kp_cache19` | Gemma4-26B-A4B-Q2_K_P | 92.7 ±4.1 (n=41) | tie with #1 within 1 SE |
| 3 | `q36_iq2m` | Qwen3.6-35B-A3B-IQ2_M | 92.7 ±4.1 (n=41) | tie with #1 within 1 SE |
| 4 | `g4_iq3m_cache16` | Gemma4-26B-A4B-IQ3_M | 92.7 ±4.1 (n=41) | tie with #1 within 1 SE |
| 5 | `g4_q3km_cache15` | Gemma4-26B-A4B-Q3_K_M | 90.2 ±4.6 (n=41) |  |

### knowledge (MMLU-Pro)

| rank | config | model | score | note |
|---|---|---|---|---|
| 1 | `g4_iq3m_cache16` | Gemma4-26B-A4B-IQ3_M | 81.4 ±4.6 (n=70) |  |
| 2 | `g4_q2kp_cache19` | Gemma4-26B-A4B-Q2_K_P | 80.0 (<=81.4) ±4.8 (n=70) | tie with #1 within 1 SE |
| 3 | `q36_iq2m_cache48` | Qwen3.6-35B-A3B-IQ2_M | 77.1! (<=87.1) ±5.0 (n=70) | floor only, not ranked against clean rows |
| 4 | `g4_q3km_cache15` | Gemma4-26B-A4B-Q3_K_M | 75.7! (<=81.4) ±5.1 (n=70) | floor only, not ranked against clean rows |
| 5 | `distill_iq2m` | Qwen3.8-35B-A3B-Distill-IQ2_M | 72.9! (<=80.0) ±5.3 (n=70) | floor only, not ranked against clean rows |

### speed (decode tok/s)

| rank | config | model | score | note |
|---|---|---|---|---|
| 1 | `ng_g4q2k_mtp2` | Gemma4-26B-A4B-Q2_K_P | 55.6 (mean 49.9 over 3 prompts) | slots 15, MTP n=2, VRAM 3306 MiB, commit f94da5aa9fdf |
| 2 | `ng_g4q2k_ngmod16_mtp2` | Gemma4-26B-A4B-Q2_K_P | 55.3 (mean 50.1 over 3 prompts) | slots 15, MTP n=2, VRAM 3306 MiB, commit f94da5aa9fdf |
| 3 | `st_g4q2k_c15_mtp2` | Gemma4-26B-A4B-Q2_K_P | 50.5 (mean 49.4 over 2 prompts) | slots 15, MTP n=2, VRAM 3306 MiB, commit f94da5aa9fdf |
| 4 | `ng_q36_c28_ngmod3_mtp3` | Qwen3.6-35B-A3B-IQ2_M | 49.5 (mean 45.4 over 3 prompts) | slots 28, MTP n=3, VRAM 3440 MiB, commit f94da5aa9fdf |
| 5 | `ng_q36_ngmod2_mtp2` | Qwen3.6-35B-A3B-IQ2_M | 49.0 (mean 46.5 over 3 prompts) | slots 30, MTP n=2, VRAM 3454 MiB, commit f94da5aa9fdf |

## Fastest config per model file

| rank | config | model | score | note |
|---|---|---|---|---|
| 1 | `ng_g4q2k_mtp2` | Gemma4-26B-A4B-Q2_K_P | 55.6 (mean 49.9 over 3 prompts) | slots 15, MTP n=2, VRAM 3306 MiB, commit f94da5aa9fdf |
| 2 | `ng_q36_c28_ngmod3_mtp3` | Qwen3.6-35B-A3B-IQ2_M | 49.5 (mean 45.4 over 3 prompts) | slots 28, MTP n=3, VRAM 3440 MiB, commit f94da5aa9fdf |
| 3 | `distill_c48` | Qwen3.8-35B-A3B-Distill-IQ2_M | 39.1 (mean 38.7 over 2 prompts) | slots 48, no spec, VRAM 3304 MiB, commit f94da5a |
| 4 | `p4_g4q3k_c11_mtp2` | Gemma4-26B-A4B-Q3_K_M | 37.2 (mean 36.9 over 2 prompts) | slots 11, MTP n=2, VRAM 3382 MiB, commit f94da5a |
| 5 | `p4_g4_c12_mtp2` | Gemma4-26B-A4B-IQ3_M | 29.7 (mean 28.9 over 2 prompts) | slots 12, MTP n=2, VRAM 3362 MiB, commit f94da5a |
| 6 | `b75_best` | Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP | 5.3 (mean 5.2 over 2 prompts) | no cache, MTP n=2, VRAM 3590 MiB, commit 290857f |
| 7 | `q38_ngl16` | Qwen3.8-27B-i1-IQ3_M | 1.4 (mean 1.4 over 2 prompts) | no cache, no spec, VRAM 3628 MiB, commit 290857f |

## Best config (fastest that holds quality)   [updated part]

**WINNER: `g4_iq3m_cache16` (Gemma4-26B-A4B-IQ3_M) at 29.7 tok/s** — GSM8K 100.0, HumanEval 92.7, MMLU-Pro 81.4.

| rank | config | speed tok/s | math | code | knowledge |
|---|---|---|---|---|---|
| 1 | `g4_iq3m_cache16` | 29.7 | 100.0 | 92.7 | 81.4 |

Closest misses

| rank | config | speed tok/s | why not |
|---|---|---|---|
| 1 | `g4_q2kp_cache19` | 55.6 | math 96.0 >1sigma below best 100.0 |
| 2 | `q36_iq2m` | 49.5 | knowledge contaminated (cut-off band wider than 1 SE) |
| 3 | `q36_iq2m_cache48` | 49.5 | knowledge contaminated (cut-off band wider than 1 SE) |
| 4 | `distill_iq2m` | 39.1 | knowledge contaminated (cut-off band wider than 1 SE) |
| 5 | `g4_q3km_cache15` | 37.2 | knowledge contaminated (cut-off band wider than 1 SE) |
```

(The misses table is reproduced exactly as rendered.)

Real-data observations: the math axis saturates at 100.0 (SE 0 → exact-equality "ties" all flagged, 96 loses with no mercy — the 0-SE binomial edge behaving as designed); the whole 55-tok/s class is barred from winning best-config purely by knowledge contamination, which the Closest misses table now surfaces next to the 29.7 winner instead of hiding in a paragraph.

## Unsure / interpretations

- Math rows at 100.0 have SE 0.0 → "within 1 SE" degenerates to exact tie (correct degeneracy, but visually everything-100 is one big tie cluster; the code-axis tie cluster at 92.7 behaves normally).
- The 5-row cap on the knowledge board applies to the CONCATENATED tiers: ≥5 competitive rows would evict contaminated ones from view (they stay visible in the main table). Literal spec reading; changeable in one place.
- Speed leaderboard is completed-only per spec 1.d, while the main table's tuned-best speed column keeps its pre-existing behavior of pooling any specbench row incl. completed-less ones (left untouched deliberately — changing it would alter existing outputs beyond this task). Flagging as the one deliberate cohort asymmetry.
- `MTP n=?` renders when a spec flag exists but no parsable `--spec-draft-n-max` (legacy alias `--spec-draft-max` not matched — current harness writes the long form; easy to widen if old args show up in the ledger).
- "over 1 prompt" singularizes (cosmetic deviation from the literal "over P prompts" template). `VRAM -` placeholder for missing vram; commit printed verbatim (ledger mixes 7/12-char forms — passthrough, not normalized).
- Closest misses shows in BOTH branches (including the "no config is eligible" case); the old Excluded paragraph only printed when elig was non-empty. Strictly-more-information per the section's purpose.

CODE5_DONE
