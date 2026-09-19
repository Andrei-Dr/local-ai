# CODE7 report — over-refusal set from published benchmarks

## Files touched

- `bench/qual/fetch.py` (modified): `--overrefusal N` (default 0 = skip; N per source) → `OUT_DIR/overrefusal.jsonl`, items `{id, source, category, prompt}` exactly as specced (`orbench/<abs-row-index>`, `xstest/<dataset id>`). New injectable-selection functions `select_overrefusal(rows_fn, n, pause)` + `stratified_pick` + `_vdc`; `_get` grew an optional `pause` (unchanged behavior for existing callers). Source constants hard-pin the verified ids: `("bench-llm/or-bench", "or-bench-hard-1k", "train")` and `("Paul/XSTest", "default", "train")`; the toxic config never appears anywhere in code. XSTest keeps a row only when `str(label).lower() == "safe"`.
- `bench/qual/qual.py` (modified): `overrefusal` set wired in — `MAX_TOKENS["overrefusal"]=256`; `prompt_for` returns the item prompt verbatim as the sole user message (ask() already sends no system prompt); `score("overrefusal") = not refused(text)`; `refused()` = case-insensitive substring scan of the spec's 17 standard markers over the first 300 chars, with U+2019 normalized to `'` (models emit smart quotes). Not in default `--sets`; runs only when named. Set summary's `pct` is compliance by construction (ok = complied), plus `by_source: {orbench|xstest: {n, compliant, pct}}` derived from the id prefix. Docstring notes it.
- `bench/tests/test_overrefusal.py` (new, 4 cases): counts/balance/nested-prefix (N=50 prefix of N=100 in both sources; all categories present, per-group counts within 1; selection reaches past mid-split proving spread), safe-only filter (unsafe ids provably absent + degenerate empty source no-crash), `refused()` positives incl. mixed case + smart-quote + negatives, score/prompt glue.
- `bench/qual/data_h2/overrefusal.jsonl` (new data, 200 items) from the required real fetch.

No prompt or answer text anywhere in this report or in git beyond the fetched set itself; `src/` untouched; no git mutations.

## Dataset verification (what was actually checked through the API)

| what | URL | result |
|---|---|---|
| or-bench configs | `https://datasets-server.huggingface.co/splits?dataset=bench-llm%2For-bench` | configs `or-bench-80k`, **`or-bench-hard-1k`** (train), `or-bench-toxic` — hard-1k selected, toxic never touched |
| or-bench columns | `.../rows?dataset=bench-llm/or-bench&config=or-bench-hard-1k&split=train&offset=0&length=1` | columns `prompt`, `category`; `num_rows_total = 1319` |
| XSTest mirrors | `.../splits?dataset=walledai%2FXSTest` → **HTTP 401 (gated/unauthorized)**; `.../splits?dataset=Paul%2FXSTest` → config `default`, split `train` | `walledai/XSTest` unusable via API (not guessed past), `Paul/XSTest` used |
| XSTest columns | `.../rows?dataset=Paul/XSTest&config=default&split=train&offset=…` at offsets 0/90/180/270/360/440 (10 rows each) | columns `id, prompt, type, label, focus, note`; `label` values observed `safe`/`unsafe` (filter folds case); `id` unique over sampled window (ints 1..450); `num_rows_total = 450` |

Selection (published-items-only, nothing authored): per source, rows are grouped by category (`category` / `type` column), groups ordered alphabetically, and inside a group consumed in **van der Corput bit-reversed index order**; rounds over groups produce one fixed global sequence and the file is its first N per source. Properties: any smaller N is a prefix of a larger N (nested like the other sets), prefixes stay balanced across groups within 1, and consumption order is evenly spread over the split rather than head-loaded. 429s occurred and were absorbed by the built-in retry/pacing.

## Incident worth knowing (auto-corrected)

The first fetch run per the deliverable line (`--overrefusal 100 data_h2` with no other flags) deterministically **rewrote** `data_h2/{gsm8k,mmlu_pro}.jsonl` back to default sizes (50/70) — fetch.py dumps every set on every run. Caught from the output tail ("50 items"), immediately re-ran `fetch.py --gsm8k 200 --mmlu-rows 20 --overrefusal 100 bench/qual/data_h2`; verified afterwards that the first-50 gsm8k and first-70 mmlu ids still match `bench/qual/data/` (extended H2 sizes 200/280 restored, content deterministic and identical algorithm → no prefix churn). Structural lesson for whoever drives the box: always pass the full flag set for `data_h2`, or (better follow-up) make fetch.py refuse to shrink an existing set file unless told to. Not implemented here — outside the specced surface.

## Fetch result (counts only)

`bench/qual/data_h2/overrefusal.jsonl`: 200 items, 0 duplicate ids.

- orbench: 100 — categories deception/harassment/harmful/hate/illegal/privacy/self-harm/sexual/unethical/violence, **10 each**; selected absolute-row span 0..1317 (split-wide coverage)
- xstest: 100 — categories definitions/figurative_language/historical_events/homonyms/nons_group_real_discr/privacy_fictional/privacy_public/real_group_nons_discr/safe_contexts/safe_targets, **10 each**; dataset-id span 1..425
- Extended-set prefix integrity re-verified intact (True/True).

## Test tail

```
$ ~/dev/.venv/bin/python -m unittest discover -s bench/tests -v  (tail)
----------------------------------------------------------------------
Ran 31 tests in 0.910s

OK
```

(31 = 4 new + 27 pre-existing; py_compile clean.)

## Unsure / interpretations

- "Evenly spaced, category-stratified" has no single canonical definition; the vdc-round-robin above satisfies spread + balance + nestedness simultaneously. The real data cooperated exactly (10 categories × 10 in both sources).
- XSTest label vocabulary verified on a 60-row scatter only (`safe`/`unsafe` lowercase); the filter admits `label=="safe"` under any casing and silently drops anything else — conservative direction is "fewer items, never unsafe ones," which is the right failure polarity here. The full-fetch selected 100 without hitting a shortage, implying ≥ the needed safe supply (paper's 250 figure consistent; exact safe total not queried — cost of 5 more pages, no decision rides on it).
- `source` values are `"orbench"` / `"xstest"` mirroring the id prefixes (qual's `by_source` derives from the id prefix anyway; the field is documentary).
- Fixture category/type names in tests are synthetic guesses; real taxonomy verified post-fetch (above) — the tests assert properties (balance/nested/filter), not names, so they don't drift with the dataset.
- Full-population fetch per source costs ~14+5 API requests per run (vs the light MMLU page sampling) — rate-limit retries proved sufficient; a `--overrefusal` bump to e.g. 250 would need at most 3 extra XSTest pages.

CODE7_DONE
