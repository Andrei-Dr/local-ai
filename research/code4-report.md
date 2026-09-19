# CODE4 report — A/B report tool

## Files touched

- `bench/abreport.py` (new, ~120 lines, stdlib, terse style per `bench/textdiff.py`): `abreport.py LEDGER.jsonl BASE_REGEX TEST_REGEX [--md]`. Pipeline: load JSONL; keep `kind=="specbench"` + `completed` + non-empty `prompts`; dedupe per label keeping the newest `ts` (`>=`, so file order breaks ties); assign each surviving record to either/both arms by `re.search` on label; per-arm aggregate per prompt kind {decode_tps list, acceptance list} plus arm-level `moe_cache.hit_rate_pct`, `vram_mib`, commit set. Rows in first-seen order (base traversal then test), prompt kinds present in BOTH arms with numeric decode on both, then the `ALL` row. `delta = 100*(test-base)/base`; `noise = max(per-arm 100*(max-min)/mean)` across the two arms, `-` when both arms have n<=1; verdict `WIN/LOSS` past `max(noise, 1.0)`, else `flat` (lowercase as specced). Empty arm ⇒ stderr message + exit 2; success ⇒ exit 0 (also exit 2 on usage error). Footer: one line per arm with per-kind mean acceptance (`k -` where the ledger carries `acceptance: null`), mean cache hit, mean vram, deduped 9-char commits.
- `bench/tests/test_abreport.py` (new): 5 unittest cases via subprocess on temp ledgers — (a) exact strings `20.50 (n=2)` / `22.20 (n=2)` / `+8.29%` / `4.88%` / `WIN` + ALL row + commits line; (b) latest-by-ts (stale value absent, `+0.00%`, `flat`); (c) `completed:false` ignored (its 500 t/s absent) incl. the all-null-acceptance footer path; (d) empty arm rc 2 + `arm 'test'` on stderr; (e) `--md` first line `| prompt…`, second `|---…`.

## Test tail

```
$ ~/dev/.venv/bin/python -m unittest discover -s bench/tests -v  (tail)
----------------------------------------------------------------------
Ran 17 tests in 0.765s

OK
```

(17 = 5 new + 12 pre-existing; all green. `py_compile` clean.)

## Real-ledger smoke

`~/dev/.venv/bin/python bench/abreport.py bench/box/ledger.jsonl 'ab_.*_fork$' 'ab_.*_main$'` (exit 0):

```
prompt | base decode t/s | test decode t/s | delta % | noise % | verdict
-------|-----------------|-----------------|---------|---------|--------
code   | 34.57 (n=2)     | 36.63 (n=2)     | +5.96%  | 18.34%  | flat
reason | 34.80 (n=2)     | 35.86 (n=2)     | +3.05%  | 20.19%  | flat
ALL    | 34.69           | 36.24           | +4.50%  | 19.26%  | flat
base: 2 records | acceptance code -, reason - | cache hit 58.5% | vram 3393 MiB | commits f94da5a59
test: 2 records | acceptance code -, reason - | cache hit 58.5% | vram 3404 MiB | commits cfb1ecdc7
```

Reading of the real rows (facts from the output): the fork/main pair's cross-record spread is 18–20% — the +4.5/+6.0% deltas sit inside it, so `flat` is the honest verdict at n=2; `acceptance` is null in those records; both arms ran with a single build commit each and identical mean cache hit. The tool did exactly its job on this data: it refused to crown a winner the noise cannot support.

## Unsure / interpretations

- **ALL-row noise** isn't defined by the spec (there are no "repeats" across prompts): implemented as the mean of the numeric per-prompt noises, `-` when no prompt has repeats. Deliberately NOT propagated as "max" — mean keeps the ALL line from inheriting the noisiest single prompt; one line in the tool can change it if you prefer max.
- A prompt kind with no numeric `decode_tps` on an arm is skipped from rows (and from the ALL means); "present in both arms" is enforced on both presence and numbers. Decode 0 would render `n/a`/`flat` via the div-guard (unreachable in practice).
- ALL cells intentionally drop the `(n=…)` suffix (cell is a mean-of-means; per-kind ns remain visible).
- Records whose label matches BOTH regexes land in both arms (literal spec reading); harmless for disjoint anchored patterns.
- Commits truncated to 9 chars (the ledger's short `git.commit`); `commit_full` left alone. Usage error exits 2 like an empty arm (spec silent on the case).
- Plain mode prints a dash separator line under the header for legibility; `--md` mode is strict GitHub table then the same footer lines.

CODE4_DONE
