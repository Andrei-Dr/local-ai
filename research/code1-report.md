# CODE1 report — H2 harness upgrade

## Files touched

- `bench/box/specclient.py` (modified, +21/-6): `URL` env override (default `http://localhost:8099/v1/chat/completions`), `OUT` env override (default `/ai/bench/runs`) now used for both `makedirs` and the client-json path; rows gained `text` + `text_sha256` (sha256 hex over the utf-8 completion text) after the untouched `text_head`; `LONG=1` appends the `("long", …)` W4 prompt built from `w4_doc.txt` beside the script via `os.path.dirname(os.path.abspath(__file__))`, following the exact prompt template with three document-answerable questions (cache-path batch validity; GPU-vs-fan-out throttle; #28391 rebase hazard). Printed row format and every pre-existing JSON column left byte-identical.
- `bench/box/w4_doc.txt` (new, 6916 chars): verbatim `## 4.` header through end of `## 6.` of `SPEC.md`, cut at a paragraph boundary — see notes.
- `bench/textdiff.py` (new): per-kind `kind  IDENTICAL` / `kind  DIVERGES at char N: 'ctx' vs 'ctx'` (20-char context from the first differing char), `(head only)` marker + `text_head` fallback when either file lacks `text`, rc 0 all-identical else 1, stdlib only.
- `bench/tests/test_specclient.py` (new): stdlib unittest; threaded `ThreadingHTTPServer` on an ephemeral 127.0.0.1 port serving the canned OpenAI-style payload (incl. full `timings` block); subprocesses `specclient.py` with `URL`/`OUT`/`EDIT=1`/`LONG=1` and asserts 4 rows in order, `text`/`text_sha256` correctness, `text_head` intact, posted `long` prompt contains the document verbatim + template prefix, doc < 9000 chars, stdout = 4 lines; plus the two mandated textdiff tests (identical rc 0, divergent rc 1 asserting the exact string incl. char offset).

Untouched: everything else. `git status` shows only the four deliverables (`bench/box/traces/`, `research/scripts/moe-cache-sim/warmup.py` were already untracked in the tree before this task — not mine). No git commands run beyond read-only `status`/`diff --stat`. No network beyond 127.0.0.1; no ssh; system python untouched (all runs `~/dev/.venv/bin/python`).

## Gate output

`~/dev/.venv/bin/python -m py_compile bench/box/specclient.py bench/textdiff.py` → clean.

`~/dev/.venv/bin/python -m unittest discover -s bench/tests -v` tail:

```
test_four_rows_fields_and_long_prompt (test_specclient.ClientCase.test_four_rows_fields_and_long_prompt) ... ok
test_divergent_reports_char_offset (test_specclient.TextdiffCase.test_divergent_reports_char_offset) ... ok
test_identical_exits_zero (test_specclient.TextdiffCase.test_identical_exits_zero) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.675s
OK
```

Extra smokes (not file-producing): head-only fallback prints `code  DIVERGES at char 3: 'a' vs 'b' (head only)` rc 1; and with `EDIT`/`LONG` unset the client stays a 2-row run, keys exactly the ten originals plus the two new appended fields, stdout 2 lines — default-path compatibility held.

## Unsure / judgment calls

- **Doc cut point**: sections 4–6 are 13,288 chars; the nearest paragraph boundary ≤ 9000 is 6916 (the following block — dense section-6 list paragraphs without blank lines — spans past 9000 on its own), so `w4_doc.txt` ends at 6916 rather than ~8999. Allowed by the cut rule; all three question anchors verified present in the kept text (`Validity: k <= 4 only`, `throttled by expert fan-out`, `#28391`).
- **"through the end of ## 6."** interpreted as `## 4.` header line through the line before `## 7.` header.
- Question wording inside the fixed template was my choice; document-grounded one-liners, each quotable-verbatim for the "quote the exact sentence" requirement.
- textdiff prefix-only divergence (one text a strict prefix of the other) reports `DIVERGES at char N: '' vs ''` at the seam — follows the mandated line shape; no special case invented. Bad argv surfaces as a traceback rc≠0 (stdlib-terse); rc 0/1 reserved for comparison outcomes as specified. Exactly the two mandated textdiff tests are in the suite; head-only covered by smoke only, deliberately not a third test.
- Kept the pre-existing quirk of dumping client-json through an unclosed `open()` handle (flush-on-exit); cleanup additions went to the test file only, to avoid behavioral drift in the client.
- `POSTED` order assumed stable because the client is strictly sequential; the threading server adds no concurrency on this path.

CODE1_DONE
