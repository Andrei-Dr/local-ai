# Handoff, 2026-09-21 evening: the box grinds for ~2 days, nothing large is left to code. For whoever watches the queue.

State of record: `SPEC.md` section 1.0 (queue + verdicts), `notes.md` tail (every result with its reading). This file = decision
rules, so results get RECORDED AND READ BY RULE, not re-interpreted. Standing rules in `SPEC.md` section 0 and CLAUDE.md apply
(nothing runs beside a benchmark; stop the queue only with `systemctl stop ai-queue`; reorder with `/ai/bench/qorder.sh ID...`;
one-shot waiters via `/ai/bench/watch.sh`; never force-push). **Disk housekeeping is yours, not Andrei's**: when `/` gets tight, `mv /ai/models/<f> /mnt/md0/models-cold/<f> && ln -s /mnt/md0/models-cold/<f> /ai/models/<f>` (leave the symlink so job scripts keep working, verify it resolves), and move bulk build data there the same way. Do not ask. Cross-filesystem copies take minutes — background them, never beside a running benchmark. Ask only before destroying the last copy of something (a downloaded model with no cold copy, a result or ledger file).

Queue order now: whittle1 (running) -> cpu1bench2 -> hq1_iq2m (last ~11 AIME items) -> dl_stock -> dense1 -> hq1_stock -> qx3 ->
lq1 -> lq2 -> hq1_k2 -> bonsai1_easy -> bonsai1_hard. After each job: pull logs + results + ledger, commit, push, one dated
notes.md entry, update the SPEC row. Report to Andrei in a few lines.

| job | read | rule |
|---|---|---|
| whittle1 | t/s at cache 0 / 16; GSM8K-200 + MMLU-Pro-280 vs Qwen3.6 IQ2_M 96.0 +-1.4 / 71.4 +-2.7 | kill if > 2 sigma under either or < 30 t/s. Pass => write an `hq1.sh whittle` arm (same file pattern as `stock`) and queue it after hq1_stock. A weak row may be the Q2_K quant, say so, do not call the MODEL bad |
| cpu1bench2 | 6-thread speedup with FIXED expert ids vs re-drawn (3.5x) | jumps toward 5-6x => "CPU miss phase is DRAM bound" confirmed, leave notes as is. Stays ~3.5x => RETRACT the bandwidth reading and its consequences (notes 2026-09-21 cpu1bench entry, SPEC CPU1 row); CPU1 stays dead either way (task = ggml) |
| hq1_iq2m | final MATH-L5 / EvalPlus / AIME-15 with truncated + looping counts | record only. No verdict on the quant until hq1_stock is in |
| dense1 | pull `/ai/bench/imx/qwen38_27b_dense.imatrix`, run `bench/imx_heat.py` on the Mac | FLAT closes DM1's static split; CONCENTRATED = note it, a per-token probe would be new C++ (call Fable) |
| hq1_stock | same sets on stock weights, `bench/qual/paired.py` stock vs served on the shared items | stock also cuts ~40% of MATH-L5 chains => it is the 2.5-bit quantization (RAM1 / QX3 become the priority). Stock finishes => it is the fine-tune; serving the uncensored file on hard reasoning is then Andrei's call |
| qx3 | mean KLD: K2 0.2181 -> mix25 -> mix50 -> X4 | X4 not far below K2 => QX3 dead. mix25 >= 25% lower => GO for the runtime hot / cold split = new C++ (call Fable). Remember the measured speed price of bigger experts (cpu1bench entry) |
| lq1 | needle pct + vt mean per depth per KV type | kill q4_0 if worse than f16 / q8 by > 1 in 10 at any depth; q4norot vs q4 says whether attn-rot matters |
| lq2 | KROW 0 vs 1, same instrument | KROW within 1 in 10 at every depth => it becomes the code default at the next build |
| hq1_k2, bonsai1_* | paired vs the iq2m rows | record; K2-as-default and any model switch are Andrei's calls |

Do NOT: write or change CUDA / C++ kernels, flip defaults in code, re-derive dead verdicts (FA3, FA5, CPU1, P2, R1), or conclude
from text identity at long context (any float reordering moves logprobs by ~0.13 there). A failed job: read its log tail, fix the
SCRIPT if it is a script bug (check `set -u` vs preflight.sh, paths, PYTHONPATH for gguf-py), `queue.sh retry ID`, move on.

Call Fable back for: the runtime hot / cold expert split (if qx3 says GO); HAND1 (the ~0.3 ms per layer CPU<->GPU handoff gap:
analyze `/ai/bench/prof2_nsys.nsys-rep` memcpy / sync timeline first, then C++ in the expert-cache path); a per-token neuron probe
(if dense1 says CONCENTRATED); anything where a measurement contradicts a recorded verdict.

## Addendum, 2026-09-21 21:10 (state at the model switch)

- **whittle1 is mid-run**: both speed rows are in (decode 29.5 t/s cache 0, 34.8 t/s cache 16, no MTP), about 340 quality items done
  (gsm8k finished, mmlu_pro in progress). A background waiter in the Claude session is armed on it. When it ends, apply the whittle1
  rule above, record it, push. The speed rows are CLEAN: see the notes entry "foreign CPU beside whittle1".
- **Foreign CPU.** Andrei's GitHub Actions runner (`ci-runner@1/2`) ran vite builds beside whittle1's quality section. He keeps the
  runners off himself during the queue; do NOT touch those services. `mon.py` now prints `FOREIGN CPU [label]` above a telemetry line
  when non-benchmark processes average over half a core in that window, and sets `foreign_cpu_flag` in `runs/<label>.mon.json`.
  **Rule:** a speed row with that flag is void. Rerun that job (`queue.sh retry`) after confirming the box is quiet; never record
  its t/s. Accuracy-only rows (qual, KLD, retrieval) stay valid.
- **Queue order after whittle1:** cpu1bench2, hq1_iq2m (resumes, ~11 AIME items left), dl_stock, dense1, hq1_stock, qx3, lq1, lq2,
  hq1_k2, bonsai1_easy, bonsai1_hard. The `rc=interrupted` on hq1_iq2m is from the deliberate queue stop, not a failure.
- **Unverified until the next reboot:** queue autostart (the systemd ordering-cycle fix). After any reboot check
  `systemctl is-active ai-queue` and the governor before trusting a number.

## Addendum, 2026-09-22 02:00 (top-model review of the night's watch)

- **cpu1bench2 was read BACKWARDS by the watcher and has been corrected.** The rule in the table above said "jumps toward 5-6x =>
  confirmed". It measured 4.98x and the watcher retracted anyway, with a new theory (cold working set, hugepages, locality). Fixed
  ids = bytes served from the L3 = no DRAM traffic; that the gain appears is the confirmation. **Apply the written rule literally.
  If a result seems to call for a new theory, that is the "call the top model back" trigger, not a license to rewrite SPEC.**
  There is NO hugepage / locality lever (THP is `always`; an expert is a ~1.1 MB sequential stream) and the bytes-proportional
  speed price of fatter experts STANDS.
- **dense1 = FLAT** (median top-20% share 0.52). DM1 is closed; no per-token probe. `imx_heat.py` had a GGUF offset bug (fixed,
  tested against the real `gguf` library); the first MODERATE printout was garbage.
- **hq1_iq2m final** recorded (57.5 / 85.4 / 20.0). `hq1_stock` is running; its rule in the table is unchanged.
- Disk: MOVE + symlink large models and build data to `/mnt/md0`, never ask. Moving is not deleting: nothing gets `rm`'d outright
  without Andrei. Do the copies between jobs (`systemctl stop ai-queue` at a job boundary, copy, start) — last night they ran
  beside `hq1_iq2m`, which only survived because it is an accuracy job.
