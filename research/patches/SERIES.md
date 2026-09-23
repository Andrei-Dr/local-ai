# llama.cpp patch series — the single source of truth (updated 2026-09-23 06:45)

One linear chain on mainline, regenerated with `git format-patch 0af8ea3^..tu116-kernels` from the local clone
`~/dev/llama.cpp-mainline` (branches: moe-cache -> moe-cache-det -> prefill-mux -> tu116-kernels). Apply in order; each patch
applies on the one before. Older docs (notes.md, SPEC.md, opt-hunt) use the pre-09-23 numbering — the "was" column maps it.
The box keeps an operational copy of 0015-0019 in `/ai/bench/patches/tu116/` (tu1 / tu2 apply them to the ov tree).

Status: **SERVED** = in the served build (/ai/src/llama.cpp-mainline) · **TEST** = only in the test tree (/ai/src/llama.cpp-ov)
· gate = what still decides promotion. Switches: env vars unless noted; "default" = behavior with the variable unset.

| # | was | patch | status | switch / default | evidence | gate |
|---|---|---|---|---|---|---|
| 0001 | 0001 | GPU-resident expert cache for host-offloaded MoE weights | SERVED | `--moe-expert-cache N` | the base of everything | — |
| 0002 | 0002 | warm the expert cache from the prompt's routing | SERVED | `--moe-expert-cache-warm` | | — |
| 0003 | 0003 | size n_rs_seq for short n-gram drafts | SERVED | | | — |
| 0004 | 0004 | warm only from single-sequence batches (fix) | SERVED | | | — |
| 0005 | 0005 | cache-aware routing | SERVED | `--moe-expert-cache-bias`, off | lossy, off by default | — |
| 0006 | 0006 | cache-aware routing scales only plain probabilities (fix) | SERVED | | | — |
| 0007 | 0010 | LLAMA_MOE_CACHE_SYNC=1 deterministic cache publish | SERVED | env, off | identity-test tool | — |
| 0008 | 0007 | shared expert inside the cache split (overlaps host experts) | SERVED | always | HAND1: part of +5.4% decode, byte-identical | — |
| 0009 | 0008 | CUDA concat flat kernel (delta-net conv state) | SERVED | always | same | — |
| 0010 | 0009 | sched enqueues host->device split copies async | SERVED | always | same | — |
| 0011 | — | **prefill mode** `--ubatch-prefill N`: cache slots released for a big prefill ubatch | TEST | CLI, off unless `-ubp` | pmux1/3: prefill 3.2x (2.2k tok), 3.5x (9.3k); decode -2.0% (noise); identity 3/4 (long: near-tie) | **pmux5** KLD ub2048 vs ub128 |
| 0012 | — | prefill mode trims CUDA pools before slots return (fix) | TEST | with 0011 | needed for resume | with 0011 |
| 0013 | — | cache suspend detaches slot tensors (fix) | TEST | with 0011 | needed for resume | with 0011 |
| 0014 | — | **dp4a MMQ** on tensor-core-less Turing | TEST (local `#define`) | CMake `-DGGML_CUDA_MMQ_NO_MMA=ON` | mmqdp1: long prefill 48.5 -> 372.4 t/s with 0011; decode flat; unit 929/929 + 1297/1297 | **pmux5** KLD dp4a arm |
| 0015 | tu116/0011 | FA never selects the MMA kernel | TEST | `GGML_CUDA_FA_NO_MMA=1`, off | tu1: prefill +29%, decode -3.9% -> use 0019 instead | — (superseded by 0019 for serving) |
| 0016 | tu116/0012 | MoE MMQ tile width per expert | TEST | `GGML_CUDA_MMQ_MOE_EXPERT_COLS=1`, off | tu1: ub128 prefill +8.8%; nothing at ubp 4096 | **pmux5** KLD; likely stays off once 0011 serves |
| 0017 | tu116/0013 | whole-tensor expert upload, no router readback (>= 8 tok/expert) | TEST | ON; `GGML_SCHED_MOE_READBACK=1` = old | tu1: exact (IDENTICAL x4), prefill neutral | ships with the set (groundwork for upload overlap) |
| 0018 | tu116/0014 | no host sync before stream-ordered split input copies | TEST | ON; `GGML_SCHED_SYNC_BEFORE_COPY=1` = old | tu1: exact (IDENTICAL x4), decode +3.4% on a contaminated run | **tu2** clean 3-round decode |
| 0019 | tu116/0015 | FA tile kernel only for batches of N+ tokens | TEST | `GGML_CUDA_FA_TILE_MIN_BATCH=32` for serving | expected: tu1's +29% prefill without the decode loss | **pmux5** KLD (fatile arm) + **tu2** |

## Promotion plan (when pmux5 + tu2 pass)
1. Clean full build of the served tree at tu116-kernels with `-DGGML_CUDA_MMQ_NO_MMA=ON` (drop the ov tree's local define).
2. Identity smoke (LLAMA_MOE_CACHE_SYNC=1) vs the ov build, one specbench round.
3. Serving config: `-ub 128 -b 2048 -ubp 2048` (or 4096 by fit), env `GGML_CUDA_FA_TILE_MIN_BATCH=32`.
4. Update this table (TEST -> SERVED), SPEC, notes.

## Not in the series (dead or parked; see research/opt-hunt-2026-09-23.md)
Router F32 (lossy), state-gather skip (MTP rollback), per-layer slot allocation, alternative cache policies (incl. pure/global
LFU, lfu_variants.py), -t 5, OMP pinning, experts on huge pages.
