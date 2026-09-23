# local-ai notes

Repo: **https://github.com/Andrei-Dr/local-ai** (private; `~/dev/local-ai`). Commit + push after every recorded result: notes, `bench/` (harness, ledger, logs mirrored from the box via `rsync root@i5.local:/ai/bench/{*.log,ledger.jsonl} bench/box/`), `research/`. The llama.cpp work itself lives on the box (`/ai/src/llama.cpp` branch `i5-tuning`, worktree `/ai/src/llama.cpp-moecache` branch `moe-cache`); it is exported here as `research/patches/box-series/` (`git format-patch 9a9394a..moe-cache`) — re-export after every box commit. A GitHub fork of llama.cpp would be PUBLIC (forks of public repos cannot be private), so that waits for an explicit go.

## i5 box (`ssh root@i5.local`)

- GPU: GTX 1650 SUPER 4 GB — Turing cc 7.5 but **TU116: no tensor cores**, no bf16/FP8. ~3.45 GB usable. PCIe 3.0 x16 (idles at gen1, gen3 under load). Power cap already at max (100 W).
- CPU: i5-10400F 6c/12t — `avx2 fma f16c bmi2`, **no AVX-VNNI / AVX-512**. RAM 2x8 GB DDR4-2667 dual channel (~38 GB/s theoretical, ~30 real). 2 GB swap.
- Disk: `/` is nvme ext4 (**~21 G free on 2026-09-21**, 111 G of it models). `/opt` and `/mnt/md0` are the same spinning md0 btrfs+zstd (531 G free): **never serve models or build from there**; it holds `models-cold/` (cold model files, symlinked back into `/ai/models`) and is the place for large intermediates. `/root`, `/tmp` are nvme.
- Layout: **`/ai/{src,models,bench,.venv}`** (nvme). tmux session `ai` (windows: build, dl, bench, mv).
- `/root/bin/docker-cleanup.py` (mirrored at `~/bin/docker-cleanup.py` on the Mac): `-a` now prunes anonymous 64-hex dangling volumes (named volumes always spared), `-y` skips the prompt (`-ay` for cron), refuses without a tty unless `-y`.

### Host settings changed (persisted 2026-09-19 via `/etc/systemd/system/ai-perf-tweaks.service`, oneshot, enabled; remove with `systemctl disable --now ai-perf-tweaks && rm` the unit)

| setting | was | now | why |
|---|---|---|---|
| cpufreq governor (all cores) | `powersave` | `performance` | CPU-resident layers are CPU-bound |
| `/sys/kernel/mm/transparent_hugepage/enabled` | `madvise` | `always` | fewer TLB misses on the ~5 GB of CPU-side weights (pairs with no-mmap) |
| power profile (power-profiles-daemon) | `balanced` | `performance` (2026-09-21) | the daemon re-applied `powersave` + EPP `balance_performance` after our sysfs write; the unit now runs `powerprofilesctl set performance` first and orders `After=power-profiles-daemon.service` (the old `After=multi-user.target` made an ordering cycle that silently dropped `ai-queue` at boot) |
| GPU persistence mode | off | on (`nvidia-smi -pm 1`, 2026-09-20) | no driver reload between jobs |

Box was hard-reset 2026-09-19 12:09 EEST (see RAM rule in the runbook); tmux session `ai` recreated (windows build, dl, bench).

## Runtime

**Default tree (decided 2026-09-20): mainline llama.cpp + our patches**, `/ai/src/llama.cpp-mainline` @ 2582f5c (b11056), `build75`
(arch 75) = every Qwen / Gemma benchmark and the decode server. Same source, two more builds: `/ai/src/llama.cpp-fa1` (worktree; the
CUDA attention patches from `bench/box/patches/`: FA1 GQA vec kernel default on, FA4 `GGML_CUDA_FA_VEC_KROW=1`, FA2 / NTC opt-in) and
`build6180` there (`61-virtual;80-virtual` + `FORCE_MMQ`: the Pascal-path build, prefill 2.4-3x, used as the PREFILL server). Status of
every piece: `SPEC.md` section 1. Not ollama (an `ollama.service` is active on the box, not ours, Andrei's call), not vllm.

**The PrismML fork is kept for ONE job: the ternary Bonsai model.** PQ2_0 / PTQ1_0 only load in PrismML-Eng/llama.cpp, branch `prism`
(not `prism-v6`); vLLM has no PQ2_0 support. The rest of this section and the Bonsai section below are the 2026-09-19 fork work, kept
as the record (the fork lacks `qwen4exp`, and mainline + patches measured +5-7% on the same configs, SPEC U2).

Local branch `i5-tuning` at `/ai/src/llama.cpp` = prism@9a9394a plus:
1. `fix:` BoldingBuilds `0001-qwen35-mtp-hadamard-inverse.patch` — without it `--spec-type draft-mtp` dies with "Hadamard-latent table 'token_embd.weight' is read without the inverse transform". Not upstream yet.
2. `perf:` **AVX2 path for `ggml_vec_dot_pq2_0_q8_0`**. Upstream has only a VNNI path and a scalar `#else`; this CPU has no VNNI so 60% of the model ran scalar. 38.5 -> 5.6 cycles/32 weights (6.9x). `test-quantize-fns` exit 0, `test-backend-ops MUL_MAT type_a=pq2_0` 45/45 CUDA-vs-CPU. Upstream candidate.

Build: `cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`.
The suggested `-DCMAKE_CUDA_ARCHITECTURES="61-virtual;80-virtual" -DGGML_CUDA_FORCE_MMQ=ON` for tensor-core-less Turing was tested on mainline (`arch1`, 2026-09-21): prefill 2.4-3x, decode -8% => two binaries.

## Model: Ternary-Bonsai-2-27B (dense, qwen35 hybrid, base Qwen3.8-27B, multimodal via mmproj)

- 64 blocks; full attention every 4th (16 layers, 24q/4kv heads, dim 256), other 48 are gated-delta-net. ~96 MiB/layer.
- Bytes: ffn 55% | attn 18% | `output.weight` 995 MiB Q6_K (13%) | `token_embd` 682 MiB Q4_K (CPU row lookup) | gdn 5%.
- KV = 64 KiB/token at f16 (8K ctx = 512 MiB). KV quant buys little here; davetha measured f16 > q8_0 on GDN hybrids (quantizing K is free, V is what costs).
- **Dense => no experts => an LRU/LFU expert cache has nothing to cache.** Every weight is read every token.
- Reasoning model: emits `reasoning_content` first; small `max_tokens` returns empty content.

**Housekeeping 2026-09-19 21:45:** deleted (user OK) the four superseded Bonsai files below (davetha PQ2_0, abliterated PTQ1_0, both stock prism-ml files; ~26 GB) and the stale `/opt/ai` copy (35 GB on the btrfs array). `/ai/models` now = Bonsai PQ2_0-MTP, Qwen3.6 IQ2_M + `mtp-Qwen3.6` Q4_0 head, Gemma4 IQ3_M / Q3_K_M / Q2_K_P + `mtp-gemma-4` drafter. 88 GB free on `/`.

Weights (historical table; only the first row is still on disk):

| file | size | role |
|---|---|---|
| BoldingBuilds `...Abliterated-PQ2_0-MTP.gguf` | 7.66 GB | **primary** — 2.13 bpw + grafted MTP head (blk.64) |
| BoldingBuilds `...Abliterated-PTQ1_0.gguf` | 5.95 GB | dead end here: no x86 kernel at all (generic scalar), and compute-bound unpack even on a 3090 |
| davetha `...Abliterated-PQ2_0.gguf` | 8.25 GB | superseded (2.45 bpw repack for AMD CDNA, no MTP); used for the benches below |
| prism-ml stock PQ2_0 / PTQ1_0 | 7.21 / 5.95 GB | only for the provenance diff; deletable |

Provenance (verified by byte-diff, `/ai/tdiff.py`): both BoldingBuilds files differ from stock in **exactly 98 tensors** (`ffn_down` 49, `ssm_out` 36, `attn_output` 13; blocks 15..63; 0.45–0.71% of bytes), 753 byte-identical, chat template identical, metadata identical except the MTP file adds `nextn_predict_layers` + 15 `blk.64.*` tensors. Matches their card and davetha's independent diff. Their refusal/HumanEval numbers are NOT re-verified.

## Measurements (davetha PQ2_0, `-ngl 24 -fa 1 -t 6 --load-mode none`)

- `-ngl 24` = 3,454 MiB VRAM = the ceiling. GPU util ~7% is expected: serial pipeline, GPU does its 24 layers in ~20 ms then waits ~300 ms for 40 CPU layers.
- decode: 0.71 tok/s (scalar kernel) -> **2.99 tok/s** (AVX2 kernel). CPU-only 0.47 -> 1.97. `-t 12` no better than `-t 6`.
- perf after fix: vec_dot 56%, libgomp spin 20%, `ggml_compute_forward_mul` 6%. CPU side pulls ~15 GB/s of ~30 => single-stream ceiling ~5 tok/s.
- **`GGML_OP_OFFLOAD_MIN_BATCH=2`** (default 32) streams CPU-resident weights over PCIe and verifies on the GPU. ms per forward pass:

| batch | default (CPU) | `=2` (streamed) |
|---|---|---|
| 1 | 334 | 334 |
| 2 | 676 | 519 |
| 4 | 1036 | 558 |
| 8 | 1743 | 649 |
| 16 | 3313 | 946 |
| 32 | 1354 | 1364 |

  => verifying 8 tokens costs 1.94x one token. Speculation pays here with LONG drafts (6–8), unlike the 3090 where `--spec-draft-n-max 2` won.

## Build (current, `/ai/src/llama.cpp/build`, 6m31s, 6 targets)

`-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_LTO=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61-virtual -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_CURL=OFF`

- `61-virtual` + `FORCE_MMQ` = the fork's own advice for Turing without tensor cores. Batch 2–8 unchanged (PCIe-bound); **batch 16: 946 -> 693 ms**. Tensor-core warning gone.
- LTO: no measurable effect (CPU-only 2.02 vs 1.97 tok/s).
- `prism` is NOT tracking mainline: merge base 2026-08-25, 431 commits behind (pr-scout). Fixes merged upstream since then are absent.

## Speculation sweep — BoldingBuilds PQ2_0-MTP, real chat completions, 200 tok, temp 0, thinking off, `-c 4096 -fa on -t 6 --load-mode none`, `GGML_OP_OFFLOAD_MIN_BATCH=2`

Harness: `/ai/bench/specbench.sh NGL LABEL [server args]` (+ `specclient.py`, `mon.py` 1 Hz telemetry: GPU util/power, PCIe rx, CPU busy, DRAM read via `uncore_imc`). A config only counts if it completes a generation — loading is not a fit test (n=8 @ ngl 22 loaded, then died in `cublasCreate`).

| config | ngl | decode tok/s (code / reason) | acceptance | GPU util | PCIe->GPU avg/max GB/s | CPU busy | DRAM read GB/s |
|---|---|---|---|---|---|---|---|
| no spec | 29 | 3.22 / 3.15 | — | 13% | 0.06 / 6.0 | 580% | 12.9 |
| MTP n=2 | 24 | 4.79 / 4.61 | 0.75 / 0.71 | 88% | 8.8 / 10.8 | 556% | 8.8 |
| **MTP n=3** | 24 | **4.86 / 4.89** | 0.60 | 89% | 8.0 / 10.6 | 550% | 8.5 |
| MTP n=4 | 22 | 4.71 / 4.75 | 0.50 | 88% | 7.4 / 10.7 | 560% | 8.2 |
| MTP n=15 | 18 | 3.12 / 3.08 | 0.18 | 83% | 4.7 / 9.4 | 597% | 6.4 |
| ngram-mod only | 28 | 3.09 / 3.10 | never drafted | 14% | — | 571% | 12.9 |
| ngram-mod + MTP n=2 | 24 | 4.78 / 4.61 | 0.75 / 0.71 | — | — | — | — |
| MTP n=2, `OMP_WAIT_POLICY=passive` | 24 | 4.50 / 4.34 | same | 84% | 8.2 / 10.6 | 128% | 8.5 |
| MTP n=2, `-ctk q8_0` | 24 | 4.72 / 4.42 | 0.74 / 0.72 | 88% | 9.2 / 10.6 | 547% | 8.7 |

Findings:
- No-spec: 6 threads pegged, DRAM at 1/3 of cap => CPU side is COMPUTE-bound (kernel headroom ~2.5x before DDR4 binds). GPU idle.
- Spec on: PCIe-bound (10.8 of ~13 GB/s peak), GPU 88%. Enabling MTP costs ~5 resident layers (187 MiB "rs cache" rollback snapshots + draft ctx + MTP head).
- The single recursive MTP head's acceptance collapses with depth (0.75 -> 0.60 -> 0.50 -> 0.18). Cheap long verification needs a real draft model or n-gram hits.
- The ~550% CPU in spec mode is OpenMP spin, not work; passive wait frees it but costs 6% — keep default.
- K-quant KV is pointless at 4K ctx (saves 64 MiB < 1 layer). Revisit at 32K (KV ~2 GB).
- ngram-mod never drafted. Known fork bug, not the workload: `--spec-type ngram-*` silently no-ops on Bonsai-2 (PrismML-Eng/llama.cpp#203, open, root cause unknown). n-gram speculation is blocked until that is fixed.
- **`61-virtual` build WILL crash on the IQ2/IQ3 MoEs** (ggml-org/llama.cpp#24064, present at `mmvq.cu:260` vs `:386`: host picks Turing MMVQ MUL_MAT_ID batch limits, sm_61 device code uses the Pascal table => `MUL_MAT_ID failed / invalid argument` on verify batches of 5–7). MoE work needs a `75-real` build.
- rs cache is linear in `--spec-draft-n-max` (`need_n_rs_seq()`), dtype hardcoded F32. `-np 1` always (acceptance -> 0 with `-np > 1`, ggml-org#27572). `-ub 128 -b 256` shrinks compute buffers => more resident layers (untested here).

### Placement: FFN-only on CPU (pr-scout, mainline #26622 equivalent via `-ot`)

`-ngl 99 -ot "blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_(gate|up|down)\.weight=CPU"` — attention + GDN + output head on GPU for all 64 layers, only FFN matmuls on CPU (blk.64 = MTP head stays on GPU).

| config | decode tok/s (code / reason) | acceptance | VRAM | GPU util | PCIe->GPU avg/max | CPU busy | DRAM read |
|---|---|---|---|---|---|---|---|
| FFN-on-CPU, no spec | 3.88 / 3.83 | — | 2910 MiB | 14% | — | 603% | 16.2 GB/s |
| FFN-on-CPU + MTP n=3 | OOM at load (compute pp buffers), also with `-ub 128 -b 256` | | | | | | |
| **FFN-on-CPU + MTP n=2, `-ub 128 -b 256`** | **5.24 / 5.05** | 0.74 / 0.69 | 3590 MiB | 97% | 10.0 / 11.1 GB/s | 131% | 9.0 GB/s |

**Best Bonsai-27B config (5.15 tok/s):**
```
GGML_OP_OFFLOAD_MIN_BATCH=2 /ai/src/llama.cpp/build/bin/llama-server \
  -m /ai/models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf -ngl 99 \
  -ot "blk\.([0-9]|[1-5][0-9]|6[0-3])\.ffn_(gate|up|down)\.weight=CPU" \
  -ub 128 -b 256 -fa on -c 4096 -t 6 --load-mode none --jinja --parallel 1 \
  --spec-type draft-mtp --spec-draft-n-max 2
```
Now PCIe-bound: GPU 97% "busy" at 46 W = waiting on ~3.7 GiB of FFN weights streamed per pass at ~10 of ~13 GB/s. Remaining levers: fewer bytes per pass (PTQ1_0 + grafted MTP head, ~18% less), more accepted tokens per pass, upload/compute overlap (mainline #21067 PoC, port needed).

Tried, no effect: mainline #27781 (MTP shared-KV misdetection) — applies cleanly, acceptance identical to the token (118/160, 115/166); this fork's grafted MTP head evidently does share the target's KV. Reverted.

ngram: `need_n_rs_seq()` (`common/common.h:387`) returns 0 for ngram types => every partial acceptance restores a full host checkpoint on this recurrent model. And `ngram-mod` defaults need a 24-token match / 48-token draft, which a 200-token non-repetitive answer never produces — so "never drafted" was most likely the workload, not (only) fork #203. Proper test = code-edit prompt + local patch adding ngram types to `need_n_rs_seq()` with draft capped ~8 (each rs snapshot ~62 MiB).

Progression: 0.71 (scalar kernel) -> 2.99 (AVX2 kernel) -> 3.2 (2.13 bpw file, ngl 29) -> 3.85 (FFN-only on CPU) -> **5.15 tok/s** (+ MTP n=2, streamed verification). 7.3x.

## MoE baselines (queue step 2) — `build75`, `/ai/bench/moe1.sh` -> `moe1.log`, same harness (200 tok, temp 0, `-c 4096 -fa on -t 6 --load-mode none -np 1`), `OFFLOAD=32` unless noted

### Qwen3.6-35B-A3B IQ2_M (40 MoE layers, 256 experts top-8, no MTP tensors in file)

| config | decode tok/s (code / reason) | VRAM | GPU util | power | PCIe->GPU avg | CPU busy | DRAM read |
|---|---|---|---|---|---|---|---|
| `-ngl 999 -ot exps=CPU` | 27.74 / 28.37 | 1412 MiB | 42% | 45 W | 0.56 GB/s | 525% | 8.1 GB/s |
| `--n-cpu-moe 36` | 27.52 / 28.42 | 2394 | 43% | 46 W | 0.31 | 529% | 7.4 |
| `--n-cpu-moe 34` | 27.62 / 25.95 | 2886 | 44% | 42 W | 0.28 | 565% | 6.7 |
| `--n-cpu-moe 32` | 29.33 / 28.54 | 3378 | 46% | 48 W | 0.27 | 534% | 6.6 |
| `--n-cpu-moe 30` | OOM (compute pp buffers) | | | | | | |
| `--n-cpu-moe 34 -ub 128 -b 256` | 29.21 / 29.69 | 2846 | 47% | 48 W | 0.30 | 527% | 7.0 |
| `--n-cpu-moe 32 -ub 128 -b 256` | 29.42 / 29.28 | 3338 | 47% | 48 W | 0.28 | 535% | 6.7 |
| `--n-cpu-moe 30 -ub 128 -b 256` | OOM | | | | | | |

**Finding: Qwen decode is nearly flat in expert placement** — 20% of expert layers fully GPU-resident = +3–5%. NOT because experts are free: the thread sweep below fits T = 18.5 ms fixed + ~20 ms CPU experts at `-t 6`; the 1650 runs the same IQ2_S experts (MMVQ `MUL_MAT_ID`) at only ~2x the 6-core CPU (~9.5 ms for all 40 layers, inferred from the placement delta), so residency buys half of what it moves. Levers: tokens per pass = MTP (needs a grafted head; `research/qwen-mtp-scout.md`), then the cache (+10–20% projected).

### Gemma4-26B-A4B IQ3_M (30 MoE layers, 128 experts top-8, + `mtp-gemma-4-26B-A4B-it.gguf` drafter)

| config | decode tok/s (code / reason) | acceptance | VRAM | GPU util | power | PCIe->GPU avg | CPU busy | DRAM read |
|---|---|---|---|---|---|---|---|---|
| `-ngl 999 -ot exps=CPU` | 16.24 / 16.51 | — | 2144 MiB | 33% | 32 W | 0.45 GB/s | 554% | 9.9 GB/s |
| `--n-cpu-moe 28` | 17.50 / 17.33 | — | 2834 | 36% | 33 W | 0.32 | 556% | 10.0 |
| `--n-cpu-moe 27` | 17.58 / 17.77 | — | 3176 | 36% | 33 W | 0.41 | 555% | 9.6 |
| `--n-cpu-moe 26` | 18.18 / 18.20 | — | 3520 | 38% | 34 W | 0.17 | 561% | 9.7 |
| `--n-cpu-moe 25` | OOM at load | | | | | | | |
| **`--n-cpu-moe 26 -ub 128 -b 256`** | **18.31 / 18.33** | — | 3438 | 38% | 33 W | 0.17 | 549% | 9.6 |
| `--n-cpu-moe 25 -ub 128 -b 256` | OOM at load | | | | | | | |
| exps=CPU + MTP n=1 | 14.86 / 18.39 | 0.79 / 0.82 | 2444 | 22% | 29 W | 0.39 | 599% | 8.8 |
| exps=CPU + MTP n=2 | 15.34 / 17.42 | 0.67 / 0.74 | 2444 | 21% | 29 W | 0.52 | 568% | 8.3 |
| exps=CPU + MTP n=3 | 14.29 / 16.31 | 0.55 / 0.69 | 2444 | 23% | 28 W | 0.55 | 549% | 7.8 |
| MTP n=1, `OFFLOAD=2` | 15.10 / 15.47 | 0.85 / 0.86 | 2448 | 89% | 45 W | 8.30 | 122% | 7.7 |
| MTP n=2, `OFFLOAD=2` | 15.20 / 16.24 | 0.66 / 0.76 | 2448 | 90% | 47 W | 7.65 | 111% | 7.8 |
| MTP n=3, `OFFLOAD=2` | 14.92 / 16.96 | 0.56 / 0.71 | 2448 | 91% | 49 W | 8.76 | 132% | 7.8 |

**Findings:** Gemma decode IS expert-bound and scales ~linearly with GPU-resident expert layers (4 of 30 layers = +11%) => CPU expert compute (IQ3_S gate_up + IQ4_NL down kernels; DRAM only 10 of 38 GB/s => compute-bound, not bandwidth-bound) is ~75% of the 61 ms/token. MTP is a wash both ways: CPU verify of k tokens computes ~k x top-8 experts (cancels the gain); streamed verify (`OFFLOAD=2`) is PCIe-bound at ~8.5 GB/s. Drafter loads and drafts fine on Turing with default p-min (fork default is already 0.00). => Gemma is the expert-cache candidate (cache hits move expert compute to the GPU AND make MTP verify cheap), and the Q2_K_P file (K/legacy-quant experts = faster AVX2 kernels) is the no-code alternative.

### Where MoE decode time goes (`/ai/bench/prof1.sh` -> `prof1.log`; llama-bench tg64, exps=CPU, build75)

| `-t` | 2 | 3 | 4 | 5 | 6 | 6, `GGML_CUDA_DISABLE_GRAPHS=1` |
|---|---|---|---|---|---|---|
| Qwen3.6 tok/s | 14.15 | 18.93 | 22.40 | 25.20 | 26.05 | 26.63 (no effect) |
| Gemma4 tok/s | — | 11.06 | 13.66 | 15.40 | 16.72 | 16.95 (no effect) |

- Fit T = a + b/t: **Qwen a = 18.5 ms fixed (GPU attention/GDN/router + handoffs), b/6 = ~20 ms CPU experts; Gemma a = 21.7 ms, b/6 = ~38 ms.** 5 -> 6 threads is already sub-linear (thread 0 also drives CUDA).
- perf (decode only): Qwen `ggml_vec_dot_iq2_s_q8_K` 49%, libgomp spin 35%; Gemma `ggml_vec_dot_iq3_s_q8_K` 49%, `iq4_nl_q8_0` 12%, `q5_0_q8_0` 2%, spin 27%. libcuda ~3%. DRAM 8–10 of 38 GB/s => **CPU expert matvecs are compute-bound on the i-quant AVX2 kernels**, not bandwidth-bound. Consequence: CPU verify cost is per (token, expert) pair, so MTP verify of k tokens = k x expert compute no matter how much routing overlaps => MTP only pays once expert compute moves to the GPU.

### Router traces + cache simulation (`bench/traces/*.bin`, 8000 tokens each, code + prose corpora; `research/scripts/moe-cache-sim/sim.py --skip 64`)

Traces are sane: mean adjacent-token expert reuse Qwen 0.38–0.40 (chance 0.031), Gemma 0.40–0.42 (chance 0.062); layer 0 is the outlier (Qwen 0.06–0.10). Hit % (code / prose), uploads in MiB per token:

| model | slots/layer | cache MiB | LRU | decayed LFU | gate-LRU w16 a3 | gated upload MiB/tok (ungated) |
|---|---|---|---|---|---|---|
| Qwen3.6 | 16 | 653 | 48.7 / 50.7 | 45.4 / 37.2 | 51.1 / 48.4 | 25–29 (161–168) |
| Qwen3.6 | **32** | 1306 (fits ~1900 spare) | 65.2 / 65.8 | 62.9 / 57.4 | 66.5 / 63.3 | 13–17 (112–114) |
| Qwen3.6 | 48 | 1958 (borderline) | 74.4 / 74.5 | 72.4 / 70.7 | 74.2 / 73.0 | 9–11 |
| Gemma4 | 12 | 1022 | 46.9 / 46.1 | 48.2 / 43.9 | 55.0 / 52.0 | 71 (353–368) |
| Gemma4 | **15** | 1278 (fits ~1300 spare) | 57.8 / 54.6 | 57.9 / 51.0 | **63.6 / 58.8** | 54–57 (288–310) |
| Gemma4 | 20 | 1704 (no) | 69.5 / 64.9 | 70.3 / 61.2 | 74.0 / 67.1 | 33–42 |

- The "window 16, admit 3" gate matches or beats ungated LRU on hit rate at small caches AND cuts upload traffic 5–8x (pjsgsy's finding reproduces on our traces). LFU is never better by a useful margin. => policy = gate-LRU.
- Projection from the timing fits (INFERRED, to be measured): Gemma 15 slots, 61% hit: 21.7 + 0.39x38 + GPU share ~3 = ~40 ms => **~25 tok/s** (vs 16.4 exps=CPU, 18.3 best static placement). Qwen 32 slots, 65% hit: 18.5 + 0.35x20 + 0.65x9.5 = ~32 ms => ~31–33 tok/s (vs 29.4). MTP stacks on top once hits are on the GPU.
- **DECISION: GO on the expert-cache port, Gemma4 first** (bigger win; needs the fused gate_up path + the `down_exps_s` MUL-wrapper trap), Qwen second. Base = mainline #27861, carried out-of-tree on `i5-tuning`. Gate = it must beat the best static placement (18.3 Gemma / 29.4 Qwen); the 1080 Ti regression in #24524 was a different design (GPU dispatch from inside the CPU kernel).

### Expert cache P1 — mainline #27861 unmodified (ungated LRU), Qwen3.6 IQ2_M, `-ot exps=CPU`, worktree build (`/ai/bench/moe2.sh` -> `moe2.log`)

| config | decode tok/s (code / reason) | VRAM | GPU util avg/max | power avg/max | PCIe->GPU avg/max GB/s | CPU busy | DRAM read avg/max GB/s |
|---|---|---|---|---|---|---|---|
| cache off (same build) | 27.91 / 28.09 | 1396 MiB | 43% / 97% | 44 / 48 W | 0.80 / 5.08 | 517% | 8.2 / 9.3 |
| `--moe-expert-cache 16` | 28.16 / 28.29 | 2056 | 62% / 100% | 51 / 61 W | 2.05 / 4.22 | 534% | 7.3 / 8.5 |
| `--moe-expert-cache 32` | 30.30 / 29.62 | 2676 | 63% / 80% | 55 / 63 W | 1.80 / 4.28 | 534% | 6.4 / 7.5 |
| **`--moe-expert-cache 48`** | **33.38 / 33.06** | 3296 | 67% / 100% | 57 / 67 W | 1.96 / 3.70 | 519% | 5.5 / 7.0 |
| 32 slots, `--moe-expert-cache-inserts 1` | 30.81 / 30.92 | 2676 | 61% / 89% | 54 / 65 W | 0.85 / 2.13 | 531% | 5.7 / 7.5 |

- **Gate passed on Qwen:** 48 slots = +19% over cache-off and +13% over the best static placement (29.4 tok/s at 3338 MiB) for the same VRAM. Output text matches the cache-off run on both prompts.
- Ungated admission pushes ~2 GB/s over PCIe continuously (16 slots: all churn, zero gain); halving inserts halves PCIe traffic (0.85 GB/s) and is FASTER at equal slots => upload churn costs decode time => the admission gate (P2) is the right lever. DRAM read falls with hit rate (8.2 -> 5.5 GB/s) as expected.

### Expert cache P2 — commit `0e4103f` on `moe-cache` (fused gate_up + GELU + per-expert scales, gated admission w16/a3 with free slots filled ungated, in-flight de-dupe, sync before table publish, CPU pinning, stats). `-ot exps=CPU`, OFFLOAD=32. Full rows: `bench/LEDGER.md` (`p2_*` = before the free-slot fix, `p2b_*` = final)

| model | config | decode tok/s (code / reason) | VRAM | GPU util | power avg/max | PCIe->GPU avg/max | DRAM read | cache hit @256 steps | upload MiB/step |
|---|---|---|---|---|---|---|---|---|---|
| Gemma4 IQ3_M | cache off (same build) | 16.65 / 16.90 | 2128 | 34% | 31 / 44 W | 0.48 / 5.63 | 10.4 | — | — |
| Gemma4 | 8 slots | 19.45 / 18.92 | 2862 | 57% | 52 / 57 W | 2.43 / 8.28 | 9.9 | ? | ? |
| Gemma4 | 12 slots | 22.15 / 21.22 | 3186 | 58% | 57 / 66 W | 2.24 / 6.65 | 8.4 | ? | ? |
| Gemma4 | 15 slots | 23.85 / 22.78 | 3430 | 58% | 57 / 70 W | 1.34 / 4.86 | 7.4 | 57.2% | 56.6 |
| Gemma4 | 15 slots, admit 2 | 22.54 / 21.97 | 3430 | 62% | 57 / 73 W | 2.12 / 6.68 | 8.7 | ? | ? |
| Gemma4 | 15 slots, ungated (admit 1) | 21.76 / 21.21 | 3430 | 65% | 58 / 72 W | 3.20 / 7.11 | 9.3 | ? | ? |
| **Gemma4** | **16 slots, `-ub 128 -b 256`** | **24.34 / 22.98** | 3430 | 60% | 59 / 73 W | 1.37 / 4.54 | 7.4 | 59.5% | 54.5 |
| Qwen3.6 IQ2_M | cache off (same build) | 27.91 / 28.09 | 1396 | 43% | 44 / 48 W | 0.80 / 5.08 | 8.2 | — | — |
| Qwen3.6 | 32 slots (pre-fix gate) | 31.01 / 30.60 | 2676 | 61% | 56 / 68 W | 0.52 / 3.08 | 4.8 | ? | ? |
| **Qwen3.6** | **48 slots** | **34.00 / 33.79** | 3296 | 68% | 57 / 72 W | 0.50 / 2.96 | 4.4 | 63.6% | 19.1 |
| Qwen3.6 | 48 slots, admit 2 | 33.79 / 33.83 | 3296 | 67% | 58 / 72 W | 0.70 / 4.42 | 4.6 | 65.0% | 32.5 |
| Qwen3.6 | 50 slots, `-ub 128 -b 256` | 33.04 / 31.98 | 3334 | 65% | 59 / 71 W | 1.21 / 5.09 | 4.3 | 64.1% | 19.4 |

- **Gemma4: 16.7 -> 24.3 tok/s (+45%; +31% over the best static placement 18.3).** **Qwen3.6: 28.0 -> 34.0 (+22%; +16% over static 29.4).** Both pass the gate.
- The simulator is validated: Gemma 15 slots predicted 59–64% hit / 54–57 MiB per token, measured 57.2% (cold start included) / 56.6.
- Gate on Gemma, same slots: admit 3 23.9 > admit 2 22.5 > ungated 21.8 tok/s (PCIe 1.3 vs 2.1 vs 3.2 GB/s). Upload traffic costs decode time; keep admit 3.
- Gating everything hurt a cold cache (Qwen 48 slots: 31.8 vs 33.4 ungated) => free slots now fill on first miss and the gate only guards evictions (34.0).
- `-lv 4` is free (23.80 vs 23.85) and is what makes libllama INFO lines (cache stats, buffer sizes) reach the server log => harness default.
- Not yet done: P3 multi-token cache path (MTP verify), prefill warm-up of the cache, 600+ token steady-state runs, move the singleton into `llama_model`, the `ggml.h` callback signature cleanup.

### Expert cache P3 + P4 + #28549 port — commits `7068c64`, `e29d5bd`, `f94da5a` on `moe-cache` (compiled first try, 9m53s full rebuild). `/ai/bench/moe5.sh` -> `moe5.log`; all rows in `bench/LEDGER.md` (`p4_*`)

P4 = the device cache chain is built/expanded BEFORE the host chain and starts with `GGML_TENSOR_FLAG_SCHED_BARRIER`; ggml-backend-sched copies the host split's inputs ahead of enqueuing the barrier split, so CPU misses and GPU hits run concurrently. Output text identical to pre-P4 runs; quality re-validated by `qual3.sh`.

| model / config (`-ot exps=CPU`, OFFLOAD=32) | decode tok/s (code / reason) | accept | VRAM | GPU util | power avg/max | PCIe->GPU avg/max | CPU busy | DRAM read | hit % / MiB per step | RAM avail min |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma4 IQ3_M, 16 slots `-ub 128` (was 24.3 / 23.0) | **28.65 / 27.21** | — | 3436 | 67% | 62 / 77 W | 1.72 / 6.14 | 532% | 8.2 | 58.0 / 54.9 | 2830 MiB |
| Qwen3.6 IQ2_M, 48 slots (was 34.0 / 33.8) | **39.81 / 39.39** | — | 3304 | 76% | 60 / 74 W | 0.93 / 3.37 | 515% | 4.9 | 61.8 / 19.8 | 3678 |
| Gemma4 IQ3_M 16 slots, `-t 5` | 26.99 / 25.46 | — | 3436 | 64% | 62 / 78 W | 2.30 / 7.16 | 450% | 7.7 | 58.0 | 2874 |
| Qwen3.6 48 slots, `-t 5` | 39.27 / 38.28 | — | 3304 | 74% | 60 / 74 W | 1.00 / 5.14 | 425% | 4.9 | 61.8 | 3705 |
| Gemma4 IQ3_M, 12 slots + drafter, MTP n=1 | 29.38 / 27.95 | 0.82 / 0.83 | 3362 | 56% | 55 / 67 W | 2.37 / 5.90 | 538% | 10.1 | 48.1 / 61.9 | 2649 |
| same, MTP n=2 | 29.68 / 28.21 | 0.73 / 0.78 | 3362 | 54% | 54 / 68 W | 2.54 / 8.73 | 541% | 9.6 | 49.6 / 47.4 | 2632 |
| same, MTP n=3 | 26.82 / 26.86 | 0.65 / 0.69 | 3362 | 50% | 55 / 71 W | 2.63 / 8.86 | 523% | 9.1 | 48.1 / 39.3 | 2625 |
| Qwen3.6, 36 slots + `mtp-Qwen3.6` Q4_0 head, MTP n=1 | 38.32 / 37.40 | **0.94 / 0.88** | 3624 | 59% | 57 / 77 W | 0.88 / 3.03 | 566% | 5.0 | 57.6 / 12.1 | 3332 |
| same, MTP n=2 | **42.67** / CUDA OOM on prompt 2 | 0.88 | 3688 | 53% | | | | | 53.2 | |
| same, MTP n=3 | failed to create MTP context (VRAM) | | | | | | | | | |
| **Gemma4 Q3_K_M**, cache off | 20.73 / 22.74 | — | 2158 | 40% | 38 / 56 W | 0.55 / 6.20 | 575% | 13.6 | — | 2038 |
| **Gemma4 Q3_K_M, 15 slots `-ub 128`** | **33.66 / 32.36** | — | 3482 | 78% | 75 / 92 W | 2.53 / 6.22 | 517% | 11.1 | 55.1 / 64.8 | 2037 |
| **Gemma4 Q2_K_P**, cache off | 27.93 / 27.88 | — | 2034 | 43% | 51 / 58 W | 0.93 / 5.54 | 517% | 13.9 | — | 4492 |
| **Gemma4 Q2_K_P, 19 slots `-ub 128`** | **40.17 / 38.85** | — | 3332 | 75% | 73 / 95 W | 1.88 / 4.91 | 501% | 8.6 | 61.5 / 39.9 | 4457 |
| Gemma4 IQ3_M 16 slots, 800 generated tokens | 28.12 / 25.56 | — | 3436 | 70% | 71 / 86 W | 1.75 / 5.56 | 614% | 8.4 | 61–62.5 / 50–52 | 2886 |
| Qwen3.6 48 slots, 800 generated tokens | 40.35 / 39.18 | — | 3304 | 79% | 73 / 82 W | 0.66 / 3.03 | 574% | 5.3 | 65 / 13.5 | 3659 |

- **P4 overlap: Gemma +18%, Qwen +17%** (projected +10 / +15). Totals vs cache off: Gemma IQ3_M 16.7 -> 28.7 (+72%), Qwen 28.0 -> 39.8 (+42%). Holds over 800-token generations (hit rate 61–65%), so the 200-token harness is not flattering it.
- **K-quant experts are the biggest single Gemma lever:** Q3_K_M (gate_up Q3_K) is +24% over IQ3_M with the cache off and **33.7 tok/s** with it (2.0x the morning baseline) at about the same bpw; Q2_K_P reaches **40.2**. DRAM read rises to 11–14 GB/s: the K-quant kernels finally pull bandwidth instead of burning cycles. Quality rows decide which file is the default (`qual3.sh`).
- P3 works: MTP verify batches are served from the cache. Gemma MTP n=1/2 = +3% over no-MTP even after giving 4 slots/layer to the drafter; n=3 loses. Qwen's standalone Q4_0 head gets 0.88–0.94 acceptance on the abliterated trunk (no graft needed) and 42.7 tok/s at n=2, but 36 slots + head does not fit (CUDA OOM at 3688 MiB) -> `moe6.sh` re-fits with 28–30 slots.
- `-t 5` loses on both (thread 0 is not the bottleneck). Keep `-t 6`.

### MTP on top of the cache, re-fitted to VRAM (`/ai/bench/moe6.sh` -> `moe6.log`, commit `f94da5a`; full telemetry in `bench/LEDGER.md`)

| model / config (`-ot exps=CPU -ub 128 -b 256`, OFFLOAD=32) | decode tok/s (code / reason) | accept | VRAM | cache hit @512 steps | upload MiB/step |
|---|---|---|---|---|---|
| **Qwen3.6 IQ2_M, 30 slots + `mtp-Qwen3.6` Q4_0 head, n=2** | **46.27 / 44.78** | 0.88 / 0.83 | 3454 | 53.4% | 12.4 |
| Qwen3.6, 28 slots + head, n=2 | 43.36 / 44.75 | 0.83 / 0.84 | 3378 | 51.7% | 13.0 |
| Qwen3.6, 28 slots + head, n=3 | 41.04 / 42.36 | 0.75 / 0.76 | 3440 | 50.9% | 12.1 |
| Gemma4 Q3_K_M, 11 slots + drafter, n=1 | 33.79 / 34.58 | 0.78 / 0.78 | 3382 | 45.2% | 71.2 |
| **Gemma4 Q3_K_M, 11 slots + drafter, n=2** | **37.17 / 36.60** | 0.69 / 0.74 | 3382 | 46.3% | 53.9 |
| Gemma4 Q2_K_P, 15 slots + drafter, n=1 | 48.16 / 43.58 | 0.81 / 0.74 | 3306 | 56.1% | 42.6 |
| **Gemma4 Q2_K_P, 15 slots + drafter, n=2** | **48.23 / 46.58** | 0.69 / 0.72 | 3306 | 56.9% | 35.3 |

- **Best configs (2026-09-19 ~21:40):** Qwen3.6 `-ngl 999 -ot exps=CPU --moe-expert-cache 30 -ub 128 -b 256 -md /ai/models/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2` = **46.3 tok/s** (28.0 this morning, +65%). Gemma4 Q3_K_M `--moe-expert-cache 11 ... -md /ai/models/mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp --spec-draft-n-max 2` = **37.2**; Gemma4 Q2_K_P with 15 slots = **48.2** (16.2 this morning). Binaries: `/ai/src/llama.cpp-moecache/build75/bin`. Which Gemma file is the default is a QUALITY decision (`qual3.sh` rows).
- n=2 is the sweet spot everywhere; n=3 loses (acceptance decay + bigger verify batches). Trading ~1/3 of the cache slots for the drafter is worth it on all three files.

**Mainline rebase (for Qwen3.8-Flash-Next, arch `qwen4exp`, absent from the PrismML fork):** `research/patches/mainline-moecache-e613ef2.diff` = the whole cache series squashed onto ggml-org master `e613ef2` (local clone `src/llama.cpp-mainline`, branch `moe-cache`, gitignored). Three trivial rejects + the fork-only Hadamard check removed; libllama compiles (CPU-only check on the Mac). Mainline already has #28549 and #28739. Flash-Next facts (config + HF listings): 48 layers, 512 experts x 4.9M params, 10 routed (2.4B routed active), MTP head; smallest GGUF 72.5 GB (unsloth UD-IQ1_S), uncensored IQ2_XXS 74.9 GB (orcarouter), REAP-256 62 GB => NVMe-streamed (Samsung 980, PCIe3 x4), expected 3-8 tok/s (INFERRED). Needs ~75 GB disk (70 free; the superseded GGUFs = ~27 GB) => **PARKED by the user (2026-09-19): "stick with Gemma and qwen36 here for now".** The mainline rebase stays in the repo for when it is wanted (also the base for any upstreaming).

### Next levers on the MoE path (ordered; each gets ledger rows + a quality row where the model file changes)

1. **P3 multi-token cache path** (`research/patches/moecache-p3.diff`, written, not applied): cache graph for 1–4 token batches (ids `cont` + flatten before the table `get_rows`; stays inside the Turing MMVQ `MUL_MAT_ID` window: IQ3_S 6, IQ2_S 7, default 8). Makes MTP verify hit the GPU-cached experts. Then measure: Gemma + `mtp-gemma-4` drafter n=1/2/3 with cache; Qwen + `-md mtp-Qwen3.6-35B-A3B-Q4_0.gguf` n=1/2/3 with cache. Cherry-pick mainline #28549 first (`research/patches/pr28549.diff`, 2 files: separate graph-result arenas for batches with/without outputs so MTP draft steps reuse graphs) and #28739 (`pr28739.diff`, 4-line OOB fix for small `GGML_OP_OFFLOAD_MIN_BATCH`).
2. **Faster CPU expert kernels by changing the quant, not the code** (perf: `ggml_vec_dot_iq3_s_q8_K` = 49% of Gemma decode): same repo, header-scanned with `ggufscan`: **Q3_K_M** 13.29 GB = gate_up **Q3_K** x30 + down Q5_0 x29/Q5_1 x1, experts 11.79 GB RAM (vs 10.92 now: tight, watch swap; non-expert 0.89 GB) — about the same bpw as IQ3_M but K-quant AVX2 kernels; **Q2_K_P** 10.70 GB = gate_up Q2_K + down Q4_0, experts 9.28 GB (speed-first, quality cost to be measured). Both downloading to `/ai/models` (tmux `ai:dl`, `/ai/bench/dl_gemma_k.log`, marker `DL_GK_DONE`). Per-expert size changes => re-pick slots/layer (VRAM budget ~1300 MiB).
3. **P4 overlap CPU misses with GPU hits** (INFERRED gain: the cache chain costs ~4 ms/token of GPU time on Gemma and ~4.6 ms on Qwen, i.e. up to +10% / +15%): both chains depend only on the router output, but ggml-backend-sched runs them back to back, because the device->host input copy of the CPU split synchronizes the whole CUDA stream. Needs scheduler surgery: force a split boundary before the cache chain, and hoist the NEXT (CPU) split's input copies ahead of enqueuing the cache-chain split, so the host blocks only on the router. Contained to `ggml-backend.cpp`, but it is a full CUDA-free rebuild of ggml-base.
4. Steady-state bench (600–800 generated tokens) + cache warm-up from the last prefill tokens: the 2x200-token harness includes the cold start, so it understates real chat throughput (hit rate 57–64% at step 256 and still rising).
5. Thread count with the cache on (`-t 5` vs `6`: thread 0 also drives CUDA and the CPU share is now smaller), `GGML_CUDA_REGISTER_HOST=1` (pins the host expert buffers => true async DMA uploads; present in the fork at `ggml-cuda.cu:4925`; 11 GB pinned on a 16 GB box needs care).
6. Turing-specific MMVQ->MMQ switch points (mainline #26079 only tuned Ada/Blackwell/Spark; `pr26079.diff`) for the 2–8 token verify batches of the NON-expert matmuls.

## Run ledger (every number, with the commit that produced it)

- **`/ai/bench/ledger.jsonl`** — appended automatically by `specbench.sh` / `qualbench.sh` (via `ledger.py`) for every run, completed or dead: timestamp, label, model file + bytes, build dir, git branch/commit/dirty, server args, `GGML_OP_OFFLOAD_MIN_BATCH`, host state (governor, THP, kernel, driver), VRAM, per-prompt wall/decode/prefill tok/s + draft acceptance, telemetry (GPU util, GPU mem util, power avg/max, temp, PCIe rx AND tx, CPU busy, DRAM read AND write), expert-cache stats (slots, admit, hit rate, uploads, MiB/step, evictions), quality scores for `kind=quality`.
- **`bench/ledger_backfill.jsonl`** — 73 records for everything measured before the ledger existed, rebuilt by `bench/ledger_backfill.py` from the surviving server logs (per-task timings + acceptance), the raw 1 Hz telemetry files (`mon_resummarize.py` on the box -> `mon_summary.json`; this recovered wattage/temp/PCIe-tx for the early rows), the sweep logs (exact args, VRAM, CPU busy) and the llama-bench-only numbers from this file.
- **`bench/LEDGER.md`** — rendered table, one section per model: `rsync root@i5.local:/ai/bench/ledger.jsonl bench/box/ && ~/dev/.venv/bin/python bench/ledger2md.py`. Tables in this file are summaries; LEDGER.md is the complete record.
- Every patch is a commit on the box (`i5-tuning` / `moe-cache`), and a copy of each upstream diff / local patch lives in `research/patches/`.

## Quality tracking (mandatory next to every tok/s claim about a MODEL FILE)

Harness: `bench/qual/` (mirrored to `/ai/bench/qual/`), runner `bench/qualbench.sh LABEL [server args]` (env `MODEL`, `OFFLOAD`, `BUILD`, `QARGS`). `fetch.py` builds the fixed item sets from the HF datasets-server (deterministic, nested prefixes via `--limit`): **GSM8K first 50** (exact final number), **HumanEval every 4th task = 41** (pass@1, executed as uid nobody / no network / 20 s), **MMLU-Pro 70** (5 per category, 10-way multiple choice). Chat endpoint, temp 0, thinking off, max_tokens 768/1024/1024 (first pass used 400/512/350: half of MMLU-Pro was cut off and scored wrong; cut-off rows are re-run automatically) (`--think` = x8). Resumable: `results/LABEL.jsonl` keeps every raw answer. Prints one `QUALITY[label]` row with +-1 SE; at n=41–70 one SE is 5–7 points, so only gaps >10 points mean anything — rerun with a larger `fetch.py` selection before ranking close calls.

Scores belong to (model file, thinking mode). Placement and speculation flags do not change them beyond batch-variance noise, so each file is scored once on its fastest config; tok/s in a quality row is informational only. For quant variants of the SAME model also record wikitext-2 perplexity (not comparable across model families).

| model file | bpw | best decode tok/s | GSM8K | HumanEval | MMLU-Pro | truncated / empty | notes |
|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B IQ2_M, cache off | 2.69 | 28.0 (27.5 in-run) | 96.0 ±2.8 | 92.7 ±4.1 | 38.6 ±5.8 (*) | 42 / 0 | first pass, old caps |
| Qwen3.6-35B-A3B IQ2_M, **cache 48** | 2.69 | 34.0 (33.1 in-run, 28 min sustained) | 98.0 ±2.0 | 92.7 ±4.1 | 40.0 ±5.9 (*) | 41 / 0 | cache-on == cache-off within noise => cache is numerically sound |
| Qwen3.6-35B-A3B IQ2_M, **cache 48, cut-off rows re-run (new caps)** | 2.69 | 34.0 (34.3 in-run) | 100.0 | 95.1 ±3.4 | 62.9 ±5.8 | 19 / 0 | 19 answers still hit the 1024-token cap with thinking off (Qwen is wordier than Gemma: 8-11 cut-offs), so MMLU-Pro is still a floor for Qwen, not a score; Gemma 71.4 vs Qwen 62.9 is NOT a clean comparison until the cut-offs are gone (raise the cap to 2048 for Qwen or score finished answers only) |
| Gemma4-26B-A4B IQ3_M, **cache 16** | 3.9 | 24.3 (22.6 in-run, 41 min sustained) | 100.0 | 87.8 ±5.1 | 44.3 ±5.9 (*) | 38 / 0 | |
| Gemma4-26B-A4B IQ3_M, **cache 16, cut-off rows re-run (new caps)** | 3.9 | 24.3 (23.3 in-run) | 100.0 | 92.7 ±4.1 | 71.4 ±5.4 | 8 / 0 | all three Gemma quants now score exactly 50/70 on MMLU-Pro: check per-item overlap in `results/*.jsonl` before reading anything into it |
| Gemma4-26B-A4B IQ3_M, cache off | 3.9 | 16.7 | 100.0 (50/50) | 87.8 (36/41) | partial (*) | | killed by systemd-oomd at item 145/161 (15:06), resumed 21:05 |
| Bonsai-27B PQ2_0-MTP | 2.13 | 5.15 | dropped | | | | quality run killed 2026-09-19 23:33 at GSM8K item 7/50 (~5.6 tok/s => ~2.5-3 h, lowest-value row, blocking the decision jobs); Andrei: "fuck that bonsai model at this rate". Bonsai's real home is MLX on the Mac (parked). Speed 5.15 tok/s stands; no quality row. Reopen: only if we ever care about the dense PQ2_0 quality number |
| **Qwen3.8-35B-A3B-Distill IQ2_M, cache 48** | ~2.7 | 39.1 (no usable MTP head; 38.3 in-run) | 92.0 ±3.8 | 82.9 ±5.9 | 72.9 ±5.3 (6 cut) | 6 / 0 | S2 vs Qwen3.6: GSM8K -8, HumanEval -12 (both > 1 sigma => real regressions on our tasks), MMLU-Pro higher (72.9 vs Qwen 62.9-floor, but Qwen 62.9 has 19 cut-offs so not comparable until qmmlu lands). Distill is NOT a clear win: it loses on code/math, no standalone MTP head so it is slower too. Verdict leaning KEEP Qwen3.6 unless the clean Qwen MMLU-Pro is much worse than 72.9 |
| **Gemma4-26B-A4B Q3_K_M, cache 15** | ~3.4 | 33.7 (37.2 with drafter n=2; 31.5 in-run, 40 min sustained) | 100.0 | 90.2 ±4.6 | **71.4 ±5.4 (new caps)** | 11 / 0 | holds IQ3_M quality on GSM8K/HumanEval => Gemma default candidate; MMLU-Pro comparable only to other NEW-cap rows |
| **Gemma4 Q2_K_P, cache 19** | ~2.8 | 40.2 (48.2 with drafter n=2; 38.9 in-run, 33 min sustained) | 96.0 ±2.8 | 92.7 ±4.1 | 71.4 ±5.4 (new caps) | 11 / 0 | vs Q3_K_M: MMLU-Pro identical, HumanEval +2.5 (1 item, <1 sigma), GSM8K -4 (2 items, ~1.4 sigma at n=50) => no statistically resolvable quality loss at +24% in-run tok/s; Q2_K_P is the Gemma speed default unless a larger-n pass separates them |

(*) MMLU-Pro first pass is NOT usable as an absolute score: 34–39 of 70 answers hit the 350-token cap and every cut-off answer scored wrong (accuracy among finished answers: Qwen 27/31, Gemma 31/36). Caps raised to 768/1024/1024; `qual.py` re-runs cut-off rows automatically on the next pass (`/ai/bench/qual3.sh`, to be run with the new quants + Bonsai).

## Research (full reports in `research/`)

- `research/pr-scout-llamacpp-prs.md` — mainline + prism PRs/issues. Key: FFN-only-on-CPU placement (#26622), MTP shared-KV misdetection bug present in our checkout (#27781, `common/speculative.cpp:2129`), #24670 (GTX 1650 SUPER + Qwen3.6-35B-A3B: `draft-mtp` never drafts unless `--spec-draft-p-min 0.0`).
- `research/lit-scout-arxiv-survey.md` — arXiv survey re-ranked against our measurements. Expert cache design: async admission, compute misses on CPU, never stall on a fetch (FreeToken 2608.16157, WiSP 2606.21868); eviction policy barely matters, cache-aware routing (Cache-Prior 2412.00099) is the lever at small caches; do NOT quantize GDN state (DAMP 2608.27513).
- `research/moe-scout-abliterated-moe.md` — MoE shortlist. On the box: Gemma4-26B-A4B IQ3_M (12.39 GB) + `mtp-gemma-4-26B-A4B-it.gguf`, Qwen3.6-35B-A3B IQ2_M (11.66 GB). The QAT-MTP Gemma (16.8 GB) does not fit. Fork supports `gemma4`, `gemma4-assistant`, `qwen35moe`; AVX2 kernels exist for all their quant types.
- **Uno (ifm-ai/uno, arXiv 2609.04010, Apache-2.0), read first-hand 2026-09-19:** lossless self-speculation where the drafter is a token-gated LoRA (rank 128, all linears; `training/lora.py` masks the LoRA output to noise rows only) on the base model itself. Cycle = draft pass (full base forward over a block of L uniform-noise positions -> L-1 parallel draft tokens) + verify pass (standard rejection sampling on base weights), `nano_vllm_uno/engine/two_pass_decoding.py`. Trained with a pure **total-variation loss to the base model's own clean logits** (`--tv-gamma 1 --ce-alpha 0 --kl-beta 0`; acceptance = 1 - TV, so the loss IS the acceptance rate), 14.7B tokens, block-size curriculum 2->16, 2x8 GPUs. Claims up to 3x over AR and above EAGLE3/DFlash at every batch size on dense 8B. **Not a fit for the i5 box as-is (inferred):** the draft pass is a full-model forward over L tokens, which on CPU-expert MoE fans out to up to 8L experts per layer and blows past our 4-token cache window; our MTP head drafts for a fraction of one layer. Dense-only, nano-vLLM + FA2/FA3 (no Turing), no GGUF path. **Worth stealing:** the chunked TV loss (`training/losses.py`, `_ChunkedTotalVariation`) as the acceptance term for fine-tuning ONLY the Qwen3.6 MTP head / Gemma drafter on self-generated data on Dave's box (small trainable set, cheap) to push acceptance past 0.9 so n=3 pays. Relevant for dense models on Dave's GPUs / the Mac, not here.
- `research/ngram-codesign-scout.md` (Sonnet scout, 315 lines; all 32 arXiv IDs resolve with matching titles, checked via the arXiv API) — n-gram x MoE co-design. **Verified in source:** `--spec-type` is a comma list in mainline AND in our fork build (`common/arg.cpp`), and `common_speculative_draft()` tries the implementations in order, first non-empty draft wins => `--spec-type ngram-mod,draft-mtp` is an n-gram-first / MTP-fallback hybrid with zero code; ngram-mod defaults are `n_match 24, n_min 48, n_max 64` (long verbatim copies), tunable with `--spec-ngram-mod-n-{match,min,max}`; fork also lists `draft-dflash` / `draft-dspark`. **Queued:** `/ai/bench/ngram1.sh` (`ai:ngram1`, after `TRACE2_DONE`; `ngram1.log`, `NGRAM1_DONE`) = Qwen cache 30 + Gemma Q2_K_P cache 15, MTP n=2 baseline vs hybrid with n-gram drafts capped at 3 (inside the 4-token cache window) vs 16 (outside it), plus n-gram alone; `EDIT=1` adds a copy-heavy rename-a-function prompt to `specclient.py`, `GEN=300`. **Scout claims to treat as leads, not facts:** draft-driven expert prefetch lifting hits to 75–85% (the scout assumes draft router outputs; our MTP head yields TOKENS, so prefetch has to go token-n-gram -> expert table = exactly what `predict.py` measures on the trace2 data); ReMoE 2605.27081 gate-only fine-tune with locality regularizers (2–8 h on MI210 is the scout's estimate); LFRU over LRU (our sim said eviction policy barely matters); MTP self-distillation 2603.23911 (+5–7% acceptance; pair with Uno's TV loss). **Dead for us:** Engram 2601.07372 / SCONE 2502.01637 / PLE / Over-Tokenized 2501.16975 need pretraining, not retrofittable by distillation. **KD ETA (scout, stated assumptions, 2x MI210 ZeRO-3 offload 60–150 tok/s): 200M tokens = 15–39 days, 1B = 77–193 days** => full logit KD of a 35B-A3B is not a Dave's-box job; the head-only / gate-only fine-tunes are. **CORRECTION (same evening):** the scout's second pass found the source of the 'Whittle 3.3 h on 96 GB' figure and I read the cards first-hand: `logic65/Whittle-Qwen-3.8-35B-A3B` (released 2026-09-19, Apache-2.0) = 25.1B A3B body (8 of 180 experts) + **10B hashed bigram/trigram n-gram memory** transferred from Flash-Next's table, distilled from Qwen3.8-27B (forward-KL, top-128, 1,840 traces) with a dependence loss, joint training 3.3 h on one 96 GB Blackwell; `qwen4exp` (mainline only), GGUF Q3_K_M 16.73 GB, memory served via `-ot per_layer_token_embd=CPU`; self-reported GSM8K-50 41-44/50, research preview. That is the dense->A3B + n-gram-table model this project was imagining => candidate S4 in `SPEC.md`, behind the mainline build.
- **`SPEC.md` (2026-09-19) is now the PLAN** (work IDs, gates, kill criteria, sequencing, risk register); this file stays the STATE.
- `research/ngram-llamacpp-scout.md` — Qwen3.8 Flash-Next on Dave's rig via pi (tmux `qwen-scout`), llama.cpp n-gram speculation code read; pending, verify before use.

## RESUME HERE (state at 2026-09-19 ~22:40 EEST, written before a context compaction)

**UPDATE 23:35: Bonsai quality run dropped (see quality table); `qual3` queue entry skipped; queue now starts at `q38` (all remaining jobs are decision-relevant).**

**UPDATE 23:45 (Opus 4.8):**
- **S3 q38 (stock dense Qwen3.8-27B IQ3_M) RESOLVED, dead as expected:** `-ngl 24/20` OOM (dense, no expert trick), fell back to `-ngl 16` = **1.38 t/s** (200-tok). The FFN-on-CPU speed + quality configs both died: the `-ot 'blk\...ffn_(gate|up|down)\.weight=CPU'` regex did NOT match this model's tensor names, so all FFN stayed on GPU -> 4.36 GB alloc OOM => NO dense-Qwen quality row (not worth chasing; dense is off the i5 path on speed alone). If we ever bench a dense model again, fix the `-ot` regex first.
- **S2 distill (Qwen3.8-35B-A3B-Distill IQ2_M) speed:** cache off 27.8 t/s, **cache-48 39.1 t/s** (200-tok, code/reason) at 3304 MiB VRAM => A3B-class speed, in the same band as Qwen3.6. `distill_c30_mtp2` died: `distill.sh` used `-md <the full 12GB distill file> --spec-type draft-mtp`, reloading the whole model onto the 4 GB GPU (11.7 GB alloc OOM). Qwen3.6's MTP uses a SEPARATE ~200 MB standalone head (`mtp-Qwen3.6-35B-A3B-Q4_0.gguf`); this distill has no extracted head => **distill MTP speed unmeasured** (separate task: extract the in-file MTP layer to a standalone head GGUF). Quality run `distill_iq2m` (cache-48, no MTP) is what decides S2 vs Qwen3.6, running now.
- **qmmlu (clean Qwen3.6 MMLU-Pro cap 2048) still pending**, was leapfrogged by `distill` on a queue-insert race (distill was already the selected pending job when I inserted qmmlu); it is topmost-pending, runs next after distill.

**RESUME CHECKPOINT 01:35 EEST 2026-09-20 (before compaction) — READ THIS FIRST, then `SPEC.md`.**
- **Read order:** `SPEC.md` (plan/gates/status board) -> this checkpoint -> the dated UPDATE blocks below -> quality table.
- **Box durable queue** (`ai-queue.service`, enabled at boot, `Restart=no`; state `/ai/bench/queue.tsv`, `queue.sh list`): DONE = qual3(skipped), q38, distill, build_mainline, mainline_ab, ngram1, steady1, trace2. RUNNING/PENDING = `mmlu2k` (clean MMLU-Pro cap-2048 reruns: 3 Gemma + Qwen, reuses finished rows) -> `mainqual` (G2 confirm: mainline quality for Qwen c48 + Gemma Q3_K_M) -> `trace2b` (re-traces WITH `.tok` sidecars after the trace_tok bug fix). **To pause the box: `systemctl stop ai-queue`, NOTHING ELSE (kills the whole cgroup); never pkill. `start` to resume.** See memory [[reference-ai-queue]].
- **Monitors:** box row Monitor `b9h0hzfu1` (re-arm on expiry, 30-min). Qwen-scout death-aware waiter `b7m1ptnbn` (hunt 2). Re-arm the box Monitor with the same grep-across-logs command from history; never a success-only loop.
- **Qwen pi scout** (tmux `qwen-scout`, live/idle, `qwen38-flashnext-abl`): finished `research/ngram-llamacpp-scout.md` (hunt 1, folded in). NOW running hunt 2 -> `research/ngram-llamacpp-scout-2.md` (marker `HUNT2_DONE`): verify PR #26499 (`LLAMA_N_RS_SEQ` override), the minimal `n_rs_seq` change + memory cost for N2, and PR #28391 opt-out. Drive it ONLY via the tmux-harness skill (`~/.claude/skills/tmux-harness/scripts/pane.sh`): status before brief, file-pointer sends, `ask`/`expect` not sleep. Verify its claims vs source before acting.
- **DECISIONS RESOLVED TONIGHT (all [M]):** S2 = KEEP Qwen3.6 (distill regresses code -12/math -8, no MTP head, slower). S3 = dense 27B dead (1.4 t/s). G2 = mainline +5-7% faster than fork AND unblocks Whittle/qwen4exp => adopt mainline as default tree pending `mainqual` score confirm. G3/N1 = keep MTP n=2; n-gram hybrid ON, cap Qwen<=2 / Gemma long; +3.8% Qwen edit, ~free Gemma, neutral code/reason. N2 justified + Gemma control proves it's the GDN replay path. n=3 loses today => T1 justified. steady-state VALIDATES headline: Qwen 47.4/45.5, Gemma Q2_K_P 50.5/48.3 (higher than 200-tok).
- **STILL OPEN / NEXT:** (a) `mmlu2k`+`mainqual` land -> `ledger2md.py` regenerates SCOREBOARD/BUILDS/LEDGER, resolves knowledge axis + "best config (fastest that holds quality)" pick (floors math90/code85/knowledge65). (b) `trace2b` `.tok` -> run `research/scripts/moe-cache-sim/predict.py` on the traces = G4 prefetch go/no-go (Mac-side). (c) hunt 2 report -> build N2. (d) Phase-2 co-design (N3 rainbow table, P2 prefetch, P4c warm-up) NOT built yet — all gated on the above.
- **Tooling hardened tonight:** `ledger.py` captures full build provenance (commit, cmake CUDA-arch/type, gcc/nvcc, + saves diff blob if dirty; backfilled 50 old records, 0 unrecoverable). `ledger2md.py` emits LEDGER.md + SCOREBOARD.md (per-axis winners, MMLU-Pro-specific truncation flags, best-config picker) + BUILDS.md (rebuildable-artifact registry). `qual.py` records per-set truncation + `MMLU_CAP` env. All in `bench/`, mirrored `bench/box/`, pushed (HEAD `e2438b6`).
- **Last user msgs:** compact + "update your waypoint". Before that: fix the truncation flag (done), best-config picker (done, mix of 1sigma+floors), track everything recreatable (done), retry all 1024-cut MMLU at 2048 no-Bonsai (done=mmlu2k), use tmux-harness (doing).

**HANDOFF 04:50 EEST 2026-09-20 (Fable 5.1 -> Opus 4.8). i5 goes OFFLINE ~6 h (Andrei powers it off); Qwen codes briefs 14-16 meanwhile.**
- **Box:** queue at poweroff = `r1` (running; it restarts from scratch at boot, tries 2 of 3) -> `kq1` -> `mainqual` -> `h2qual` -> `orf1` -> `ctx1`. `ai-queue` resumes by itself at boot. Waiter `BOXW` (id in the chat) tolerates the outage (exits on all 6 ENDs, a failure line, or 16 h). When it fires: rsync ledger + `qual/results/` + `runs/ctx1_*.slot.json` + `ctx1.log`, `bench/jobreport.py JOB`, `ledger2md.py`, verdict rows in SPEC + dated UPDATE here. Nothing is a win inside 3-5% noise.
- **ctx1 is ready** (`47bb185` script + brief-13 `slotclient.py` on the box, guard passes): staggered ladder 16k->262k, 32k variants, TWO-PHASE rows (cache-0 big-ubatch prefill server -> slot -> decode-config restore). lat1 fit: T(ubatch) = 2.5 s + 6 ms/token => ~118 tok/s at ub 1024, ~138 at 2048 (projection, ctx1 measures it). Report tool = Qwen brief 14 (`bench/ctxreport.py --runs DIR --log ctx1.log`).
- **Qwen queue 14-16** (`research/qwen-queue/14-ctxreport.md`, `15-longctx-quality.md` = RULER-style retrieval at depth + `lq1.sh`, `16-slotlib-ctxproxy.md` = NVMe LRU slot store + restoring proxy). Waiter `QW2`. When it fires: review `git -C ~/dev/local-ai-qwen log main..qwen/work` commit by commit, run the suite, cherry-pick what is good, findings go BACK to Qwen as a numbered brief (that loop worked for 13). Do NOT queue `lq1` / `n2b` / `kq1graft` on the box, and do not deploy `ctxproxy` there: those are top-model calls after ctx1 numbers.
- **Still reserved for the top model / Andrei** (unchanged list in the 03:10 handoff below) + new: promote the in-model MTP head (needs M1-controlled repeat + one quality pass on the grafted file), the shipped long-context config (KV precision, slots vs KV, two-phase), R1, KQ1.

**UPDATE 04:20 (Fable 5.1 back) — Qwen queue merged, mtp1 verdict, ctx1 = staggered context ladder:**
- **EVERY decode number before `ctx1` was measured at `-c 4096` with <= ~1.9k tokens in the KV** (`specbench.sh` hardcodes `-c 4096`; prompts are a few hundred tokens, W4 "long" = 1.9k). Cache slots, MTP n=2, admission window: all known-good at 4k ONLY. Depth is unmeasured.
- **`ctx1` (queued last, `bench/box/ctx1.sh`) = STAGGERED depth ladder 16k -> 32k -> 64k -> 131k -> 262k**, shallow first so cheap answers land first, a dead rung never aborts. Per rung: save row (cold prefill + decode 64 + slot save) -> server restart -> restore row. 16k/32k/64k = F16 KV cache 24; 131k = q4_0 KV cache 24; **262k = q4_0 KV (~1.4 GiB) cache 12**. 32k variants price the alternatives: `-nkvo`, q8_0, q4_0, MTP n=2 (cache 16). Deep restore rows are gated on the 16k restore actually skipping the prompt. `TIMEOUT=21600` per request (262k prefill est. 1-3 h). 262144 is the NATIVE trained window (no rope stretch) => depth is a VRAM/time problem, not a quality one; q4 KV quality at depth is NOT measured by ctx1 (needs a long-context eval later).
- **Opus's first ctx1 draft was replaced** (it put F16 KV on the GPU at 131k/262k = 2.6/5.1 GiB, cannot fit; restore test sent an empty prompt; JSON spliced into a heredoc). Base is now Qwen's brief-10 script + `slotclient.py`; I added `-ot exps=CPU` (Qwen's row had none => whole model to GPU => OOM) and the ladder.
- **Review finding sent back to Qwen as brief 13:** `slotclient.py` names the slot `LABEL.slot`, ctx1 uses different labels for save/restore => every restore row would 404. Fix = env `SLOT` (+ env `TIMEOUT`, 3600 s was too short). `ctx1.sh` REFUSES to run (`CTX1_REFUSED`) until the box has that slotclient: after 13 lands, cherry-pick, `scp bench/box/slotclient.py root@i5.local:/ai/bench/`.
- **Qwen briefs 08-12 merged into `main`** (cherry-picked `bb1ce1e..8e9b0c1`, 40 tests green; the "notes.md -21" in the branch diff was only `main` having moved). Brief 12 = 28 audit findings in `research/qwen-queue/reports/12-report.md`: leads, verify each before touching SPEC. `n2b.sh` / `kq1graft.sh` exist, NOT queued yet.
- **mtp1 / V1c [M] (`bench/reports/mtp1.md`, n=1 per arm, mainline `498696c`):** in-model head WORKS. Equal 30 slots: ALL decode +0.03% (flat), VRAM 3444 -> 2924 MiB (**-520 MiB**). Freed VRAM spent: c42 +2.7% (inside noise), **c46 +11.0% ALL decode, hit 53.3 -> 65.0%, 3570 MiB**. temp-0 text diverges on 3 of 4 prompts vs `-md` at equal slots with acceptance unchanged = numeric tie-breaks from a different graph (same pattern as p4c1), not a lossless-speculation failure. => Promote to default AFTER a repeat under M1 noise control + one GSM8K/HumanEval pass on the grafted file; `kq1graft` stacks it on KQ1. The 520 MiB also pays for ~95k tokens of q4 KV: in-model head is the natural partner of long context.

**UPDATE 03:05 (Opus 4.8, now driving) — L1/lat1 measured [M], `/ai/bench/lat1.log`, Qwen c30+MTP2, 200-tok:**
- **The prefill wall is a `-ub` config problem, not a hardware limit — the night's most useful real-world find.** `-ub 512` (`lat_ub512_c24`): edit prefill 38.8 -> **75.3 t/s**, long prefill 38.9 -> **91.6 t/s** (2.4x), long wall 3.26 -> **6.79 t/s**; costs 6 cache slots (30->24) and ~3-5% decode. `-ub 256` (c28): long prefill 60.6, wall 4.83, costs 2 slots. CPU prefill (`OFFLOAD=99999`): long prefill 76.0, wall 5.87, decode unhurt (gpu_util 91->40% during prefill only) — free-ish since it needs no slots. **Recommend: default `-ub 256` (cheap) and pair with C1 slot-save; test `-ub 512` per-workload.** This is the lever behind Andrei's 262k ask.
- **PCIe rx peaks 10-11.3 GB/s (base), not the ~7 seen before** — pageable copies already near the PCIe-3 ceiling, so `GGML_CUDA_REGISTER_HOST=1` (`lat_pinned`) did nothing (48.99/45.13, within noise, mem_avail -65 MiB). Pinning is NOT a lever here. Drop it from the JIT-streaming rationale.
- **CUDA graphs are already off/inert:** `lat_nographs` == base (48.36->49.49 code, within 3-5% noise). No win, no loss; the mul_mat_id sync path disables them regardless.
- **NEGATIVE levers, do not use:** `OMP_WAIT_POLICY=PASSIVE` (decode 48->42, -13%), `GOMP_SPINCOUNT=1000` (-14%), `--prio 2` (-5%). The spinning OpenMP workers are load-bearing; passive waits starve the CUDA-driving thread. `-offload 8` == base (no effect). base power still ~50W/100 = latency-bound confirmed.
- Verdict: L1 done. Only lever = `-ub` for prefill (feeds C1). Kill pinning + graph + omp/prio ideas.

**HANDOFF 03:10 EEST 2026-09-20 (Fable 5.1 -> Opus 4.8 for queue watching). Read `SPEC.md` section 1.0 first: it is the one table of what is in flight.**

**ITEMS FABLE KEPT FOR THE TOP MODEL / ANDREI (do NOT hand these to Qwen; Opus: surface to Andrei rather than deciding solo):**
1. **C++ in `src/llama.cpp-mainline`** (branch `moe-cache`, base `0af8ea3`, HEAD `2582f5c`; Mac worktree builds `build-p4c`). Pending edits: set P4c default off + make its per-layer observation cheaper (every Nth layer) — P4c measured flat/negative (`bench/reports/p4c1.md`). Any cache/routing/graph change is top-model work: a torn-slot or race bug costs box-hours per iteration. Patches export to `research/patches/mainline-series/` + a bundle per commit to `research/patches/bundles/`; box gets the EXACT commit via `git fetch <bundle>` (hashes then match).
2. **R1 keep/kill (lossy).** `--moe-expert-cache-bias` changes outputs. Decide only from `r1` rows: needs >=+8% decode AND GSM8K/HumanEval within 1 sigma of B=0. If it fails the quality bar, kill it, do not keep "a little bias."
3. **KQ1 as the new Qwen default.** Only if quality within 1 sigma of IQ2_M AND MemAvailable stays >=1.5 GB AND swap quiet. It's the biggest projected lever (60-68 t/s). Needs Andrei's manual refusal spot-check before it ships as default (memory [[feedback-no-refusal-probes]]).
4. **Phase 1 decisions** (SPEC sec 9): G2 tree default (needs `mainqual`), S1 Gemma file (needs `h2qual`), best-config pick (today wrongly IQ3_M — an n=50 artifact `h2qual` fixes).
5. **Anything on Dave's box** (T0/T1/T2 — T2 router-locality is now the top training lever, sims put Belady headroom at 75-79% vs 60%), **deletions on the box** (S4 Whittle 16.7GB needs ~clearing the Q6_K_P source first), **anything public** (U3/U4 upstreaming). All need Andrei's explicit go.
6. **U3** (split the cumulative diff into a reviewable series) — Fable left this unqueued for Qwen deliberately: git surgery + per-commit build check in the llama.cpp tree. Consider only after the Qwen queue (briefs 08-12) proves clean.
- **Two one-shot waiters are armed, nothing else wakes you** (per-row Monitors were dropped on purpose: every event costs a full-context turn): `bzobvzu1c` = box jobs p4c1+lat1+mtp1 all ended / failure / box unreachable ~10 min (Andrei powers the i5 off ~02:30-03:00 for ~6 h; the queue resumes by itself at boot, then re-arm a waiter for the remaining jobs); `b4agrim85` = Qwen queue done (`QUEUE_DONE` in `~/dev/local-ai-qwen/research/qwen-queue/reports/STATUS.md`), Qwen died, or 4 h.
- **When box jobs end:** `rsync root@i5.local:/ai/bench/ledger.jsonl bench/box/` (+ `qual/results/`), `~/dev/.venv/bin/python bench/jobreport.py JOB` for p4c1 / lat1 / kq1 (mtp1, r1 get registry entries from Qwen brief 11; until then read the `tok |` rows in `/ai/bench/JOB.log`), `bench/ledger2md.py`, then write the verdict into the SPEC board row + a dated UPDATE here, commit, push. Identical repeats differ by 3-5%: call nothing a win inside that band. Box order: lat1 -> mtp1 -> r1 -> kq1 -> mainqual -> h2qual -> orf1. Queue nothing new until it drains to `kq1` (SPEC 1.0). Stop a job ONLY with `systemctl stop ai-queue`.
- **When the Qwen queue ends:** review `git -C ~/dev/local-ai-qwen log main..qwen/work` commit by commit (run `~/dev/.venv/bin/python -m unittest discover -s bench/tests` in the worktree), merge what is good into `main` (`git merge --ff-only` or cherry-pick), deploy new box scripts with scp, and only then consider queueing `n2b` / `ctx1` / `kq1graft`. Brief 12's audit findings are leads: verify each against the file before editing the SPEC.
- **Judgment calls that stay with the top model / Andrei:** C++ in `src/llama.cpp-mainline` (P4c default off + cheaper observation; any cache/routing change), the R1 keep/kill call (lossy), KQ1 becoming the Qwen default, Phase 1 decisions, anything on Dave's box, deletions on the box, anything public.

**UPDATE 02:55 (p4c1 verdicts + context/KV direction):**
- **p4c1 [M] (`bench/reports/p4c1.md`, ABAB, mainline `498696c`):** P4c warm 32 vs 0: Qwen decode +0.8% (noise 3.6%) = flat; Gemma decode -0.5%, **wall -4.1% LOSS** (the per-layer observation node slows prefill). => P4c does not pay in this benchmark; set the default to 0 (keep the flag for a targeted short-reply/topic-shift test) or make observation cheap first. N2-lite (n-gram cap 3, c29): **+4.5% ALL, edit +9.0%, code +6.7%**, n=1 per arm => repeat before trusting. V1 (`-otd exps=CPU`): c40 +2.4% / c38 +1.9% (inside noise), long prompt +10%. temp-0 text diverges between warm 0/32 on 3 of 4 prompts per model (expected: different cache contents => different GPU/CPU numeric mix; edit prompt identical). **Harness noise (3-5% between identical repeats) is now as large as the levers: needs 3+ repeats / longer GEN / discard first run.**
- **Context / KV (Andrei: 262k would be nice, 1M better):** Qwen3.6 KV = 20.5 KB/token (10 attention layers x 2 KV heads x 256, F16) + 62.8 MiB recurrent state. 32k = 0.64 GiB, 262k = 5.1 GiB F16 / 2.7 Q8 / 1.4 Q4, 1M = 20 GiB F16. Storage is solvable at 262k (KV in RAM + CPU attention, est. 8-15 tok/s at depth); PREFILL is the wall (262k at 39 tok/s = 1.9 h, attention alone ~45 min at n^2) => long context on this box = compute once, persist to NVMe (`--slot-save-path`, [V] in source: `llama_state_seq_save_file` via `/slots/N?action=save|restore`), optionally computed on Dave's rig with the same GGUF. 1M needs KV-on-NVMe sparse retrieval (Quest/InfLLM-class, lossy, not in llama.cpp) or more RAM. Scout report `research/kv-tiering-scout.md` (Sonnet, arXiv IDs resolved; its ranking assumed 4k-8k, where KV tiering is worth ~1 slot). [V] flags exist: `--slot-save-path`, `--cache-ram N` (MiB, can be small), `--ctx-checkpoints`, `--no-kv-offload`, `-ctk/-ctv` (our build has FA_ALL_QUANTS). TODO `ctx1` after `lat1` fixes the prefill config: decode tok/s at depth 32k/64k/128k (KV on GPU vs `-nkvo`, F16 vs Q8), slot save/restore time, cache + MTP behavior at depth; plus a TTFT test of slot-save / small `--cache-ram` on a repeated long prompt (client script = Qwen task 8).

**UPDATE 02:45 (R1 cache-aware routing built + queued; P4c Gemma rows):**
- **R1:** `src/llama.cpp-mainline` `7cbf08f` + `2582f5c` (guard: only plain non-negative softmax/sigmoid probs, no selection bias, not when ids are forced). Per cached layer a device F32 `sel_scale[n_expert]` (1 + B cached, 1 otherwise), written next to the table entry at the same sync point; `build_moe_ffn` multiplies `selection_probs` by it for 1-4 token batches only (prompts keep exact routing). One extra GPU op per layer. Bundle `moe-cache-2582f5c.bundle` on the box; job `r1.sh` rebuilds `build75` (also builds `llama-quantize`/`llama-imatrix` for `kq1`), sweeps B = 0 / 0.25 / 0.5 / 1 / 2 (Qwen c30 + MTP2, prompts code/reason/edit) and 0 / 0.5 / 1 (Gemma Q2_K_P), then GSM8K + HumanEval at B = 0.25 / 0.5 / 1 (`q36_iq2m_cache48_r1b*`). Watch: hit rate, MTP acceptance (the head drafts for the unbiased model), quality.
- **P4c Gemma Q2_K_P rows (a-pair):** code (cold) 49.12 -> **51.59 (+5.0%)**, reason 47.98 -> 48.55 (+1.2%), edit 56.38 -> 56.62 (flat). **Cost: Gemma prefill -12%** (35.4 -> 30.2, 41.0 -> 35.9, 64.5 -> 57.3 tok/s) from the per-layer CPU observation node; Qwen prefill is flat. If the b-pair confirms: observe only every Nth layer or only the last ubatch of a prompt.
- Queue: p4c1 (running) -> lat1 -> mtp1 -> r1 -> kq1 -> mainqual -> h2qual. Reports: `bench/jobreport.py p4c1|lat1|kq1` (add `mtp1`, `r1` entries to its registry).

**UPDATE 02:30 (V1c in-model MTP head + first P4c rows):**
- **V1c:** the standalone head GGUF = `output.weight` + `token_embd` + `blk.40.*` (incl. `nextn.*`), arch `qwen35moe`, `block_count 41`, `nextn_predict_layers 1` [M, gguf scan]. Mainline runs MTP against the TARGET model when the file carries the nextn block (`common/speculative.cpp`: no `-md` => `llama_init_from_model(model_tgt)` with `LLAMA_CONTEXT_TYPE_MTP`). `bench/box/gguf_graft_mtp.py MAIN HEAD OUT` copies MAIN + HEAD's `blk.40.*` raw, takes the `<arch>.*` hparams from HEAD, refuses on arch/vocab/width/name clashes (smoke test `bench/tests/graft_smoke.py`: byte-exact incl. a raw Q4_0 tensor). Job `mtp1.sh` (marker `MTP1_DONE`): graft -> `...-IQ2_M-MTP.gguf` (~12.1 GB), rows `mtp1_q36_md_c30` (reference) / `inmodel_c30` / `c42` / `c46`, textdiff. Unknowns the rows answer: does the cache pick up 41 layers, does the MTP graph run through the cache chain, acceptance parity. If it works, graft the KQ1 file the same way. `kq1.sh` disk threshold lowered to 45 GB (graft output takes 12).
- **P4c first A/B rows (Qwen c30 + MTP2, mainline `498696c`, clean build, base-tree diff empty):** cold prompt (code) 48.04 -> **49.84 (+3.7%)**, simulator said ~+3%; second prompt (reason, carried cache re-ranked) 45.54 -> 45.08 (-1.0%). ABAB still running. W4 long prompt: decode 36.6 tok/s at acceptance 0.61 (vs 0.81-0.98 on the short prompts), **wall 4.7 tok/s** because prefill is 39 tok/s over 1.9k tokens.
- Queue: p4c1 (running) -> lat1 -> mtp1 -> kq1 -> mainqual -> h2qual.

**UPDATE 02:25 (Andrei's telemetry read: "util high, watts low, never past ~11 GB/s" => L1):** ledger telemetry agrees. Decode: GPU util 67-78% avg (98% live) at 43-66 W of 100 W, P2, clocks at max => "util" only says a kernel or copy was in flight; the GPU is latency-bound (per-layer D2H ids + hidden state, H2D result, ~1200 small kernel launches per token from a CPU thread that shares cores with spinning OpenMP workers). H2D peaks 3-7 GB/s (pageable), PCIe 3.0 x16 can do ~12. **Prefill: 37-41 tok/s on Qwen for 42-, 60- AND 377-token prompts (Gemma 35-64)** = prompt processing is as slow as decoding; 1.9k tokens ~ 48 s TTFT. CUDA graphs: arch gate is < Volta so Turing is allowed, keyed per split, but a syncing `mul_mat_id` or batch > 1 disables them; whether they are live in our splits is unknown => measured by `lat_nographs`. Job `lat1.sh` (10 rows, ~30 min, marker `LAT1_DONE`) queued after `p4c1`: baseline, no CUDA graphs, `GGML_CUDA_REGISTER_HOST=1` (pinned), ub 256/512, CPU prefill / offload 8, `OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=1000`, `--prio 2`. Queue: trace2b -> p4c1 -> lat1 -> kq1 -> mainqual -> h2qual.

**UPDATE 02:35 ("why not the best of both: token table + LRU?" — tested three ways, all [M-sim] on the trace2b Qwen traces, 30 slots):**
- `swapvalue.py`: uses over the next 16 tokens (from t+2, when an upload lands): the cache's COLDEST entry 1.16 (code) / 0.96 (prose) vs the token table's best UNCACHED prediction 1.18 / 0.91 => a swap is break-even in hits and pays an upload; that is why `prefetch.py` (which IS LRU + table) comes out negative. The table mostly re-predicts what recency already holds.
- `hybrid.py`: one scored cache, score = decayed frequency + gamma x context affinity (sum of P(expert | token) over the last 8 SEEN tokens, no lookahead problem), margin admission: every gamma > 0 is <= gamma 0 (code 62.1 -> 61.7 -> 60.9 -> 58.3; prose 60.1 -> 60.3 -> 60.3 -> 58.1). Token identity adds nothing on top of usage history.
- `belady.py`: the ceiling is real though: Belady-optimal (perfect foresight, ~1 upload/token/layer) = **75.1% code / 78.8% prose vs 60% runtime**; an oracle's best uncached expert is worth 4.0-5.4 uses vs ~1 for the coldest entry. So ~15-19 points (~+20% tok/s) of POLICY headroom exist, but neither token identity nor usage history reaches it: on this model routing follows the hidden state (context), not the token. Reaching it needs a hidden-state predictor with a multi-token horizon (parking lot: SpecPrefetch/ProMoE-style, unproven at horizon > 1 layer) or making routing itself sticky (T2). Cheaper admission does NOT help (a2: 54.9-58.8, ungated: 47.6-55.3; the gate a3/i2 is the best history policy we have; decayed-LFU half-life 64 with margin admission = 62.1 / 60.1 at a third of the uploads, marginal).

**UPDATE 02:12 (V1 VRAM diet + early G4 read):**
- **V1 [M]:** VRAM of the best Qwen config (c30 + MTP n=2, 3714 MiB card): main model 976, **MTP head 727.7** (= `output.weight` 272.8 + the head layer's own 256 experts 3 x 144 Q4_0 + ~23 small; `token_embd` 272.8 sits in CUDA_Host), expert cache 1201.7 (40.06 MiB per slot), RS 188.4, KV 88, compute 140, rest = CUDA context. The head's experts run 8 per draft token on ONE layer => `-otd exps=CPU` frees 432 MiB for ~10 slots at ~+1 ms of draft per round. Rows added to `p4c1.sh` (`v1_q36_c40_mtp2_otd`, `v1_q36_c38_mtp2_otd`); the log must still say `40 layers` (the cache singleton binds to the first model that has host experts; the main context is created first). V1b = share the main `output.weight` with the head (needs loader code), +7 slots more.
- **G4 early read, Qwen code trace [M-sim]:** recall@30 from the token itself 76.0% vs LRU 56.7% (+19), from the token 2 back (what a prefetch that must land a round early can use) 68.2% (+11.5), bigram adds ~1 point => the RECALL gate passes. **But the policy simulation says no:** `prefetch.py` = runtime gated-LRU + token-keyed prefetch with the real 2-step visibility: hit 60.2% -> 58.4-59.7% (evicting LRU entries), -0.3 with a 16-token eviction guard; never positive. Reason: what the table predicts beyond the cache is single-use, and an upload (~1 MiB, ~0.2-0.4 ms) costs 3-5x a CPU miss pair (~0.07 ms = 31 ms / ~450 pairs per round). **The cache earns only through REUSE; prediction without reuse cannot pay on this box.** **Prose trace agrees and is worse: lookahead recall@30 49.1% vs LRU 64.9% (the recall gate itself fails), policy sim 60.1% -> 58.8%. G4 = FAIL, P2 is dead [D]** (reopen criteria in the SPEC board). Lesson for the spec: gate on a policy simulation with real latencies, never on recall.

**UPDATE 02:00 (perf-model fit from the ledger => new top lever KQ1; box going down ~02:30-03:00 for ~6 h):**
- Fit is in `SPEC.md` 4.1. Qwen MTP n=2 round = 58.4 ms = **5.4 draft + ~21 fixed + ~31 CPU expert matvecs**; miss cost scales with uncached (token, expert) PAIRS (compute-bound IQ2_S vec_dot), not distinct experts. => N3 demoted (ceiling = the 9% draft share, copy-heavy rounds only); "sub-linear m(k)" is not a lever while compute-bound; hit rate (~+11% per 10 points) and the per-pair cost are.
- **KQ1 queued (`bench/box/kq1.sh`, marker `KQ1_DONE`):** HauhauCS has no Qwen K-quant that fits (header scans: IQ2_M experts = IQ2_S gate/up x40 + down IQ2_S x37/IQ3_S x3 = 10.41 GB; Q2_K_P experts 13.67 GB). Job: download Q6_K_P (30.65 GB, disk check >= 50 GB, resumable), imatrix from the IQ2_M file on our code+prose corpus (200 x 512), `llama-quantize --allow-requantize --tensor-type ffn_gate_exps=q2_k ffn_up_exps=q2_k ffn_down_exps=q3_k ... IQ2_M` => `...-K2-expQ2K-downQ3K.gguf` (~12.9 GB, experts ~11.7), then no-cache pair (IQ2_M vs K2), c26+MTP2 vs IQ2_M c30+MTP2, c24+MTP3, full quality at cap 2048 (`q36_k2_cache26`). Gemma precedent: same move cut miss cost ~2.7x. Projection if 1.7-2.5x: 60-68 tok/s. Fallback recipe K1 (down Q2_K, ~10.7 GB) if RAM is too tight. The Q6_K_P source stays on disk (deleting needs Andrei's OK); disk after the job ~19 GB free => S4 Whittle download needs a cleanup decision first.
- **Queue order now:** `trace2b` (running) -> `p4c1` -> `kq1` -> `mainqual` -> `h2qual`. Interrupted jobs requeue at boot (`ai-queue` enabled); `kq1` download resumes (`curl -C -`), quality runs resume per item.
- Qwen agent task 4 delivered: `bench/abreport.py LEDGER BASE_RE TEST_RE [--md]` (per-prompt delta, repeat noise, WIN/LOSS/flat); use per-model regexes (mixing models in one arm inflates the noise column).

**UPDATE 01:55 (Phase 2 started; state for a cold resume):**
- **Built, not yet measured:** P4c prompt warm-up (`--moe-expert-cache-warm N`, default 32, 0 = off) = commit `684dd3f`; N2-lite (`need_n_rs_seq()` also covers n-gram ceilings <= 8) = `e3066f1`; both on `src/llama.cpp-mainline` branch `moe-cache` (base cache commit `0af8ea3`), compile on the Mac (`build-p4c`, CPU-only, no functional test possible there). Patches: `research/patches/mainline-series/`; the box gets the EXACT Mac commits through `research/patches/bundles/moe-cache-e3066f1.bundle` (box copy `/ai/bench/builds/`), so ledger hashes == Mac hashes from now on (the old box commit `cfb1ecdc7` has the same tree as `0af8ea3`; `p4c1.sh` prints that diff, must be empty).
- **Box queue:** `mmlu2k` (Qwen row running) -> `mainqual` -> `trace2b` -> **`p4c1`** (rebuilds mainline `build75` at `e3066f1`; ABAB warm 0 vs 32 on Qwen c30+MTP2 and Gemma Q2_K_P c15+drafter, prompts code/reason/edit/long, GEN 300; N2-lite pair `n2_q36_c30_ngmod2` vs `n2_q36_c29_ngmod3`; `textdiff.py` identity report; marker `P4C1_DONE`). Expected from the simulator: ~+3% on the first (cold) prompt, more on the later topic-shift prompts, N2-lite ~+1% on edit.
- **Adversarial review** of `684dd3f` + `e3066f1` running as Sonnet agent `p4c-reviewer`; findings go into the patch BEFORE `p4c1` starts (re-bundle + re-scp + update `WANT=` in `p4c1.sh` if the commits change).
- **MMLU-Pro at cap 2048 [M]:** Gemma IQ3_M **81.4** (0 cut), Q2_K_P **80.0** (1 cut), Q3_K_M **75.7** (4 cut: the same hard engineering items the models flail on, not loops); Qwen3.6 pending. Scoreboard now shows cut-offs as a band `(<=x)`; clean-enough = band <= 1 SE. **Picker today: `g4_iq3m_cache16` (29.7 tok/s) because Q2_K_P is excluded on GSM8K 96 vs 100 = 1.4 sigma at n=50** => that is an n problem, not a model verdict: H2 larger-n sets (GSM8K-200, MMLU-Pro-280, nested prefixes so finished answers are reused) are being coded by the Qwen agent (`/tmp/code3.md`, marker `CODE3_DONE`), then queue `h2qual` for the 4 candidates (~50 min each).
- **Qwen agent as coder works:** tasks 1-2 delivered green on the first pass (W4 workload + `textdiff.py` + off-box tests; scoreboard band logic + tests); protocol = brief in a file, literal spec, "do not commit", report file with a DONE marker, I review `git diff` and run the tests myself. One real bug found in review (truncation fallback double-counted rerun rows), fixed by me.

**UPDATE 01:40 (P4c go/no-go by simulation, `research/scripts/moe-cache-sim/warmup.py` on the 40k-token trace2 files, every 5th layer; [M-sim], upper bound because prompt and "generation" are adjacent spans of one document):** the runtime neither serves nor OBSERVES batches > 4 tokens, so prefill teaches the cache nothing. Decode hit rate, Qwen 30 slots, 200-token replies: cold 57.0% (first 32 tokens 38%) | carried from the previous request 60.5% same topic / 57.0% topic shift | **prefill-warmed 61.0-61.5% (first 32: 61-63%)**; re-ranking a carried cache by prompt frequency gives the same hit rate for 12 (same topic) / 27 (shift) uploads per layer. **50-token replies: carried 58.8% / 47.7% after a topic shift vs warmed 61.8% => +14 points exactly where agentic/tool-call traffic lives.** Prompt length is irrelevant from 64 tokens up. Gemma 15 slots: cold 52.7 -> warm 54.8, carried 54.3 / 53.5 -> 54.8 / 55.1 (smaller: 128 experts, flatter routing). Raising inserts/step 2 -> 8 recovers only half of the cold start (first 32: 38% -> 51%). Through the section-4 model that is ~+3% on every 200-token benchmark row, ~+10% on short replies after a topic shift, ~+0.5% on long same-topic generations. **Verdict: BUILD P4c** (prefill routing observed through a CPU custom op on the top-k ids for batches >= 32 tokens; at the next `step()` re-rank slots by prompt frequency, carried entries as filler, uploads async on the existing worker, never stall).

**UPDATE 01:30 (N2 verdict, hunt 2 = `research/ngram-llamacpp-scout-2.md`, claims checked in `src/llama.cpp-mainline`):** `n_rs_seq` costs **62.81 MiB of VRAM per unit** on Qwen3.6 [M, `server_ng_*` logs: `RS buffer size` 62.81 / 188.44 / 251.25 MiB at `n_rs_seq` 0 / 2 / 3; `llama-memory-recurrent.cpp:101` `n_rows = mem_size * (1 + n_rs_seq)`]. 16-token n-gram drafts would need +880 MiB = ~22 of 30 cache slots => **long-form N2 is dead on 4 GB**; only N2-lite (n-gram cap 3, `n_rs_seq` 3, +63 MiB) survives, ~+1% on edit, to be bundled into the next box build. Patch = ~6 lines in `need_n_rs_seq()` (mirror `common_speculative_n_max()`); PR #26499 is only a getenv with a wider blast radius (#28425). #28391 (ngram-mod default, still open): after any rebase pin `--spec-ngram-mod-n-max` or use `--no-spec-type ngram-mod`. MMLU-Pro cap 2048 first row: Gemma Q2_K_P **80.0 +-4.8** (2 cut-offs left), up from the 71.4 floor. Qwen pi scout switched to CODING (`/tmp/code1.md`: W4 workload, full-text capture + `bench/textdiff.py`, fake-server tests; marker `CODE1_DONE` in `research/code1-report.md`; review its `git diff`, it does not commit).

**UPDATE 01:20 (N1/N2, G3) — n-gram+MTP hybrid, 300-tok, code/reason/edit workloads:**
- **Qwen3.6 (GDN):** MTP n=2 baseline edit 47.2 t/s; +ngram-mod cap2 = **49.0 (+3.8% on edit)**, flat on code/reason (47.3/44.7 -> 46.4/44.1, within noise). ngmod cap2 n_min1 = 48.3 (no better than n_min2). **n-gram cap 16 CRATERS edit to 37.4 (-24%)** and cap16 n_match16 to 35.4: 16-token drafts exceed `n_rs_seq=2` -> GDN checkpoint+replay path eats the gain. n-gram ALONE (no MTP) = 37/35 on code/reason, worse than MTP baseline. MTP **n=3 does not pay** (c28: 44.9/43.1, acceptance 0.89/0.84 -> 0.80/0.77); n=3 c28+ngmod3 edit 49.5 is the best Qwen edit but code/reason still down.
- **Gemma Q2_K_P (NO recurrent state):** MTP n=2 edit **55.6 t/s** (fastest edit number we have); +ngmod cap16 edit **55.3 (flat), acceptance 0.99** — the SAME 16-token drafts that cratered Qwen leave Gemma untouched. **This is the causal proof: the Qwen 16-token regression is the GDN recurrent-replay path (`n_rs_seq`), NOT batch size.**
- **G3 verdict:** keep MTP n=2. Turn the n-gram hybrid ON, but cap it per model: Qwen n-gram <= 2 (until N2 lifts `n_rs_seq`), Gemma n-gram can go long (no cliff). Gain is edit/copy-heavy only (+3.8% Qwen, ~free Gemma); neutral on code/reason as predicted. **N2 is now justified by measurement** (lift `n_rs_seq` to cover n-gram drafts on GDN models). **T1 is justified**: n=3 only pays after acceptance rises, which is exactly the TV-loss head fine-tune.

**UPDATE 00:45 (U1/U2, G2): MAINLINE BUILDS AND IS FASTER.** `build_mainline` linked clean (llama-server + llama-bench at build75 on `src/llama.cpp-mainline`). Fork-vs-mainline A/B decode tok/s (200-tok, cache configs, no MTP): Qwen3.6 c48 fork 37.7/37.3 -> **main 39.6/39.5 (+5-6%)**; Gemma Q3_K_M c15 fork 31.4/32.3 -> **main 33.7/32.2 (+7%/-0.2%)**; VRAM within noise. Text identity at temp 0: Qwen both prompts IDENTICAL, Gemma reason IDENTICAL, **Gemma code DIVERGES after ~10 tokens** ("use a combination" vs "combine a") — same answer, a greedy tie-break flip from the 431 commits' numerics/kernel changes, NOT a cache bug (IQ2 Qwen is bit-identical). **G2 verdict (inferred): PASS on speed; strict bit-identity not met on Gemma-code but the cause is cross-tree numerics, so confirm with a QUALITY run on mainline (not a token diff) before making mainline the default tree.** Only diff heads were captured (specclient stores text_head, 80 chars); a full-text diff would need GEN-length capture. Next: could add a mainline quality row to the queue; mainline also unblocks S4 (Whittle, qwen4exp) and any upstreaming.

**HANDOFF 2026-09-19 23:35 EEST (model switch). Read in this order: `SPEC.md` (the plan: work IDs, gates, kill criteria, sequencing) -> this block -> the quality table above.**
- **Box:** durable queue, 8 jobs: `qual3` (Bonsai is what is left of it, still running from tmux `ai:moe6`) -> `q38` -> `distill` -> `build_mainline` -> `mainline_ab` -> `trace2` -> `ngram1` -> `steady1`. Check with `ssh root@i5.local /ai/bench/queue.sh list`; add work ONLY with `queue.sh add ID 'CMD'`. Service `ai-queue` is enabled at boot, `Restart=no`. Andrei may power the box off tonight; the interrupted job requeues on boot.
- **Watching:** re-arm the 30-min row Monitor when it expires (command is in this session's history: greps QUALITY / `tok |` / SERVER DIED / PREFLIGHT REFUSED / `*_DONE` across all the logs, remote side ends in `; true`). Never wait with a success-only loop.
- **When rows land:** sync `ledger.jsonl` + logs + `bench/qual/results/`, run `bench/ledger2md.py`, fill the quality table, then the Phase 1 decisions in `SPEC.md` section 9 (G2 tree default, S1/S2 model defaults, G3 n-gram default, G4 prefetch via `research/scripts/moe-cache-sim/predict.py` on the trace2 data). After that S4: Whittle-Qwen-3.8-35B-A3B Q3_K_M on the mainline build (budgets + kill criteria in SPEC 3.3).
- **Mac-side work, one at a time:** W4 workload in `specclient.py` -> U3 (split `research/patches/mainline-moecache-e613ef2.diff` into a reviewable series on `src/llama.cpp-mainline` branch `moe-cache`) -> N3a (two-region `ngram-mod` table so `reset()` cannot wipe a seeded store).
- **Open data questions:** all three Gemma quants score exactly 50/70 on MMLU-Pro (check per-item overlap in `bench/qual/results/*.jsonl`); Qwen still has 19 answers cut off at 1024 tokens, so its MMLU-Pro 62.9 is a floor.
- **Scouts:** cost rule = always pass `model: "sonnet"` (or use the free Qwen pi scout: tmux `qwen-scout`, idle, reusable; protocol in memory `reference_qwen_pi_scout`). All scout claims are leads until checked against source.
- **Needs Andrei's explicit go:** anything public (llama.cpp fork/PRs), deleting model files, anything on Dave's box (T0 spike is the first step there).

- **THE BOX QUEUE IS NOW A DURABLE FILE (2026-09-19 23:22), this supersedes the tmux-waiter description below.** State: `/ai/bench/queue.tsv` (id, status, tries, started, ended, rc, cmd); runner: `/ai/bench/queue.sh` (mirrored in `bench/box/`) under `ai-queue.service`; log: `/ai/bench/queue.log`. One job at a time; a job starts only after two idle checks 30 s apart: no llama-server / quality client / build / download / foreign `/ai/bench/*.sh` process AND nothing on the GPU (`nvidia-smi` lists no compute process, VRAM in use < 200 MiB; a failing `nvidia-smi` counts as busy). Done = rc 0 AND the script's `*_DONE` marker in its log. Queue order: `qual3` (no-op if `QUAL3_DONE` is already in `qual3.log`, otherwise resumes; `qual.py` skips finished rows) -> `q38` -> `distill` -> `build_mainline` -> `mainline_ab` (only if the build marker exists) -> `trace2` -> `ngram1`.
  - **Enabled at boot since 2026-09-19 23:30 (Andrei ran `systemctl enable`; the unit now has `[Install] WantedBy=multi-user.target`), so after a poweroff the queue resumes by itself once the box is up; it still never auto-restarts after a crash (`Restart=no`). Manual control: `systemctl start|stop ai-queue`; to keep it from running at the next boot: `systemctl disable ai-queue`.** A job that was running when the box went down is set back to pending on that start (up to 3 starts per job), so a poweroff mid-Bonsai costs only the unfinished rows. Governor/THP and the swapfile come back on their own (`ai-perf-tweaks.service` is enabled, swap is in fstab).
  - Commands: `/ai/bench/queue.sh list` | `add ID 'CMD'` | `retry ID` | `skip ID`; `systemctl status ai-queue`; `journalctl -u ai-queue`. Stop before a manual experiment with `systemctl stop ai-queue` (kills the running job; it requeues on the next start). It runs in system.slice, so systemd-oomd's user-slice pressure kill (the 15:06 incident) no longer takes the queue down with a job.
  - The `qual3` run that was already in flight in tmux `ai:moe6` at migration time keeps running there; the queue waits behind it via the idle check.
- **Box queue, all unattended, each step preflight-guarded (historical, pre-migration):** tmux `ai:moe6` = `qual3.sh` -> `/ai/bench/qual3.log` (`QUAL3_DONE`): Q3_K_M DONE (row above), Q2_K_P running, then cut-off re-runs for `g4_iq3m_cache16` + `q36_iq2m_cache48`, then Bonsai (~3 h). -> `ai:q38` = `q38.sh` -> `q38.log` (`Q38_DONE`): stock `Qwen3.8-27B-i1-IQ3_M` (downloaded, sha256 OK) speed + quality vs Bonsai. -> `ai:idlequeue` (`/ai/bench/idlequeue.done`): `distill.sh` (`Qwen3.8-35B-A3B-Distill-IQ2_M`, downloaded; `distill.log`, `DISTILL_DONE`) -> `build_mainline.sh` (`BUILD_MAINLINE_DONE|FAILED`) -> `mainline_ab.sh` (`MAINLINE_AB_DONE`). -> `ai:trace2` = `trace2.sh` (`TRACE2_DONE`): 40k-token traces with token ids for `predict.py`.
- **Wait on any of it ONLY with `bench/watch.sh LOG DONE_REGEX PROC_REGEX TIMEOUT_MIN`** (death-aware). Row streaming: a Monitor whose remote command ends in `; true` (a grep on a not-yet-existing log made the whole ssh exit non-zero and silenced an earlier monitor).
- **Scouts:** `research/upcycle-scout.md` (done: do NOT MoE-ify the dense 27B). `ngram-codesign-scout` -> `research/ngram-codesign-scout.md` (running: n-gram x MoE co-design, 4 layers, + KD wall-clock ETA table for Dave's 2xMI210 + 512 GB). Verify its claims against primary sources before acting.
- **When results land:** sync (`rsync root@i5.local:/ai/bench/{ledger.jsonl,*.log} bench/box/`, results into `bench/qual/results/`), `bench/ledger2md.py`, fill the quality table, decide (a) Gemma default file (Q3_K_M vs Q2_K_P on quality), (b) distill vs Qwen3.6-A3B (if the distill wins: ask Dave/davetha to abliterate it, re-bench), (c) Bonsai vs stock Qwen3.8-27B verdict (Bonsai's likely real home = MLX on the Mac, parked), (d) mainline becomes the default tree if the A/B is equivalent, (e) prefetch go/no-go from `predict.py` (only if unigram/bigram recall beats `recent(LRU)` at equal budget).
- **Best configs so far:** Qwen3.6 IQ2_M cache 30 + MTP head n=2 = 46.3 tok/s; Gemma4 Q3_K_M cache 11 + drafter n=2 = 37.2; Gemma4 Q2_K_P cache 15 + drafter n=2 = 48.2; binaries `/ai/src/llama.cpp-moecache/build75/bin` (branch `moe-cache` @ `f94da5a`).

## HANDOFF RUNBOOK (resume from here; written 2026-09-19 ~10:55 EEST)

**State right now (updated 2026-09-19 ~11:05 EEST, model: Fable 5.1):** both builds done. `build/` = `61-virtual`+MMQ+LTO (crashes MoE), `build75/` = arch 75 + `FORCE_MMQ` + `FA_ALL_QUANTS` (9m14s, 0 errors) and also contains `llama-moe-trace`. Branch `i5-tuning` = prism@9a9394a + `bac645e` (Hadamard MTP fix) + `b06d088` (AVX2 PQ2_0 kernel) + `290857f` (llama-moe-trace, drop-in from Shriniwas410/cacheable-by-design; hooks `ffn_moe_topk-<layer>`, verified present for gemma4 + qwen35moe). `specbench.sh` takes `BUILD=/ai/src/llama.cpp/build75` (default `build`).
**Queue step 1 DONE (`/ai/bench/validate75.log`):** build75 best Bonsai config = 5.31 / 5.17 tok/s (parity with 61-virtual's 5.24 / 5.05; accept 0.75 / 0.72). Streamed verify ms/pass b2/b4/b8/b16: `build` (61-virtual) 519/558/650/**694**, `build75` 531/570/659/**955** => the batch-16 win comes from the Pascal code path, not FORCE_MMQ. **Keep BOTH builds:** `build75` for all MoE work (mandatory) and Bonsai MTP n=2 (parity); `build` for Bonsai batches of 9–16 (the n-gram code-edit test in step 3).
**Queue step 4 DEAD (PTQ1_0 graft):** pre-test shows PTQ1_0 is SLOWER under streamed verify despite 28% fewer bytes — b2/b4/b8 = 539/694/815 ms vs PQ2_0 518/559/649; CPU tg 0.58 tok/s (scalar kernel). GPU-side PTQ1_0 unpack is compute-bound on this card, same as BoldingBuilds saw on a 3090. Do not build the graft. PTQ1_0 files (abliterated + stock) are deletable.
**Queue step 2 baselines DONE (2026-09-19 ~11:45 EEST)** — see "MoE baselines" section: Qwen3.6 27.7–29.7 tok/s (flat in placement), Gemma4 16.2 -> 18.3 tok/s (linear in placement), Gemma MTP = wash.
**Profile + traces + sim DONE (~12:35 EEST):** decision = GO on the expert-cache port, Gemma first (see "Router traces + cache simulation").
**Expert-cache port (started ~12:50 EEST):** git worktree `/ai/src/llama.cpp-moecache`, branch `moe-cache` (from `i5-tuning`), own build dir `build75` there (same flags; building in tmux `ai:bmc` -> `/ai/bench/build_moecache.log`, marker `BUILD_MC_DONE`). Commit `8c8650b` = mainline #27861 applied as-is (`research/patches/pr27861.diff`; only reject was an include). Plan: **P1** measure it unmodified on Qwen3.6 (`--moe-expert-cache 32`; the PR only handles separate gate/up + SILU + no scales + `n_tokens == 1`) -> **P2** Gemma support (fused `gate_up_exps`, GELU, `down_exps_s` scale via the inner MUL_MAT_ID node), observe via an op_params flag instead of the `ffn_gate_exps` name match (misses `ffn_gate_up_exps`), allocate cache on the layer's GPU buffer type (bf16-router trap), gate-LRU admission (w16/a3), `synchronize()` before `llama_moe_cache_step()`, pin the CPU mmid nodes to the CPU backend (else `OFFLOAD=2` double-counts), hit/upload/bytes stats -> **P3** multi-token (MTP verify) inside the MMVQ window (IQ3_S 6, IQ2_S 7 tokens) -> then move the singleton into `llama_model` once it has proven a win. Gate: must beat best static placement.
**P2 patch written (not yet applied on the box):** `research/patches/moecache-p2.diff` (on top of `8c8650b`; authored against a local mirror in `/tmp/fork`, not durable). Leaves `ggml.h` untouched on purpose (touching it forces the 9-minute CUDA rebuild).
**Preflight (2026-09-19):** `/ai/bench/preflight.sh` is sourced by `specbench.sh` and `qualbench.sh`: sets + logs governor/THP, refuses when another llama/nvcc process is alive or MemAvailable < 12.5 GB (`PREFLIGHT_MIN_MB`, `PREFLIGHT_ALLOW_BUSY=1` to override). Any new bench script must source it.
**Qwen3.6 MTP (after the cache):** `research/qwen-mtp-scout.md`. The fork already has qwen35moe MTP incl. a standalone-head path: `-md /ai/models/mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp` (ggml-org file, 1,060,038,432 B, sha256 `606fca33…877f`, downloading via tmux `ai:dl`, marker `DL_QMTP_DONE` in `/ai/bench/dl_qwen_mtp.log`). The MTP block is itself a 256-expert MoE layer (experts follow `-ot exps=CPU`). Do not set p-min (fork default 0.0; #24670 is this exact GPU). Expect little until cache hits move verify compute to the GPU (CPU verify cost = k x experts). Missing upstream #28549 (CUDA graph for MTP draft) is a small cherry-pick. Fallback if trunk mismatch hurts acceptance: SassyDiffusion `heretic.IQ2_M.gguf` (11.88 GB, MTP built in).
**Cache P1 + P2 DONE (~12:45 EEST):** see "Expert cache P1/P2". Best configs: Gemma4 `-ngl 999 -ot exps=CPU --moe-expert-cache 16 -ub 128 -b 256` = 24.3 tok/s; Qwen3.6 `-ngl 999 -ot exps=CPU --moe-expert-cache 48` = 34.0 tok/s; binaries in `/ai/src/llama.cpp-moecache/build75/bin`.
**N-gram x MoE co-design (user direction 2026-09-19 late: "ngram distilled moe with proper lfu/lru", "ngram rainbow tables"):** memory-for-compute at every tier. (1) DRAFT: precomputed continuation store = the rainbow-table analog (REST / SuffixDecoding / infini-gram / llama.cpp static+dynamic lookup caches), ideally SELF-distilled from the target's own outputs, mmapped from NVMe; hybrid with MTP (n-gram when the context matches, MTP n=2 otherwise — measured: ngram-mod never drafts on free-form answers). (2) MODEL: hashed n-gram lookup tables as a sparsity axis (DeepSeek Engram, SCONE, Flash-Next's PLE) served from RAM. (3) CACHE: drafted tokens give free lookahead into routing => token-n-gram -> expert table drives PREFETCH into our expert cache (LRU only reacts). (4) TRAINING (Dave's MI210 pair): KD Qwen3.8-27B -> A3B with teacher logits + MTP acceptance + an expert-locality term, i.e. distill a student that is cacheable and draftable. Known limits: cache multi-token path caps at 4 tokens (CUDA MMVQ window 6–8; beyond needs the skip-id primitive, mainline #26631); Qwen3.6's recurrent layers have the same ngram rollback problem as Bonsai (`need_n_rs_seq`), Gemma4 does not => Gemma is the first n-gram testbed. Scout `ngram-codesign-scout` -> `research/ngram-codesign-scout.md` (all four layers + the KD wall-clock ETA table the user asked for).
**Own measurement queued (premise of layer 3):** `bench/box/trace_tok.py` patches `llama-moe-trace` to write token ids (`<trace>.bin.tok`); `/ai/bench/trace2.sh` (tmux `ai:trace2`, runs after `/ai/bench/idlequeue.done`, marker `TRACE2_DONE`) records 40k-token code + prose traces for Qwen3.6 and Gemma4 Q3_K_M into `/ai/bench/traces/*_tok.bin`; `research/scripts/moe-cache-sim/predict.py TRACE.bin` then reports recall@B of a token's real experts from unigram / bigram tables vs what an LRU already holds, per layer band (smoke-tested on synthetic data). Prefetch is only worth building where the table beats `recent(LRU)` at the same budget.
**Upcycle scout verdict (`research/upcycle-scout.md`): (iv) don't MoE-ify Qwen3.8-27B.** Attention+GDN+lm_head ~9B always-active => FFN-splitting bottoms out ~9-10B active, 3-4B unreachable (VERIFIED from config). No existing 3-4B MoE of it. Cheap split+SFT loses 20-45 pts; quality-preserving = 200B-1T tokens; full 27B FT = 432 GB (won't fit friend's 192/256 GB); mixed gfx90a+gfx1201 RCCL has no working evidence. => the friend's box is for imatrix/REAP surgery + KD, not this.
**Qwen3.8-35B-A3B-Distill = the real Qwen3.8-quality-at-A3B-speed play (user OK'd 2026-09-19):** empero-ai distill, arch `qwen3_5_moe_text` (== our qwen35moe), 40L/256e/top-8, vocab 248320, MTP head in-file. IQ2_M 12.56 GB (same class as our Qwen3.6 IQ2_M @ 46 tok/s). Downloading behind q38 (`ai:dl` -> `dl_distill.log`, `DL_DISTILL_DONE`). Benched by `/ai/bench/distill.sh` (cache off / c48 / c30+MTP n=2) + quality -> direct challenger to Qwen3.6-A3B.
**Mainline rebase as an exercise (user OK'd):** we are 431 commits behind because the PrismML fork (needed ONLY for Bonsai's PQ2_0/PTQ1_0) hasn't rebased. Our cache series already rebased onto ggml-org master `e613ef2` (`research/patches/mainline-moecache-e613ef2.diff`, compiles). Plan: `/ai/bench/build_mainline.sh` builds a 2nd tree `/ai/src/llama.cpp-mainline/build75` (does NOT touch the fork), then `/ai/bench/mainline_ab.sh` reruns Qwen c48 + Gemma Q3_K_M c15 on BOTH trees (rows tagged `_fork` / `_main` in the ledger) to measure what the 431 commits buy + prove the mainline port is production-equivalent. Then mainline becomes default for everything except Bonsai (fork or MLX). Also the base for upstreaming (AVX2 PQ2_0 kernel; expert-cache PR).
**Idle queue (tmux `ai:idlequeue`, marker `/ai/bench/idlequeue.done`):** after `QUAL3_DONE`+`Q38_DONE`, runs distill dl-gate -> `distill.sh` -> `build_mainline.sh` -> `mainline_ab.sh`. Each step preflight-guarded (won't run next to a resident model). Watch with `bench/watch.sh`, never a bare grep loop.
**MLX/OSX (parked, user: "runs better on my Mac via MLX anyways ... don't feel like dealing with it yet"):** Bonsai `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` on the Mac is the likely better home for the dense 27B. Not started.

**Bonsai investigation (user flagged its quality/speed 2026-09-19 ~22:00):** Bonsai-2-27B's base IS Qwen3.8-27B, so Bonsai = Qwen3.8-27B abliterated + 2.13 bpw PQ2_0 ternary. Its slowness is structural (dense => every weight per token) AND possibly quant-specific (PQ2_0 ternary matvec is costly on AVX2; no fast K-quant path). Bonsai quality NEVER measured yet. Two tests queued: (1) Bonsai quality row (last block of `qual3.sh`); (2) stock `Qwen3.8-27B-i1-IQ3_M` (mradermacher, 12.77 GB, sha `7544860b1898`, downloading in `ai:dl` -> `dl_q38.log` marker `DL_Q38_DONE`; loads via fork `qwen35` dense arch) benched + quality via `/ai/bench/q38.sh` (tmux `ai:q38`, gated on DL + `QUAL3_DONE`, marker `Q38_DONE`). This isolates "bad because dense" (unfixable) from "bad because THIS ternary-abliterated build" (fixable). Both are still dense 27B => single-digit tok/s either way; the question is quality + which quant is least-slow.
**Qwen3.8-27B->MoE upcycling: PARKED as a training project** but scout `upcycle-scout` (-> `research/upcycle-scout.md`) is checking whether an already-MoE-ified / REAP-pruned Qwen3.8-27B exists (skips all training), the real upcycling token budget + quality delta, and what fits on the friend's 192->256 GB MIXED AMD box (2xMI210 CDNA2 + 2-4x R9700 RDNA4, ROCm). Gate on every path: must beat Qwen3.6-35B-A3B @ 46 tok/s + quality >= Bonsai, which we already run.
**P3/P4 DONE (~21:40 EEST), see "Expert cache P3 + P4". IN FLIGHT now:** tmux `ai:moe6` runs `/ai/bench/moe6.sh` (VRAM-fitted MTP: Qwen 28–30 slots + head n=2/3, Gemma Q3_K_M 11 slots + drafter, Q2_K_P 15 slots + drafter; marker `MOE6_DONE` in `moe6.log`) and then `/ai/bench/qual3.sh` -> `qual3.log` (`QUAL3_DONE`, ~5–6 h: quality for Q3_K_M cache 15, Q2_K_P cache 19, re-run of cut-off answers for `g4_iq3m_cache16` + `q36_iq2m_cache48` under the new caps, then Bonsai). Wait on it ONLY with `bench/watch.sh` (death-aware). Still owed afterwards: cut-off re-runs for the two cache-off reference labels (`q36_iq2m`, `g4_iq3m`), prefill warm-up of the cache, `GGML_CUDA_NO_PINNED=1` test, moving the cache singleton into `llama_model`, upstreaming (AVX2 PQ2_0 kernel, +1 on fork PR #205).
**(done) STAGED for when the quality pass ends (all on the box in `/ai/bench`, mirrored in `bench/box/` + `research/patches/`):** `bash /ai/bench/build_p34.sh` (applies + commits P3 `moecache-p3.diff`, P4 `moecache-p4.diff`, `pr28549-ported.diff`, then FULL rebuild ~10 min because `ggml.h` changes; markers `BUILD_P34_DONE` / `BUILD_P34_FAILED`; P4 has never been compiled — expect to fix errors) -> `bash /ai/bench/moe5.sh > /ai/bench/moe5.log` (P4 overlap on the best cached configs, `-t 5`, MTP n=1/2/3 on top of the cache for Gemma (12 slots + drafter) and Qwen (36 slots + `mtp-Qwen3.6` head), the two K-quant Gemma files cache off/on, 800-token steady state via `GEN=800`) -> quality rows for `Gemma4 Q3_K_M` and `Q2_K_P` (both downloaded + sha256-verified in `/ai/models`) and for Bonsai. If P4 misbehaves (wrong text, hangs), rebuild without it: `git revert` the P4 commit; P3 alone is a 1-file change.
**IN FLIGHT:** `/ai/bench/qual2.sh` in tmux `ai:bench` -> `/ai/bench/qual2.log` (marker `QUAL2_DONE`, ~2 h): quality for Qwen (cache off, resumes the 38 finished items) -> Qwen cache 48 -> Gemma cache 16 -> Gemma cache off. Cache-on vs cache-off scores are the correctness check for the cache. Bonsai quality (~2.5 h) still owed: third block of `qual1.sh`. NEXT after that: P3 (multi-token cache path) + Qwen MTP head test + Gemma MTP with cache.
**(superseded) before the reboot:** `/ai/bench/qual1.sh` in tmux `ai:qual1` -> `/ai/bench/qual1.log` (marker `QUAL1_DONE`): quality pass Qwen -> Gemma -> Bonsai (Bonsai takes ~2.5 h at 5 tok/s; it is resumable, kill the window if the box is needed and rerun later). Scout `qwen-mtp-scout` writing `research/qwen-mtp-scout.md` (source of a graftable Qwen3.6 MTP head). Box harness scripts are mirrored in `bench/box/`.
**Ready for step 2:** expert-cache simulator `research/scripts/moe-cache-sim/sim.py` (smoke-tested on synthetic data; `~/dev/.venv/bin/python sim.py TRACE.bin --experts N --expert-mb X --vram-mb Y`; Qwen3.6: 256 experts, ~1.02 MiB each, 40 layers; Gemma4: 128 experts, ~2.84 MiB each, 30 layers). Trace run: `MOE_TRACE_OUT=x.bin MOE_TRACE_MAX_TOKENS=8000 build75/bin/llama-moe-trace -m MODEL -f corpus.txt -c 512 -b 512 -t 6 -ngl 999 -ot "exps=CPU" --load-mode none` with a code corpus and a prose corpus; sanity-check adjacent-token reuse is well above chance (the tool's README documents the garbage-read failure signature). Port/no-port is decided from these hit rates at OUR spare VRAM, plus the 1080 Ti regression gate below. The expert-cache port itself is Fable's job (it is the active model again; no agent needed).
`/opt/ai` (old copy on btrfs) still exists — delete only after `build75` is validated: `rm -rf /opt/ai`.

**READ FIRST — `research/` is the idea backlog, not an archive.** Before starting any queue item, read the matching sections in full; before declaring any lever exhausted, re-scan them for untried items. Every tok/s gain so far after the AVX2 kernel came out of these files (FFN-only placement, `-ub 128 -b 256`, the 61-virtual MoE crash, the rs-cache sizing). Each entry is tagged with effort and with verified-vs-inferred — trust the verified parts, MEASURE the inferred ones (two scout inferences were already overturned by measurement: "streamed verification is compute-bound on a 1650S" and "#27781 fixes acceptance here").

| file | use it for |
|---|---|
| `research/pr-scout-llamacpp-prs.md` (466 lines) | mainline + PrismML PRs/issues. Section 1 = ranked actionable items with links, quoted numbers, present/absent in prism@9a9394a, effort tag. Also: the (a)/(b)/(c) answers on rs-cache VRAM, pinned memory, draft models; the KNOWN BUGS list (read before every new model/flag); expert-cache prior art + traps (item 7). Raw dumps were in `/tmp/lcpp/` on the Mac (not durable). |
| `research/lit-scout-arxiv-survey.md` (222 lines) | arXiv techniques re-ranked against our measured curves: expected-tok/s table by acceptance x draft length, hybrid n-gram + MTP drafting rules, GDN rollback by reconstruction (frees the ~187 MiB rs cache), expert-cache design (async admission, CPU-computed misses, cache-aware routing), what NOT to do (quantize GDN state, prefetch predictors, LUT kernels on AVX2). |
| `research/moe-scout-abliterated-moe.md` (200 lines) | the MoE shortlist: per-file expert vs non-expert byte splits, KV arithmetic, launch shapes, sha256s of the downloaded files, fallbacks (Gemma Q2_K_P 10.70 GB if RAM is tight; gpt-oss-20b MXFP4; Nemotron ShimQuant), rejected list with reasons. |
| `research/scripts/ggufscan/scan.py` | reads a GGUF header over HTTP range requests -> tensor sizes/types without downloading. Use it to vet any new model file before pulling it. |
| `research/scripts/axv/{q,ids,ax}.py` | arXiv / alphaXiv query helpers, for follow-up literature searches. |

Untried items worth pulling from there, beyond the queue below: mainline #28739 (OOB fix for small `GGML_OP_OFFLOAD_MIN_BATCH` — we run `=2`; merged upstream, absent in prism); mainline #28549 (CUDA graph for the MTP draft, "+4–5%", merged, absent); `--spec-draft-type-k/-v` to shrink the draft context so MTP n=3 fits; fork issue #143 in-place GDN state (+6.7–7.2% tg, unmerged); mainline #21067 `--prefetch-weights` (overlap upload with compute — the only upstream work on our exact PCIe bottleneck; port needed); BITCOS-style 1.70 bpw packing (arXiv 2609.16338, Bonsai-27B is 29.66% zeros). When a new question comes up, spawn a scout and have it WRITE its report into `research/` first, message second — messages truncate.

**Rules learned the hard way:**
- **llama-server's host prompt cache (`--cache-ram`, default 8192 MiB) grows ~72 MiB per request** and `--n-cpu-moe` puts experts in PINNED host memory (shmem, unreclaimable): the Qwen quality server reached 14.3 GB RSS and was OOM-killed on 2026-09-19 12:46 when a download added page-cache pressure. Harness now always passes `--cache-ram 0`, quality reference runs use plain `-ot exps=CPU`, and every ledger record carries `mem_avail_mib_min` / `swap_used_mib_max`. No downloads while a server is resident either.
- **systemd-oomd kills the whole tmux scope on memory PRESSURE, not only on OOM** (15:06: Gemma cache-off quality run, scope at 10.8 G, pressure avg10 61%). Gemma IQ3_M leaves only ~2.2 GB `mem_avail_mib_min`; ~10 GB of the server is pinned host memory (shmem). Added `/swapfile-ai` 8 GB (fstab, pri -3; remove: `swapoff /swapfile-ai`, delete the fstab line + file). Knob to test for headroom: `GGML_CUDA_NO_PINNED=1` (host buffers become ordinary swappable memory; costs upload speed). Q3_K_M Gemma will be ~0.9 GB tighter still.
- **RAM is the hard limit (16 GB, 2 GB swap): never run a build, a second model, or anything RAM-hungry while a `--load-mode none` server is resident (Qwen/Gemma hold ~11–12 GB).** A `-j5` CUDA build next to the Qwen quality server wedged the box into swap on 2026-09-19 ~12:00 and needed a hard reset. `free -m` before every launch; quality runs and builds are strictly sequential.
- One measurement at a time on the box. Never bench while a build, download, or another model is running. Never edit source while a build is running.
- Long jobs go in tmux session `ai` with output to a log + a DONE marker; wait with an `until grep -q MARKER log; do sleep 20; done` loop. Do not `pkill -f` a pattern that matches your own ssh command line.
- A config only counts if it completes a generation (loading is not a fit test). Always `--parallel 1` (MTP acceptance -> 0 with `-np > 1`).
- Measure tok/s as completion_tokens / wall or the server's `timings.predicted_per_second`, via the chat endpoint (needs the chat template), never SSE chunk rate.
- Verify the filesystem before writing anything big: `findmnt -T <path>`. `/opt` and `/mnt/md0` are spinning btrfs.
- Greedy output is not bit-identical between spec and non-spec runs (batch variance) — do not gate correctness on byte identity; use `test-quantize-fns` + `test-backend-ops` for kernels.
- Python on the box: `/ai/.venv/bin/python` only.

**Harness:** `cd /ai/bench; [MODEL=/ai/models/x.gguf] [OFFLOAD=32|2] ./specbench.sh NGL LABEL [llama-server args…]` -> two prompts (code, reasoning), 200 tok, temp 0, thinking off, plus a 1 Hz telemetry line (GPU util/power, PCIe->GPU, CPU busy, DRAM read). Server log: `/ai/bench/server_LABEL.log`. It uses `build/bin/llama-server`; to test another build edit the path in `specbench.sh` (or add a `BUILD` env var). `OFFLOAD` = `GGML_OP_OFFLOAD_MIN_BATCH` (2 = stream CPU weights to GPU for batches >= 2; use 32 for no-spec baselines).

**Queue (in order):**
1. **Validate `build75`:** run the best Bonsai config (see "Best Bonsai-27B config") on `build75` and the streamed batch timing `GGML_OP_OFFLOAD_MIN_BATCH=2 build75/bin/llama-bench -m …Abliterated-PQ2_0.gguf -ngl 24 -fa 1 -t 6 -p 2,4,8,16 -n 0 -r 2 --load-mode none` (61-virtual build: 517/563/647/693 ms; old arch-75 build without FORCE_MMQ: 519/558/649/946). If build75 >= parity, make it the only build (the `61-virtual` build crashes MoE `MUL_MAT_ID` on batches of 5–7, ggml-org#24064).
2. **MoE baselines on build75** (both files in `/ai/models`, byte-verified; sha256s in `research/moe-scout-abliterated-moe.md`):
   - Qwen3.6-35B-A3B IQ2_M: `-ngl 999 -ot "exps=CPU"`, then sweep `--n-cpu-moe N` downward until VRAM is ~3.4 GB. No MTP tensors in this file.
   - Gemma4-26B-A4B IQ3_M: same, then the drafter `-md /ai/models/mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp --spec-draft-n-max 1` and `2`, add `--spec-draft-p-min 0.0` if `draft_n` stays 0 (ggml-org#24670). Do NOT combine ngram-mod with Gemma MTP (40 -> 5 tok/s, ggml-org#24266).
   - For MoE, `OFFLOAD=2` may hurt (verify batches touch the union of experts) — A/B `OFFLOAD=32` vs `2`.
   - Then trace expert reuse (`llama-moe-trace`, github.com/Shriniwas410/cacheable-by-design) BEFORE any cache code. **Gate:** a GTX 1080 Ti (also no tensor cores) REGRESSED 19.3 -> 13.3 tok/s with a cache; if the trace shows low reuse or the first prototype regresses, drop the port rather than sink effort into it.
   - **Expert-cache port = the one non-mechanical step.** Touches the scheduler, CUDA `mul_mat_id`, and async uploads; several of pr-scout's traps fail SILENTLY (duplicate dummy slots -> illegal memory access in batched CUDA `mul_mat_id`, cache init must precede `sched_reserve()`, ungated admission churns the cache: "window 16, admit 3" gave 48–51% hit / 15k uploads vs ungated 37–43% / 55k–74k). Spawn a dedicated agent for it (Agent tool, `model: "fable"` — Fable built this harness and the AVX2 kernel; or `subagent_type: "fork"` to inherit full context) with `isolation: "worktree"`, hand it pr-scout item 7 verbatim, then REVIEW the diff before it reaches the box and run every benchmark yourself (one measurement at a time). Cache design + prior art: `research/lit-scout-arxiv-survey.md` items 6–8, `research/pr-scout-llamacpp-prs.md` item 7 (port base: ggml-org/llama.cpp#27861; async admission, misses computed on CPU, admission gate "window 16, admit 3"; a no-tensor-core GTX 1080 Ti REGRESSED with a different design — measure early).
3. **ngram on Bonsai:** local patch in `common/common.h:387` `need_n_rs_seq()` — add the NGRAM types so rollback uses rs snapshots instead of full host checkpoints; cap `--spec-draft-n-max 8` (each snapshot ~62 MiB). Test with a code-EDIT prompt (paste ~80 lines, ask for a small change) and `--spec-type ngram-map-k` / `ngram-mod --spec-ngram-mod-n-match 8 --spec-ngram-mod-n-min 8`, FFN-on-CPU placement, MTP off (VRAM).
4. **PTQ1_0 + MTP graft:** gguf-py script copying the 15 `blk.64.*` tensors + `qwen35.nextn_predict_layers` + `block_count=65` from `…PQ2_0-MTP.gguf` into `…Abliterated-PTQ1_0.gguf` -> ~18% fewer bytes over PCIe per verify pass. BoldingBuilds saw only +1.6% on a 3090 (compute-bound there); here it is PCIe-bound, so test one verify-batch timing (`llama-bench -p 2,4,8` with OFFLOAD=2 on the plain PTQ1_0 file) BEFORE building the graft.
5. Housekeeping: persist governor=performance + THP=always (systemd oneshot); delete `/opt/ai`, the stock prism-ml GGUFs, and davetha's 8.25 GB PQ2_0; upstream the AVX2 kernel to PrismML-Eng/llama.cpp (they merge outside PRs) and +1 fork PR #205 (same Hadamard fix).

## Next

- [ ] rebuild at `/ai`, then delete `/opt/ai`
- [ ] llama-server + `--spec-type draft-mtp` on the MTP file with `GGML_OP_OFFLOAD_MIN_BATCH=2`; sweep `--spec-draft-n-max` 2..8; stack `ngram-mod`. Measure `completion_tokens / wall`, not SSE chunk rate (undercounts under spec decode).
- [ ] A/B the Pascal/FORCE_MMQ CUDA build; FA on/off; `-ctk q8_0 -ctv f16`; `--ctx-checkpoints` for multi-turn (GDN state cannot roll back => full re-prefill on prefix mismatch)
- [ ] persist governor + THP
- [ ] MoE candidate that fits (scout running) — that is where LRU/LFU expert caching applies
- [ ] osx (mlx): https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit — not started

## UPDATE 2026-09-20 20:26 box clock (Fable) — ctx1 read: restore NOT dead, the restore row was the wrong test

- ctx1 restore rows: prompt_n 12397 == save 12397. Server log: `forcing full prompt re-processing due to lack of cache data (hybrid/
  recurrent)`. Cause = my row design: slot saved AFTER 64 generated tokens, restore re-sent the bare prompt => needs a recurrent
  rollback. Slot I/O worked (307 MiB 0.16-0.6 s; 723 MiB at 131k 0.4 s). Untested: restore + token-exact EXTENSION. => brief 17
  (slotclient presave/extend, token-id arrays on /completion) + box job `ctx1b` (16k PRE_GEN x DROP matrix, gate, two-phase 32k/131k/262k).
- ctx1 numbers that stand (n=1): 32k prefill ub 512/1024/2048 = 75.7/107.6/135.4 t/s (fit predicted 138); decode 27.8 @12k, 28.1 @27k,
  11.5 @120k (q4 KV, cache 24); 32k decode by KV: f16 28.1, q8 25.7 (-9%), q4 22.1 (-21%); -nkvo 9.0 (dead); 131k prefill 46.2 t/s = 43 min.
  64k F16 OOM, 32k MTP cache 16 OOM, 262k died on a 926 MiB compute buffer (ub 512 + cache 12), not on KV => two-phase (cache 0 prefill,
  ub 128 decode) is what makes 262k fit.
- ctxproxy consequence: a chat-level proxy only gets a cache hit on a hybrid if the re-rendered conversation is TOKEN-EXACT to the
  saved ids (template re-rendering of assistant turns breaks that). Carry into the brief-16 review.
- kq1 (K2 experts): +15.6% decode ALL best-config, quality in band, n=1 — reserved call. Qwen queue 2 done; 16 partial, see HANDOFF-16.

## UPDATE 2026-09-20 ~22:00 box clock (Fable) — queue drained: kq1 / ctx1 / mainqual / h2qual / orf1 / r1b / ctx1b run 1

- **ctx1b run 1: slot restore + token-exact extension WORKS on the GDN hybrid.** All four 16k combos (PRE_GEN 64|1 x DROP 0|1):
  prompt_n 14-15, cache_n 12460/12461 (or 12397), 0.5-1.0 s wall vs 147 s cold, no "forcing full prompt re-processing". Reply was
  1 token (end-of-turn): EXT was raw text inside the assistant turn, tokenized with parse_special false (my brief-17 spec error).
  Gate correctly refused (no coherent reply = state validity unproven) => deep rungs not run. Fix: brief 18 (parse_special true) +
  chat-formatted EXT in ctx1b.sh (c96bdd7). Rerun = `queue.sh retry ctx1b` after deploying Qwen's 18.
- **r1b**: speed rows all OOM (cache 48 + 727 MiB MTP head do not fit; design flaw, MTP rows belong at cache 30). Quality rows at
  c48 no-MTP, n=50/41, GSM8K / HE / decode: b0 96.0 / 92.7 / 38.0; b0.25 98.0 / 90.2 / 39.1; b0.5 98.0 / 95.1 / 40.2;
  b1.0 98.0 / 97.6 / 41.4 (+8.9%); b2.0 **86.0** / 95.1 / 42.1. b1.0 passes the gate at this footing, b2.0 fails (GSM8K -10, 1.8 sigma).
  Missing: quality at the shipped footing (c30 + MTP, where r1 measured +20-27% decode) => `r1c` queued. Keep/kill = Andrei.
- **kq1** (K2 experts): +15.6% decode ALL best-config (46.1 -> 53.3), +21% no-cache, quality in band, n=1. bench/reports/kq1.md.
- **mainqual**: mainline q36 100.0 / 92.7 / 77.1; g4 Q3_K_M 98.0 / 95.1 / 81.4 — in band of the fork rows.
- **h2qual** (n=200/280): GSM8K cannot separate the four (95.0-96.5, SE 1.4). MMLU-Pro: g4 IQ3_M 76.4, Q3_K_M 75.0, Q2_K_P 73.9,
  q36 71.4 (SE 2.6). Speeds 25.1 / 31.1 / 38.5 / 38.0 t/s. The "slow IQ3_M" pick is no longer supported by quality: +2.5 MMLU-Pro
  (< 1 sigma) for -35% speed vs Q2_K_P.
- **orf1**: compliance q36 100%, g4 Q2_K_P 100%, stock distill 48% (metric validated by the contrast row).
- Qwen: brief 17 delivered clean in one turn (small + fenced works; the big brief 16 wedged twice). 16 = partial, HANDOFF-16.

## DECISION 2026-09-20 (Andrei): mainline + our patches = default tree. New box jobs build/run from /ai/src/llama.cpp-mainline.
- K2 (Qwen experts) and Gemma Q2_K_P: NOT switched. Andrei does not follow the case from the scoreboard => defaults stay IQ2_M /
  as-is until he says otherwise. `kq1h` queued: K2 at the h2qual sample size (GSM8K-200 + MMLU-Pro-280) for a row directly
  comparable to IQ2_M's 96.0 +- 1.4 / 71.4 +- 2.7. R1 waits on r1c, long-context config on the ctx1b rerun.

## DECISION 2026-09-20 (Andrei): correctness over speed — "not faster at getting it wrong". Gate = paired non-inferiority (SPEC 8.x).
- Consequences: R1 stays OFF (b2.0 dead: loses 7 / wins 3; b1.0 loses 0 / wins 3 of 91 but unproven, r1c is the last box time it gets
  until a ~1000-question paired run is worth it); Gemma stays IQ3_M (Q2_K_P -1.7 [-4.4, +0.9], not proven); q4 KV at depth cannot
  ship without lq1 (q8 where it fits); K2 undecided (-0.6 [-5.3, +4.1] at n=161; it has MORE bits than IQ2_M, 12.9 vs ~11 GB, so
  the prior is not "lossy shortcut"; kq1h adds n=480). Lossless levers lead. Brief 19 = durable paired.py (send after 18 lands).

## PROPOSAL 2026-09-20 (Fable, Andrei's idea: "roll our own quant = our rainbow table") — workstream QX, not yet approved
- [M] Routing concentration from trace2 (Qwen3.6, 40 layers x 256 experts, top-8, ~40k tokens each): top 10 / 25 / 50% of experts
  carry 53.9 / 77.9 / 92.8% of routing mass on code, 44.9 / 72.4 / 91.5% on prose (per-layer top-25%: 49-85%). BUT the hot sets
  are workload-specific: top-25% overlap code vs prose = 24% = chance (n=2 corpora; confirm with a third before building on it).
  => a GLOBAL "hot experts get more bits" file is wrong; a WORKLOAD-SPECIALIZED file (e.g. code: code-hot experts Q4_K, rest Q2_K,
  same size) is the quantization analog of a single-purpose model and needs no training. Fused expert tensors hold one type, so
  per-expert types need a hot/cold tensor split in the loader (C++).
- Our objective is not the one public quants optimize: (1) CPU kernel speed on the miss path (AVX2, no AVX-512), (2) bytes per
  expert (VRAM slots + PCIe), (3) accuracy. Codebook/trellis formats (IQ2_*, AQLM, QuIP#, QTIP) win accuracy per bit but decode
  slowly on CPU (the IQ2_M miss cost K2 removed) => codebooks only for GPU-resident tensors, linear k-quant blocks on the CPU path.
- Order (each step gated: KL divergence vs Q6_K_P logits to screen, paired non-inferiority for finalists):
  QX1 imatrix from the Q6_K_P source with expert-balanced calibration (kq1's imatrix came from the IQ2_M model; rare experts are
  under-calibrated) — zero C++, Dave's box; QX2 per-layer x per-role type search with existing ggml types — zero C++; QX3 hot/cold
  expert split for a code-specialized file — C++ loader; QX4 port ik_llama.cpp IQK types only if QX2 shows the FORMAT is the limit.

## UPDATE 2026-09-20 late (Fable) — r1c verdict, queue reordered after the scouts, kld1 built
- **r1c (shipped footing: cache 30 + MTP n=2), GSM8K / HE / t/s:** b0 98.0 / 92.7 / 45.0; b0.5 100.0 / 92.7 / 52.6 (+17%);
  b1.0 **92.0 / 90.2** / 57.4. The bias bites harder at the smaller cache, as predicted: b1.0 (clean at c48) loses GSM8K -6 at c30.
  R1 = OFF (accuracy-first). b0.5 is unproven, not pursued. No more box time on R1.
- Queue after the scout findings: ctx1b rerun (brief 18 deployed; lossless, the 262k goal) -> kld1 (NEW: KL divergence vs the Q6_K_P
  reference for IQ2_M and K2; the sensitive quality instrument, also the QX screening rig) ; kq1h keeps running (MMLU-Pro-280 is
  not saturated, gives the paired n=480 read for K2).
- Next builds: Unsloth-recipe reader (GGUF header -> --tensor-type args), hard reasoning set (AIME 24/25 + MATH-500 L5 +
  EvalPlus, thinking on), Bonsai quality redo, lq1 with 5 KV arms.

## UPDATE 2026-09-20 night (Fable) — three builds done, queue = ~28 h of accuracy-first work
- **Hard quality sets** (qual.py + fetch.py --hard, 9 tests): aime (AIME 2024+2025, 60, integer), math_l5 (MATH-500 L5 numeric
  golds, 40), humaneval_plus (EvalPlus tests on the same 41 tasks). Thinking on, caps 24.5k / 16k / 8k tokens. EvalPlus executor
  verified on the box (canonical passes, wrong fails, sandboxed uid nobody + numpy). Box job `hq1.sh iq2m|k2` (~8 h per arm).
- **lq1** rebuilt on Qwen's longctx (cherry-picked 50662a0): 5 KV arms f16 | q8_0 | q8_0/q5_1 | q4_0 | q4_0 no attn-rot, cache 0,
  131k for the quantized arms; longctx got a reuse guard (deep depth stops when the 2nd request re-prefills) + test.
- **bonsai1** easy (resumes the killed run, paired vs q36 rows) and hard (AIME first 10, thinking, -c 16384).
- Queue: kq1h (running) -> ctx1b rerun -> kld1 -> hq1_iq2m -> lq1 -> bonsai1_easy -> bonsai1_hard. hq1 k2 arm waits on kld1.
- Qwen root cause for its two stalls: typing the literal chat special-token strings ends its own generation (server stop token).
  Briefs must never require it to write them; told it so. Queue 5 = briefs 19 (paired.py) + 20 (gguf_types.py).

## RESULT 2026-09-20 (Opus) — ctx1b GATE PASSED: slot restore + token-exact extension WORKS on the GDN hybrid
- 16k, pre_gen=64 drop=0 (the production/ctxproxy path — state saved AFTER 64 generated tokens, extend with a chat-formatted
  continuation): restore reprocessed prompt_n 32 of 12397 (cache_n 12460), wall 3.0 s vs 148.7 s cold = ~50x, decode 30.5 t/s,
  reply COHERENT. All four PRE_GEN x DROP combos passed; gate picked pre_gen=64 drop=0. drop=0 (keep every saved id) works, so the
  last sampled token IS in the restored KV.
- Kills the earlier ctx1 "restore dead" verdict: that re-sent the bare prompt against a post-generation slot => forced rollback =>
  full re-prefill. Extend-the-token-ids is the supported path. The two-phase architecture (prefill server -> slot -> decode server)
  is VALID; deep rungs (32k/131k/262k two-phase) now running. Restore turns the 43-min 131k prefill / hours-long 262k prefill into
  a ~3-5 s slot load => 262k is reachable on the 4 GB card.
- Consequence for ctxproxy (brief 16, still partial): a chat proxy on this model must send the continuation as TOKEN IDS extending
  the saved prefix (the slotclient presave/extend path), NOT re-rendered chat text — re-rendering breaks token-exactness => full
  re-prefill. The current ctxproxy forwards chat messages, so it will miss on this model. Carry into the brief-16 rework.

## RESULT 2026-09-20/21 (Fable, verified against ctx1b.log) — two-phase long context WORKS through 131k; 262k server is up
- **Cross-config restore (the two-phase claim):** slot saved by the PREFILL server (cache 0, ub 2048: 32k in 200.7 s, 135.9 t/s)
  restores under the DECODE server (cache 24, ub 128): prompt_n 32, cache_n 26887, 3.1 s, decode 28.6 t/s, coherent reply.
- **131k:** prefill server (q4_0 KV, cache 0, ub 1024) 119,569 tokens in 2110 s (56.8 t/s; ctx1 had 46.2 at ub 512 + cache 24);
  slot 723 MiB saved in 0.4 s; decode server restore + 32 new tokens + 64 generated = **6.6 s vs 35 min (~320x)**, decode at
  120k depth 12.4 t/s, reply coherent and specific to the document.
- **262k:** the prefill server LOADED (cache 0 freed what the 926 MiB compute buffer needed) and is prefilling; ctx1's 262k died at load.
- **MTP after a restore is a net LOSS with the separate -md head:** 17.2 t/s vs 28.6 without, acceptance 26/72 = 36% (normal ~85%).
  The standalone head is its own model with its own (empty) KV: it drafts blind to the restored context. Prediction: the IN-MODEL
  head (V1c / kq1graft file) shares the target's state and should keep its acceptance after a restore => test row (ctx1c).
- Correction to the previous entry: with drop=0 the server still re-evaluates the last saved token (prompt_n 32 = 31 extension
  ids + 1), with drop=1 it is 31; both work. The claim "the last sampled token IS in the restored KV" was an inference, not shown.
- New bottleneck for long context = DECODE AT DEPTH (28.6 t/s @27k F16 -> 12.4 @120k q4_0; expect ~6-7 @240k). lq1 gives
  decode t/s per KV type per depth for free (each question decodes 48 tokens at depth).

## DESIGN 2026-09-21 (Fable) — FA1: GQA-aware quantized decode attention (the decode-at-depth lever; lossless)
- Symptom: decode 28.6 t/s @27k -> 12.4 @120k (q4_0 KV). Attention adds ~0.45 ms per 1k tokens of q4 KV (F16: ~0.30); the
  bandwidth bound of the card is ~0.03 (q4) / 0.11 (F16) ms per 1k. At 240k: ~135 ms/token now vs ~35-45 ms achievable.
- Cause, read in our tree (ggml/src/ggml-cuda/fattn.cu, ggml_cuda_get_best_fattn_kernel): cc 7.5 takes the
  `turing_mma_available` branch (a GTX 16xx has NO tensor cores; the check cannot tell it from an RTX 20xx). There:
  quantized KV + 1 query token => VEC kernel; quantized KV + >1 token (MTP verify) => MMA_F16 with need_f16_K/V = the used KV is
  dequantized to F16 into a pool buffer EVERY step; F16 KV + GQA => MMA_F16. The VEC kernel launches one block per QUERY head
  (blockIdx.z = head, K/V = head / gqa_ratio): with Qwen3.6's 16 q heads on 2 KV heads, the same K/V rows are walked and
  dequantized 8x per step.
- The VEC kernel (fattn-vec.cuh, warp-synchronous) already has a column dimension `ncols` (1 or 2 query TOKENS) that shares the
  K walk and dequantizes each V row ONCE for all columns (VKQ[j] += tmp * KQ_k[j]). GQA heads of one group are mathematically
  more columns: same K/V rows, same mask row per token, slope 1 when max_bias == 0.
- Patch plan: template param ncols2 (GQA group width, 8 here); column j -> (head0 + j / ncols, token ic0 + j % ncols) for the Q
  pointer, mask row (token only), sink (per head) and dst index; launch grid z over head GROUPS (launch_fattn already knows
  ncols2 for the tile/mma kernels); KQ shared array = ncols*ncols2*D (8 x 256 floats = fine under the 48 KB shared limit);
  instances limited to D=256, ncols2=8, K/V in {q4_0/q4_0, q8_0/q8_0, q8_0/q5_1, f16/f16} to bound compile time; dispatch:
  use it when gqa_ratio % 8 == 0 && max_bias == 0 && Q->ne[1] == 1 (then ne[1] <= 3 for the MTP verify batch, ncols = 4 with
  the existing out-of-bounds column handling) instead of VEC-per-head / MMA_F16-with-dequant.
- Gate: test-backend-ops -o FLASH_ATTN_EXT green (CUDA vs CPU) incl. a gqa 8 / D 256 / q4_0 case; temp-0 text identical to the
  old kernel on the 131k slot; prof3 (nsys at 131k, queued) sizes the prize first, the same job after the patch measures it.
  Build + test go THROUGH the box queue (a CUDA compile next to a benchmark corrupts the benchmark).

## RESULT 2026-09-21 (Fable) — FA1+FA2 patch BUILT (810 s, no errors) and unit-tested; arch finding
- bench/box/patches/fa-gqa-{vec,dispatch}.patch: vec FA kernel gets head-group columns (ncols2 = 8) => one K/V walk per K/V head
  for its 8 query heads; quantized KV + <= 4 query tokens routed to it (decode AND MTP verify batches, which used to dequantize
  the used KV to F16 every step for MMA_F16). Env GGML_CUDA_FA_VEC_GQA 0|1|2, GGML_CUDA_FA_VEC_GQA_NCOLS1.
- test-backend-ops FLASH_ATTN_EXT, run NEXT TO the 262k prefill (70 MB VRAM free): all 80 added cases (hs 256, gqa 8/16, q4_0 / q8_0 /
  q8_0+q5_1, 1-5 query tokens, sinks on/off) PASS vs the CPU reference under the new kernels. 4054/4059 new vs 4056/4059 old: every
  failure is an F16-KV nb=512 kv>=4096 case (unreachable by the patch, fails on the old kernels too, set varies run to run) =
  memory pressure; the queued fa1 verify reruns with the GPU free, then text-identity + decode t/s from the 131k/262k slots.
- llama.cpp's own banner: "suboptimal performance due to a lack of tensor cores: GTX 1650 SUPER. Consider compiling with
  CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and GGML_CUDA_FORCE_MMQ". All our mainline numbers are arch-75 builds => arch1
  (build started, verify queued right after fa1): llama-bench A/B short context + depth 32768.
- Priority rule (Andrei): new code is built + verified ahead of benchmarks; builds run CPU-side next to the GPU job as system units.

## CONTAMINATION 2026-09-21 (Fable, my error) — ctx1b 262k presave shared the box
- While `ctx1b_c262k_pf_presave` was prefilling I ran, outside the queue: the fa1 build (4 threads, 810 s), the FA unit test on the
  GPU (2 x ~155 s), and ~6 min of the arch1 build (stopped). => that row's prefill t/s and wall time are NOT a measurement; rerun the
  262k prefill number clean if it is ever quoted. The slot file it writes is valid (same computation), and every later row (262k
  extend, fa1 A/B from the kept slots) runs with the box exclusive. Rule fixed in memory: code PREEMPTS benchmarks, never runs beside them.

## RESULT 2026-09-21 (Opus) — fa1 run 2: FA1 decode kernel WINS (+10%), FA2 verify path LOSES, NTC1 is a wash
- **Gates all green**: test-backend-ops FLASH_ATTN_EXT 4079/4079 under GQA=2, GQA=0, GQA=2+F16, and NTC=1+GQA=2+F16.
- **FA1 (1 query token, quantized KV) = the win.** Real workload, restore from the kept 131k slot: decode 12.68 -> **13.99 t/s
  (+10.3%) with a BYTE-IDENTICAL reply** (prompt_n 27, cache_n 119632 both runs). llama-bench tg32 @ d32768 q4_0: 16.11 -> 17.61
  (+9.3%). => default ON (mode 1).
- **FA2 (2-4 query tokens / MTP verify) = a regression, default OFF.** pp3 @ d32768: 25.30 -> 23.14 (-8.5%); 1-token-per-block
  variant worse (22.19); real MTP round 33.01 -> 32.44 t/s, acceptance flat (108/181 vs 106/183). Reading: giving up the
  tensor-core matmul of MMA_F16 costs MORE than the whole-KV F16 dequant it performs each step. Patch default changed 2 -> 1;
  GGML_CUDA_FA_VEC_GQA=2 keeps it available for a later retry (e.g. after the cache/quant work changes the balance).
- **NTC1 (no-tensor-cores attention dispatch) = within noise**: tg32 @ d32768 16.37 vs 16.11; short context pp512 112.0 vs 111.2,
  tg64 24.41 vs 24.93. Scoped to dispatch after run 1's crash, it now changes little. arch1 (FORCE_MMQ + Pascal arch BUILD) is the
  remaining form of that question.
- **Run-1 crash, root cause (my bug, fixed):** forcing MMQ via turing_mma_available() desynced host tile-config selection
  (mmq.cuh ~190) from device kernel selection (__CUDA_ARCH__, mmq.cuh ~279) => scratch buffer sized for DP4A tiles while MMA
  kernels ran => illegal memory access at q4_0 m=16 n=8 k=256. Lesson: a runtime switch may only touch predicates used for
  RUNTIME DISPATCH, never ones that pick a kernel variant whose device code is compile-time selected.
- **My harness bugs, not code bugs:** the 262k A/B rows died (`-c 262144` + cache 8 + restore does not fit; needs cache 0-4 or
  the two-phase split) and the F16 llama-bench rows passed -ctk/-ctv flags that config rejects. Both fixed in the next run.
- MTP text differs after 287 chars between GQA=0 and GQA=2 — expected: MMA-on-F16 vs vec-on-q4 are different arithmetic, so a
  late greedy tie can flip. The FA1 decode path (the one shipping) was byte-identical on the 131k slot.

## RESULT 2026-09-21 (Fable, review of the Opus turns + two results it did not see) — arch1, kld1, the 262k OOM
- **arch1 [M, n=2 short / n=1 depth]: the Pascal-path build (61-virtual;80-virtual + FORCE_MMQ) prefills 3x faster**: pp512
  336.1 vs 113.4 t/s. Decode is slower there: tg64 26.42 vs 28.76 (-8%), pp3 @d32768 17.93 vs 23.48 (-24%); tg32 @d32768 19.04 vs
  17.94 (+6%). The arch-75 MMQ tiles are sized for tensor cores the GTX 1650 lacks. The old "4 s fixed per ubatch = PCIe expert
  streaming" reading (lat1 / ctx1 fit) was WRONG: most of it was that matmul path. => two-phase gets TWO BINARIES: Pascal build
  = prefill server, arch 75 + FA1 = decode server. ctx1d verifies on the real server (32k presave both builds, ub 4096, cross-build restore).
- **kld1 [M]: vs the Q6_K_P reference (wikitext-2, 40 x 512): IQ2_M mean KLD 0.2188, PPL ratio 1.180, same top token 79.8%,
  99% KLD 2.00; K2 0.2181 / 1.165 / 79.9% / 1.88.** (1) K2 >= IQ2_M in fidelity (tail better), + paired non-inferior, + 15.6%
  faster => every instrument so far favors K2. (2) BOTH are heavily degraded: the served model disagrees with its own
  high-precision self on 1 token in 5. The standard sets hide it. hq1 shows whether it bites on hard reasoning; RAM1 (32 GB =>
  3-4 bit experts) and QX have large headroom to attack. Report: bench/reports/kld1.md. Caveat: wikitext is off-distribution
  for a chat model; add a code / chat corpus before quoting the absolute level.
- **Why every 262k decode row died (ctx1b extend, fa1 A/B): runtime OOM** = model ~1430 + KV 1440 + RS 63 + cache 8 (349) +
  compute buffer **698 MiB even at ub 128**; ~512 MiB of it is the reserve to dequantize the whole used KV to F16 for the
  multi-token attention path on quantized KV. FA2 (GQA vec, <= 4 query tokens) needs no such copy => a decode server at
  -ub 4 -b 4 with GGML_CUDA_FA_VEC_GQA=2 should reserve almost nothing and fit 262k WITH a cache. FA2 lost on speed but may be
  what makes 262k decode fit. ctx1d tests it (queued first).
- Corrections to the Opus entries: (1) FA2's loss is NOT "giving up the tensor-core matmul" — this card has no tensor cores;
  the MMA/tile kernels simply have better arithmetic intensity than the vec kernel, whose threads split ONE 256-dim dot 32
  ways (latency design, poor at 100k+ keys). (2) FA1's +10% (not the ~3x I predicted) says the vec kernel at depth is bound by
  the per-head dot/accumulate work and its fine-grained threading, not by K/V dequant. The real fix is a throughput-oriented
  quantized decode kernel (each thread = many keys, full-width int8 dots, GQA-shared K unpack); prof3 sizes it.
  (3) the 262k OOM cause above replaces "cache 8 does not fit" as the explanation.

## RESULT 2026-09-21 (Fable) — ctx1d + ctx1c: 262k decode works; two-binary two-phase verified; MTP is a net loss at depth
- **262k decode [M, n=1], from the kept slot (239,170 tokens restored in 0.87-0.97 s, request total ~10 s vs 2 h 10 min cold):**
  cache 0 / ub 128 / FA1: decode **8.68 t/s**, compute buffer 698 MiB, 3322 MiB VRAM — it FITS once the expert cache is 0.
  ub 4 + FA2 (mode 2): compute buffer **7.11 MiB** (the ~512 MiB F16-KV reserve is gone, hypothesis confirmed), cache 16 = 8.67,
  cache 20 = 8.79 t/s (3482 MiB); Pascal build same (8.78). Suffix prefill 9.2 t/s at ub 4 vs 14.3 at ub 128. All replies coherent.
  **Reading: at 240k the expert cache is worth ~1% — the round (~114 ms) is attention.** Simplest 262k decode config = cache 0,
  ub 128, FA1. The lever left for long-context decode is the attention kernel itself (throughput-oriented quantized decode
  kernel; prof3 sizes it), not VRAM.
- **Prefill server on the REAL server, 32k, cache 0 [M, n=1]:** arch 75 ub 2048 = 136.4 t/s (199 s); **Pascal-path build ub 2048 =
  328.1 t/s (85 s, 2.4x); ub 4096 = 362.7 t/s (77 s, 2.66x)**, compute buffer 1014 MiB at ub 4096. A slot written by the Pascal
  build restores under the arch-75 decode server (3.1 s, decode 28.9 t/s, coherent) => **two-phase = two binaries, verified.**
- **ctx1c: the in-model MTP head is NOT better after a restore, and "blind after restore" was my wrong inference.** Acceptance
  was already 29/66 = 44% in the fresh presave run (no restore involved) and 27/71 = 38% after the restore; decode 18.5 t/s with
  MTP vs 23.75 without, same slot. ctx1b's -md head: 36%. So MTP acceptance is LOW AT 27k DEPTH on this prompt regardless of
  head or restore (it is ~85% in the short benchmarks) => long-context decode runs WITHOUT speculation until someone shows
  otherwise; worth one row on a second prompt type before calling it general.

## 2026-09-21 (boot 3) — queue autostart root cause, hq1 was off-spec, prof4, FA3 in flight

**ai-queue never started at boot: systemd ordering cycle.** ai-queue `After=ai-perf-tweaks`, ai-perf-tweaks `After=multi-user.target`,
multi-user.target wants ai-queue => systemd deleted the ai-queue start job ("Job ai-queue.service/start deleted to break ordering
cycle"). Fixed: ai-perf-tweaks is now `After=power-profiles-daemon.service`. `systemd-analyze verify` clean; unproven until the next boot.
**Governor kept falling back to powersave:** power-profiles-daemon re-applied `balanced` after our sysfs write. The tweak unit now runs
`powerprofilesctl set performance` first (governor + EPP = performance, checked). Every run logged `governor=performance` in its
preflight line, so past numbers stand.
**queue.sh:** `systemctl stop` could record the killed job as `failed` (no requeue). The runner now traps TERM and leaves the row `running`.

**hq1 runs 1+2 are void — the harness was off the model card.** 3 of 3 items ran the whole cap INSIDE the reasoning block (2 AIME at
24.5k, 1 MATH-L5 prealgebra at 16k). The new per-row trace says repetition 0.00 = a live chain that never concludes, not a loop. We
decoded greedily; the Qwen3.6-35B-A3B card (read 2026-09-21) specifies thinking = temp 1.0 / top-p 0.95 / top-k 20 / min-p 0 /
presence 1.5 (coding: temp 0.6, presence 0), budget 32,768 normal and up to 81,920 for competition math. qual.py now samples per the
card with a seed hashed from the item id (reproducible, both paired arms share seeds), caps 32k math_l5 / 41k aime (81k of F16 KV
does not fit beside cache 24), rows from another sampler are rerun. Whether the 2.5-bit files overthink is still OPEN — that is what
the rerun measures. Arms are now ~12-20 h each.

**kq1graft (K2 + in-model MTP head).** Equal slots (26): in-model 50.6/48.4/56.8/43.0 vs separate head 54.0/54.6/58.2/45.6 t/s
(code/reason/edit/long) = 2-11% slower; it frees 528 MiB, and at 38 slots (same VRAM, 3446 MiB) it is 56.6/56.2/60.1/46.7 = +3-5%
over the separate head. Promotion stays Andrei's call.
**Temperature-0 output is not run-to-run stable:** the same config (K2, separate head) answered the `reason` prompt with two different
openings in prof2_perf vs prof2_nsys. Likely the expert cache: a cached expert runs the CUDA kernel, an uncached one the CPU kernel,
and which is cached depends on history. This is the noise floor of every paired text comparison; fa3 measures it (two VSKIP=0 runs).
**prof2 (CPU side, perf):** 48% of samples in libgomp (threads spinning at barriers while the GPU works), 19% q2_K dot, 13% q3_K dot.

**prof3 was blind, prof4 is the real decode-at-depth profile.** nsys 2022.4 does not trace kernels launched from CUDA graphs, so prof3
held one decode step. prof4 (graphs off, FA1 build, kept 120k slot, q4_0 KV): `flash_attn_ext_vec` 4.36 ms/layer (GQA off) vs 3.66
(FA1) = 57% / 53% of GPU kernel time per token; 12.89 vs 13.78 t/s with graphs off. Everything else is small (largest: mmvq 72 us x
2520). CORRECTION to my earlier "9x headroom to the bandwidth floor": the kernel is COMPUTE bound — ~1 G multiply-adds per layer per
token (16 heads x 120k x 256 x {K dot, V axpy}) on a ~4 TFLOPS card; the 0.4 ms bandwidth floor is not the binding one. A plausible
optimum is ~1.5-2 ms. The lever is less WORK, not a cleverer walk.
**FA3 (fa-vskip.patch, opt-in `GGML_CUDA_FA_VEC_VSKIP=1`):** skip the V dequant + axpy of (position, head) pairs below 1e-8 of the
running max (KQ_sum still exact; skipped mass <= n_kv x 1e-8). Lossy in principle => ships only on proof: unit gates, per-token top-10
logprob drift vs VSKIP=0 on the same restored slot (probcmp.py) read against the VSKIP=0-twice floor, at 120k and 239k. Mode 2
(diagnostic build only) times the K walk alone = the ceiling for any V-side work.
**fa1 worktree had a stale mmq.cu hunk** (from the dropped ntc-mmq patch: force MMQ under NTC mode; fa1.sh did not reset that file).
Inert unless GGML_CUDA_NO_TENSOR_CORES=1, so FA1/FA2 numbers stand; the "NTC1 = wash" row included forced MMQ. fa1.sh now resets it
(takes effect on the next full fa1 build; the fa3 job does not touch mmq.cu).

**FA3 result (2026-09-21): V-skip is DEAD, and it located the real cost.** Unit gates 4079/4079 under both modes. Decode at 120k:
14.89 / 14.63 (VSKIP=0 twice) vs **7.69** (VSKIP=1) vs 18.30 t/s (timing-only, no V work); at 239k: 8.89 / 9.00 vs **4.10** vs 11.24.
The data-dependent branches halve the kernel's speed (loss of unrolling / register spill), and even a free V walk is only +23-26%.
Logprob drift VSKIP=1 vs 0: mean |dlogprob| 0.10-0.13, max 0.9-1.15 with identical text — far more than a 1e-8 mass bound
explains; unexplained, not pursued (slower AND unexplained = dropped; patch and job deleted, commit 212b038 has them).
Two keepers: (1) **run-to-run floor is exactly 0.0** on this prompt at both depths (VSKIP=0 twice: identical tokens and logprobs), so
probcmp drift is signal, not noise; (2) the timing-only arm puts the **K walk at ~2/3 of the attention kernel** (step 67 ms at 120k:
~24 ms K walk, ~12.5 ms V walk) => FA4 = row-owning K walk (fa-krow.patch): each thread owns a K row, loads it once for the 8 columns,
exact integer block sums, one float multiply-add per block, one max-reduction per column per tile instead of a 5-step reduction per
(position, column). Estimated ~3x fewer K-walk instructions. fa4.sh also measures GQA=0 vs FA1 drift = the yardstick for what an
already-accepted kernel change does to logprobs.

**FA4 result (2026-09-21): row-owning K walk works.** Unit gates 4079/4079 (KROW 0 and 1). Decode, same kept slots, graphs on:
120k: upstream 12.41, FA1 14.64, **KROW 17.05 / 17.17** (+17% on FA1, +38% on upstream); 239k: 8.32, 8.99, **10.95 / 11.00** (+22% / +32%).
Logprob drift (top-10, 96 tokens): KROW twice = 0.0 exactly; upstream vs FA1 = mean 0.136 / 0.146, max 1.03 / 1.08; KROW vs FA1 = mean
0.134 / 0.136, max 1.79 / 0.77; text diverges after token 83-94 in 3 of 4 comparisons. Reading: FA1 vs upstream is the same math in a
different float order, and it moves logprobs as much as KROW does, so ~0.13 is what ANY reordering does to this 2.5-bit model at 120k+
(this also explains FA3's "unexplained" 0.10-0.13: V-skip was not inaccurate, only slow). Side finding worth keeping: at depth the
model's output is chaotic in the float order — long-context quality claims need task-level measures (lq1 / lq2), not text identity.
KROW stays opt-in in code until the next build (no preemption for a default flip); launch configs set it.

**fa5probe (2026-09-21): exact / bounded-error block-sparse attention is DEAD on this model.** Real Q and K of all 10 attention
layers at 120k (summarization prompt), decode steps 8 / 32 / 64, per KV-head group of 8 query heads, 128-position tiles, n=60:
tiles with every weight < 1e-8 of the max: **0.0%** (oracle, so no scheme can do better); < 1e-6: 0.0% for the group (0-19% per
single head); < 1e-4: 10.9% mean for the group (0-63%), 5-80% per head. The three per-tile upper bounds (Quest min/max on the
stored Hadamard-rotated channels, centroid + radius, min/max after a plain un-rotation) PROVE 0.0% even at 1e-4. The logit range is
compressed (QK-norm): nothing is provably negligible. This is also why FA3's V-skip only ever paid its branch cost — nothing was
under its threshold. What IS there: mass concentration — the top 1% of tiles hold 63% of the softmax mass on average (35-88%), the
top 5% hold 79% (53-98%). Exploiting that is LOSSY top-k attention (3-47% of the mass dropped at 5%), not a bounded-error
optimization; under accuracy-first it would need lq-style task proof per threshold and is parked. Long-context kernel work stops
at FA4 unless a new idea shows up.
Next C++ target is the SHORT-context round instead: ~31 of its 58 ms are CPU expert matvecs (~12 miss pairs x 3.4 M weights per
layer in ~0.78 ms = ~9 G MAC/s per core, about a third of what AVX2 int8 can do); prof2 puts only ~60% of CPU-phase thread time in
the q2_K / q3_K dots, the rest is per-op barriers on 65-microsecond matvecs. Plan: a task-parallel fused expert FFN on the CPU
backend (a thread owns whole experts: gate_up -> act -> down with no barrier in between, one barrier per layer). Ceiling ~+26% on
the round, realistic +10-15%; needs a perf profile of the CPU phase alone first.

**Scout vs measurement (bounded sparse attention, 2026-09-21).** `research/scout-2026-09-21-bounded-sparse-attention.md` recommends
BLASST-style inline thresholding (skip a tile when tile max - running max < ln(lambda)) and expects 50-70% of tiles pruned per GQA
group. fa5probe measured the thing the scout flags as unverified: for a group of 8 heads at 120k, 0.0% of tiles are under 1e-8 or
1e-6 and 10.9% under 1e-4 — and BLASST needs the tile's K.Q dots BEFORE it can skip, so it only ever saves the V walk (~1/5 of the
step; FA3 territory), never the K walk that dominates. The measurement wins: not pursued. The paper numbers (70-90% per head,
"250x sparsity tolerated", arXiv:2605.24168 [L, unverified]) are lossy top-k regimes (lambda 1e-3..1e-2), consistent with our mass
concentration (top 1% of tiles = 63% of the mass) and parked under the same rule as FA5. Kept fact [scout-verified from config.json]:
Qwen3.6 partial_rotary_factor = 0.25 (64 of 256 head dims rotated).

**cpu1bench (2026-09-21): CPU1 is DEAD — the CPU expert phase sits on the DRAM wall, not on barriers.** (Control: `cpu1bench2` entry below, which CONFIRMS the memory-bound reading.) Microbenchmark, real
shapes / types (gate, up Q2_K 2048->512; down Q3_K 512->2048; 256 experts, distinct random experts per call), microseconds per
layer call:
| pairs | ggml t1 | ggml t6 | task-parallel t6 (a thread owns whole pairs, one barrier per layer) |
|---|---|---|---|
| 4  | 582  | 174.8 (3.33x) | 193.7 (3.01x) |
| 12 | 1721 | 489.1 (3.52x) | 484.0 (3.56x) |
| 24 | 3423 | 952.1 (3.60x) | 938.7 (3.65x) |
The task-parallel layout buys nothing: ggml's row split is already fine. Both stall at ~3.5x on 6 cores because 12 pairs x ~1.1 MB
of expert bytes in 0.49 ms = **~27 GB/s of DRAM traffic** (1 thread = 22 G MAC/s = 8 GB/s, compute bound; 6 threads = memory bound;
confirmed by the `cpu1bench2` control below. "= the DDR4-2667 ceiling" was a shade too strong: the measured peak on this box is
~38 GB/s, 27 is what six concurrent ~1 MB streams get). The telemetry's "DRAM 5 of 38 GB/s" was a whole-round AVERAGE that hid the
burst; SPEC 4.1's "misses are compute-bound" was true for the IQ2_S dot and is no longer true with K-quant experts.
Consequences: (1) no CPU kernel work — the only lever on the miss term is BYTES per token: hit rate (VRAM) or bits per expert;
(2) higher-precision experts cost speed in proportion to their bytes on the miss path (Q4_K pair = 1.75 MB vs 1.1 => miss phase
+55%) and cost cache slots on the hit path (1.6x VRAM per expert) — the QX3 / RAM1 accuracy levers have a quantified speed price,
and a hot / cold split is the way to pay less of it; RAM1 adds capacity, not bandwidth (B460-class boards cap at 2666);
(3) the server spends ~0.78 ms per layer on 12 pairs where the bare ops need 0.49: ~0.3 ms per layer (~12 ms of a 58 ms round) is
CPU<->GPU handoff (copies, syncs, graph splits) = the last software slack in the short-context round; needs a timeline
measurement (the prof2 nsys file has the memcpy / sync trace) before any code.

### 2026-09-21 — foreign CPU beside whittle1, and a detector for it

Andrei's GitHub Actions runner (`ci-runner@1/2`, vite builds) ran on the box during whittle1; load average hit 9.7. He stopped it.
- The two speed rows are clean: box-wide CPU busy was 335% (c0) and 319% (c16) while they ran, which is the server alone.
- The contaminated window is the quality section. Accuracy does not depend on CPU contention; the per-item t/s there is not used.
- `mon.py` now samples per-process CPU at 1 Hz and sums everything that is not `llama-*`/`nvidia-smi`/`perf`. A window averaging over
  50% of one core prints `FOREIGN CPU [label]` above the telemetry line and sets `foreign_cpu_flag` in the run's mon.json.
  Idle baseline on the box is about 6% of a core. Same pid + same start time = same process (kernel workers rename themselves).

### 2026-09-21 — whittle1: Whittle-Qwen-3.8-35B-A3B (i1-Q2_K) killed on quality

First contact with the dense-27B-distilled-to-A3B model. It loads and runs on our mainline tree; the 10B n-gram table
(`per_layer_token_embd.weight`) is pinned to the CPU by name, experts on the CPU, 2458 MiB VRAM at cache 16.

| row | result | reference (Qwen3.6 IQ2_M, same harness, think off) |
|---|---|---|
| decode, cache 0 / 16 | 29.5 / 34.8 t/s, no MTP head | ~56 t/s served config |
| expert-cache hit rate, 16 slots | ~43% | |
| GSM8K-200 | 81.5 +-2.7 | 96.0 +-1.4 |
| MMLU-Pro-280 | 43.6 +-3.0 | 71.4 +-2.7 |

- Kill rule was "> 2 sigma under either set or < 30 t/s". It is 5 and 7 sigma under. 38 of 480 answers hit the 2048-token cap;
  counting all 38 as correct still leaves MMLU-Pro at 57%.
- Our GSM8K equals the author's self-reported 41-44/50, so the weak row is the model as published, not only the blunt Q2_K quant.
  The card calls it a research preview (3.3 h of joint training on 1,840 traces).
- Speed rows are clean (see the foreign-CPU entry above); the runner overlapped only the accuracy section.
- Consequences: no `hq1 whittle` arm; T5 (continuing the recipe on Dave's box) stays parked; DM1's only live item is now `dense1`.
- The 13 GB file stays on the box until Andrei OKs deleting it (`/ai/models/Whittle-Qwen-3.8-35B-A3B.i1-Q2_K.gguf`).

### 2026-09-21 — cpu1bench2: the control CONFIRMS the memory-bound reading (an earlier version of this entry had it backwards)

`cpu1bench` re-drawn vs FIXED expert ids. Fixed = the same 12 pairs every call, so the bytes are served from the CPU cache
(12 pairs = 13.7 MB against a 12 MiB L3) and never cross the DRAM bus. The decision rule was written BEFORE the run (handoff):
"jumps toward 5-6x => DRAM bound confirmed; stays ~3.5x => retract".

| pairs (bytes) | re-drawn t1 -> t6 | fixed t1 -> t6 |
|---|---|---|
| 4 (4.6 MB, fits L3)   | 590 us -> 175.1 (3.37x) | 511 -> 111.9 (4.57x) |
| 8 (9.1 MB, fits L3)   | 1170 -> 350.3 (3.34x) | 1015 -> 228.2 (4.45x) |
| 12 (13.7 MB, ~L3)     | 1730 -> 473.9 (3.65x) | 1626 -> 326.5 (**4.98x**) |
| 24 (27 MB, 2x L3)     | 3496 -> 955.0 (3.66x) | 3464 -> 838.3 (4.13x) |

- Take the DRAM traffic away and 6-thread scaling goes 3.65x -> 4.98x, t6 drops 31%. The gain shrinks exactly where the fixed set
  stops fitting the L3 (24 pairs: 4.13x). **=> at 6 threads the miss phase is memory bound; about a third of its time is waiting
  on DRAM.** Single thread barely moves (21.8 -> 23.2 G MAC/s): compute bound, as stated.
- The first version of this entry (written by the cheaper model) RETRACTED the reading with "a bandwidth wall cannot be lifted by
  changing which addresses you touch". That is a fallacy: touching the SAME addresses means the bytes come from cache, which is the
  whole point of the control. It also inverted the pre-registered rule. Its two follow-ons are withdrawn too:
  (a) "hugepages / locality is a cheap lever" — THP is `[always]` here and the bench weights are one anonymous 278 MiB buffer
  (139 huge pages, fits the TLB), so TLB misses are not the cost; an expert is a ~1.1 MB sequential stream, far above any
  prefetch / page granularity, so placement does not help. (b) "the QX3 / RAM1 speed price is void" — it stands: on a memory-bound
  path the miss phase scales with bytes per expert (Q4_K pair 1.75 MB vs 1.1 => about +55% on that phase). qx3 measuring it
  directly is still welcome, but the estimate is not void.
- What the control does NOT separate: bandwidth from latency. 27 GB/s is under the ~38 GB/s measured peak, so "memory bound" is the
  proven statement, "at the hard DDR4 ceiling" is not. The lever is unchanged either way: BYTES per token (hit rate, bits per
  expert). CPU1 stays dead. HAND1 (~0.3 ms per layer of CPU<->GPU handoff) is untouched.

### 2026-09-22 — dense1: Qwen3.8-27B FFN energy is FLAT. DM1's static split is closed. (And a parser bug that almost said otherwise)

llama-imatrix on the dense 27B (24 chunks, 507 s, PPL 2.34), then `bench/imx_heat.py` on the `ffn_down` input statistics
(= mean squared activation per FFN neuron, 64 layers x 17,408 neurons).

| | top 1% | top 5% | top 20% | top 50% | Gini |
|---|---|---|---|---|---|
| median over layers | | 0.32 | **0.52** | | 0.43 |
| L0 (most concentrated region, layers 0-3: top-20% up to 0.93) | 0.47 | 0.66 | 0.80 | 0.91 | 0.76 |
| L16 | 0.11 | 0.23 | 0.45 | 0.73 | 0.36 |
| L32 (flattest) | 0.07 | 0.18 | 0.42 | 0.70 | 0.31 |

- Verdict by the pre-registered rule (median top-20% < 0.60): **FLAT**. A static hot set of 20% of the neurons carries about half of
  the FFN energy; our A3B MoE gets 11x sparsity because it was TRAINED for it. With `whittle1` killed the same night, DM1 has no live item.
  Report: `bench/reports/imx_heat_qwen38_27b.json`.
- **Tool bug, caught on the first real file.** `imx_heat.py` (built from brief 26) read GGUF tensor data at absolute offsets; GGUF
  offsets are relative to the ALIGNED END OF THE HEADER. Its test-side GGUF writer made the same mistake, so the tests agreed with
  the bug. On the real file it printed MODERATE (median 0.77) from misaligned garbage; the tells were a negative energy share on L0
  (impossible for a sum of squares) and "zeros 21" on every layer. Fixed: data-section start, a guard that refuses negative energy,
  the test writer now uses relative offsets, and a round-trip test against the real `gguf` library. All 496 entries now match the
  library bit for bit. Lesson for briefs: a parser test must include one file written by the reference implementation.

### 2026-09-22 — hq1_iq2m final (served IQ2_M file, thinking on, card sampler)

| set | correct | cut at the budget | finished chains correct | looping |
|---|---|---|---|---|
| MATH-L5 (40) | 23 = 57.5% +-7.8 | 16 at 32k | 23 of 24 | 0 |
| EvalPlus (41) | 35 = 85.4% +-5.5 | 1 at 8k | 35 of 40 | 5 |
| AIME (15) | 3 = 20.0% +-10.3 | 10 at 41k | 3 of 5 | 0 |

When a chain finishes it is almost always right; the losses are chains that never finish inside the card's own budget. Whether that
is the 2.5-bit quantization or the fine-tune is what `hq1_stock` (running since 01:24) answers. No verdict before it lands. The t/s
in this job is not usable: model files were being copied to the array beside it.

### 2026-09-22 — hq1_stock: the served fine-tune is WORSE than stock at the same quant class (paired, p 0.04)

Same 96 items, same card sampler, same seeds; stock = bartowski imatrix IQ2_M of Qwen3.6-35B-A3B (12.96 GB).

| set | served fine-tune IQ2_M | stock IQ2_M | paired (stock - served): wins / losses, diff, 95% CI, exact p |
|---|---|---|---|
| MATH-L5 (40) | 23 = 57.5%, 16 cut at 32k | 32 = 80.0%, 7 cut | 10 / 1, +22.5, [+7.8, +37.2], p 0.012 |
| AIME (15) | 3 = 20.0%, 10 cut at 41k | 7 = 46.7%, 7 cut | 4 / 0, +26.7, [+4.3, +49.1], p 0.125 |
| EvalPlus (41) | 35 = 85.4%, 1 cut at 8k | 32 = 78.0%, 6 cut | 1 / 4, -7.3, [-17.8, +3.1], p 0.375 |
| ALL (96) | | | 15 / 5, +10.4, [+1.5, +19.3], p 0.041 => served file WORSE |

(`paired.py q36_stock_iq2m_hard.jsonl q36_iq2m_hard.jsonl`; the table flips its sign to read as stock minus served.)
Reading: on hard math the fine-tune's chains run longer and more of them never finish inside the card's own budget (16 vs 7 cut on
MATH-L5); when a chain finishes, both files are almost always right (served 23 of 24, stock 32 of 33). The quant class is the same,
so the 2.5-bit quantization is not what separates them. Caveat: the two files come from different quantizers / imatrices, so "the
fine-tune" is the lead cause (inferred), not an isolated one. Code goes the other way and is not significant (stock cuts 6 at 8k).
Per the pre-registered rule this feeds C2 (which file serves hard reasoning) — Andrei's decision. t/s: 32.0 stock, clean (no FOREIGN).

### 2026-09-22 — qx3 failed on a tool bug, not on the model: fixed and re-queued

`expert_mix.py` refused `mix25` with "HI/LO tensor mismatch: blk.0.ffn_down_exps.weight ([256, 2048, 288] vs [256, 2048, 220])".
Those are BYTE shapes: X4's down experts are Q4_K (144-byte blocks), K2's are Q3_K (110-byte blocks); the element shape is
(256, 2048, 512) in both. The mix dequantizes both sides into a Q8_0 container, so only element shapes must agree. The pre-check
now compares element shapes (header-only inversion, no data read); a test with HI in a block type and LO in F16 reproduced the
exact box error before the fix. Checked on the real files: 733 tensors, 0 element-shape mismatches, 120 byte-shape differences
(= 40 layers x 3 expert tensors in different types). The X4 arm already ran before the failure: mean KLD 0.1370 vs K2 0.2181
(-37%), same top token 84.0% vs 79.9%, 99% KLD 1.03 — the all-Q4_K ceiling is well below K2, so QX3 is alive; the mixes decide GO.
`hand1` finished (sqlite exports of prof2 / prof4 present) => B1 is unblocked.

### 2026-09-22 — HAND1: the "0.3 ms handoff" was the serial GPU dense phase; real handoff is 84 us per layer

Source: the prof2 nsys trace (K2, cache 26, MTP n 2, short context), `bench/box/hand1_phases.py prof2_nsys.sqlite`. Decode runs
through CUDA graphs, so nsys 2022.4 sees no kernels there (same blind spot as prof3), but the host CUDA API trace is complete:
per layer the main thread does D launch (async: expert-cache hits) -> CPU experts -> H2D of the host outputs -> B launch + a
long sync (serial device work) -> D2H of hidden state + router. Over 8,153 decode layers:

| per layer (mean) | us | per 40-layer pass |
|---|---|---|
| CPU experts (gap after the D launch) | 473.0 | 18.9 ms |
| device phase (sync after the B launch) | 458.1 | 18.3 ms |
| cudaStreamSynchronize (short ones) | 35.4 | 1.4 ms |
| cudaGraphLaunch API | 26.5 | 1.1 ms |
| cudaMemcpyAsync API | 12.4 | 0.5 ms |
| host gaps between calls | 9.9 | 0.4 ms |
| period | 1018.3 (p10 754, p50 955, p90 1287) | 40.7 ms |

Reading: (1) the server's CPU expert phase is 473 us, the SAME as the bare ops (0.49 ms, cpu1bench): the 0.78 ms figure was
inferred from a round model, never measured, and nothing is lost on the CPU side. (2) The device phase is REAL kernel time, not
latency: a stretch of the trace that ran without graphs (7.6 s) shows back-to-back kernels with ~0.5 us gaps, ~470-520 us per
layer: merge of host + cache outputs, the SHARED EXPERT (~55-60 us), residual / norm, the next layer's attention (QKV mmvq 70-120,
FA 59, out 58) or delta net (in-proj 70, concat_non_cont 53, state get_rows 27, gated_delta_net 57, out 62 + 57), router + top-k.
It runs while the CPU waits, and the CPU experts run while the GPU idles apart from the hit chain. (3) The true handoff (syncs,
launches, copies, gaps) is 84 us per layer = ~3.4 ms per pass, not ~12 ms.
Levers, all bit-identical (same kernels, same inputs, reordered or moved), ranked by size:
- **SHEXP overlap** (~2 ms per pass): qwen35moe builds the shared expert AFTER `build_moe_ffn` (`qwen35moe.cpp:468` then
  `:488-496`), so it lands in the serial B split. It depends only on the layer input, so it can go into the D split next to the
  cache-hit chain and run while the CPU does the misses. Needs a hook in our `build_moe_ffn` barrier section.
- **GDN concat** (~1.5 ms per pass): `concat_non_cont` takes 53 us per delta-net layer (30 of them) to move ~100-200 KB; a
  contiguous layout or a better non-contiguous kernel is pure data movement.
- **Async H2D of the host expert outputs** (~0.7 ms): the 196,608 B result copy goes through the scheduler's synchronous fallback
  (CUDA cannot `cpy_tensor_async` from the CPU backend): sync, memcpy, sync 18.8 us (15.8 of it the transfer), then the ~20 us
  graph launch. Enqueued on the stream, the launch would overlap the transfer. Needs care on host-buffer reuse.
- Small: the two D2H copies each carry a sync (~4 us).
Ceiling of the three: ~4 ms of the ~58 ms round (~7%). The rest of the device phase is dense-weight matvecs (bandwidth bound,
changing them is a precision question), and CPU vs GPU are serial by data dependency across layers.

### 2026-09-23 — HAND1 levers 2 and 3 built; all three go through one gate job (hand2)

- 0008 (CUDA concat): the non-contiguous kernel ran one 256-thread block per row; the delta-net conv state is 6 elements x 8,192
  rows (3 cached + 3 new tokens, the new part a transpose), so 8,192 blocks each used 6 threads. New flat kernel (one thread per
  destination element, identical element selection) whenever ne0 <= 64. test-backend-ops gains the transposed-b variant (v & 16)
  at the real shape for 1 / 3 / 4 / 17 tokens, plus a perf case.
- 0009 (ggml-backend sched): a host -> device split input the backend cannot `cpy_tensor_async` went sync + copy on the per-thread
  stream + sync (18.8 us per layer for the 196 KB expert outputs, 15.8 of it the transfer, before the ~20 us graph launch). It is
  now `ggml_backend_tensor_set_async` on the split backend's stream when the source is a host buffer. Safe because every later
  host split first synchronizes the device backend (it copies inputs from it, or has none and syncs the previous backend).
- Build: the ov tree is a `cp -a` of the served tree; after touching the old objects, make rebuilt exactly ggml-backend.cpp,
  concat.cu and test-backend-ops.cpp (checked with make -n first; the new kernel symbols are in libggml-cuda). `shov1` was replaced
  by `hand2` (unit -> identity -> nsys mechanism -> ABAB speed); a divergence gets bisected over the three commits.

### 2026-09-23 — work units already resume; the queue now treats a clean stop as a pause

Audit (Andrei: nightly downtime lands mid-job): `qual.py` (hq1 / race1 / bonsai1) appends a row per item and skips finished ids,
`longctx.py` (lq1 / lq2) skips finished (depth, kind, index); proof: `hq1_stock`'s restart kept its first 15 items (96 rows, 96
unique; an earlier "hq1.sh has no partial resume" in this log was wrong). A stop costs only the item in flight (<= ~17 min on the
32k MATH-L5 chains). Two real gaps, fixed:
- `queue.sh`: every clean stop used to spend a try (MAX_TRIES 3), so a long job across three nightly downtimes would be failed out.
  The TERM trap now puts the running job back to pending and gives the try back (`PAUSE` in queue.log); a crash / power loss (no
  trap) still counts and stays bounded. `bench/tests/test_queue.py` (runs on the box: needs flock + setsid) — 4 cases, green 3x.
  Deployed by atomic mv (the live runner keeps the old inode; bash reads scripts incrementally); active from the next start.
- `qx3.sh`: per-arm resume. A finished arm writes qx3_kld_LABEL.done = hash of its inputs (models by size + mtime, hot set and
  mix tool by content, chunk count); a restart reprints the stats instead of re-running (and skips rebuilding a 36 GB mix). The X4
  arm's key was seeded from the 09-22 19:20 measurement (same X4 file, 40 chunks).
Not resumable by design: A/B jobs (hand2, specbench arms) — a paired comparison should not straddle a reboot; ~45 min to redo.

### 2026-09-23 — hand2: the three HAND1 levers are +5.4% decode; identity needed a deterministic cache (hand2id)

prof2 config (K2, cache 26, MTP n 2), ABAB, decode t/s:
| prompt | base_a / base_b | ov_a / ov_b | gain |
|---|---|---|---|
| code | 52.43 / 50.82 | 56.04 / 57.67 | +10.1% |
| reason | 53.47 / 54.06 | 55.37 / 56.59 | +4.1% |
| edit | 57.93 / 56.83 | 59.57 / 59.22 | +3.5% |
| long | 45.43 / 45.05 | 47.58 / 46.63 | +4.1% |
Mean +5.4%, every prompt above base by more than the base spread; cache hit rate equal in all four arms (49.1 / 49.0 / 49.0 /
48.6%), so the gain is work removed, not a luckier cache. Unit: test-backend-ops CONCAT 182/182 (CUDA vs CPU).
Mechanism (nsys API trace, `hand1_phases.py`): serial GPU phase 467 -> 399 us per layer (0007 + 0008), short syncs 36.1 -> 18.6
(0009), graph launch 26.2 -> 35.9 (the D graph is bigger); everything that is not CPU work -69 us per layer (-12.5%). OPEN: the
CPU phase in that profiled run was 464 -> 526 us; those two servers logged no cache stats, so a lower hit rate there cannot be
ruled out — not read as a result. (nsys under the queue service leaves a raw .qdstrm: its importer is in
/usr/lib/nsight-systems/host-linux-x64; hand2.sh now imports it explicitly. hand1_phases.py anchors on the API table.)
Identity: UNDECIDED — base diverged from ITSELF on code and long (both builds flip between the same two variants). Cause, from
the source: the expert cache publishes an upload at the first step() after the PCIe copy finishes, so hit (GPU kernel) vs miss
(CPU kernel) depends on timing and near-ties flip. Patch 0010 adds LLAMA_MOE_CACHE_SYNC=1 (off unless set): step() waits for
every scheduled upload, so uploads from step N always publish at step N+1 and hit/miss is a function of the token history.
`hand2id` runs det (served code + 0010) vs ov (det + 0007-0009) under it, base twice as the floor. The same switch gives every
future paired text comparison a real noise floor. Trees: /ai/src/llama.cpp-det (moe-cache-det), llama.cpp-ov (shexp-overlap-det).
Queue: the new pause-on-stop was exercised live (PAUSE race1, tries restored to 1).

### 2026-09-23 — hand2id: IDENTICAL; 0007-0010 promoted into the served build

Under LLAMA_MOE_CACHE_SYNC=1 the floor is clean (det_a vs det_b identical on all 4 prompts) and det vs ov is byte-identical on all
4 prompts in both pairs. The cache counters match exactly across all four runs (1,792 steps, hit 48.9%, 33,171 uploads, 32,131
evictions): the sync mode is fully deterministic. (Its decode speed is not a measurement: every step waits for the uploads.)
With hand2's +5.4% that is the pre-registered GO. `/ai/src/llama.cpp-mainline` (branch moe-cache) fast-forwarded to
shexp-overlap-det; build75 rebuilt incrementally in 70 s (libllama, ggml-base, concat.cu). From here every queue job runs the
faster build; texts differ from older runs only by the cache-timing noise that was always there.
Slips, all caught: the promotion build overlapped race1's start, whose preflight refused the busy box (failed before any item,
requeued); a `pkill -f` pattern matched its own ssh command line; and the red test_queue runs had leaked four runners polling
deleted scratch dirs (harmless to the real queue, but CPU noise for the FOREIGN detector) — killed, and the tests now reap their
runners in cleanup even when an assertion fails.

### 2026-09-23 — opt1 (zero-code levers on the promoted build) and the prefill-mode build

opt1, decode 4-prompt mean vs 3 interleaved bases (55.69 t/s, spread 4.07), long-prompt prefill vs 48.6 t/s:
-t 5 -6.2% (dead) | OMP_PROC_BIND=close + OMP_PLACES=cores -8.2% (dead) | GGML_CUDA_NO_PINNED=1 -1.7%, prefill -7% (dead: THP
engaged, 6.6 GB AnonHugePages, so the CPU phase is not TLB bound) | MTP n-max 3 +0.5% (parked: +3.7/+7.8 on code/edit, -3.2/-8.1
on reason/long) | **-ub 256 at cache 26: prefill +50% (72.8 t/s), decode -0.2% = free**. Offline: non-uniform cache slots per
layer at equal VRAM = +0.08 / -0.05 pts (dead; `research/scripts/moe-cache-sim/alloc.py`). The multi-turn probe OOMed on my own
config (-c 16384 + cache 26 + the MTP head); it reruns in pmux1 at 8k.
Prefill mode (Andrei: "prefill and decode don't share VRAM?" — they did not): `--ubatch-prefill N` on branch prefill-mux. A batch
larger than -ub suspends the expert cache (drops queued uploads, waits for one in flight, frees the device slot buffers, lookups
return nullptr so graphs build without the cache chain), re-reserves the scheduler for N (the node budget of this architecture
grows with the ubatch past ~590 tokens, so growth-on-demand alone would overflow the hash set), and the first small batch
re-reserves for -ub BEFORE the slots are re-allocated (all-or-nothing; failure = cache off, outputs unchanged) and re-ranked by the
prompt's routing. Only the context that owns the cache and runs the main decoder toggles it (the MTP draft context calls decode
too). Build slip, caught: the relocated test tree had no dependency info for untouched objects, so the server library kept the
old common_params layout and --help asserted on n_gpu_layers; every non-CUDA target was rebuilt clean (159 objects). The served
build was never affected (built in its own tree with full dependencies).

### 2026-09-23 — prefill mode works end to end (pmux1 -> pmux2); ubp 2048 = 3.19x prefill, decode -2.0%

pmux1: prefill 97 / 129 / 156 t/s at ubp 512 / 1024 / 2048 (48.8 base) but the slots never came back ("could not re-allocate")
so decode ran cache-less (-12..-20%). Two bugs, both mine: (1) suspend freed the slot buffer but left every slot tensor's data /
buffer pointing at it, and ggml_backend_alloc_ctx_tensors_from_buft only allocates tensors whose data is NULL -> resume
"allocated" nothing; (2) the CUDA VMM pool keeps the prefill's temporaries forever -> ggml_backend_cuda_trim_pools (proc address,
called after the shrink). After both: released 1172.9 MiB, 1388 MiB free after the trim, restored 1172.9 MiB, every switch.
pmux2 (served config, K2, cache 26, MTP): long prompt (2,181 tok) prefill 48.8 -> 155.3 t/s at ubp 2048 (3.19x), long wall 51.0 ->
20.4 s; decode 4-prompt mean -2.0% (base spread 2.88) = GO; decode on the long prompt right after the switch = base (47.3 vs
47.5 / 48.6: the prompt re-ranking works). ubp 1024 -9.2%, mostly on code / reason, whose prompts never switch (noise, inferred).
Identity (LLAMA_MOE_CACHE_SYNC=1, served ub128 vs ubp 2048): code / reason / edit IDENTICAL (edit, 377 tok, went through prefill
mode), long diverges at char 318 on a near-tie. Two ULP-level causes (ubatch boundaries; cache re-ranked by the whole prompt vs
per 256-token chunk); pmux4 measures the first as a KLD. Multi-turn (thinking on, reasoning stripped from history): turn 2 reused
55 of 99, turn 3 95 of 135 — the hybrid's checkpoints keep prefix reuse working (O6 closed).
The 362.7 t/s Andrei saw was ctx1d's dedicated ub-4096 cache-0 prefill server on a 26.8k-token document: a 2.2k prompt caps any
ubatch at 2.2k, so the per-ubatch expert upload (~2.5 s) cannot be spread further; pmux3 runs ~10k tokens at ubp 2048 / 4096.

### 2026-09-23 05:57 — pfprof1: dp4a MMQ doubles prefill; FlashAttention is now the co-bottleneck
9,279-token prompt, prefill mode ubp 4096, cache 22, MTP head, dp4a MMQ build (llama.cpp-ov), nsys with graphs off: **348.4 t/s**
prefill (pmux3 MMA build: 167.1) and 43.0 t/s decode after it. Window 29.4 s: kernels 25.0 s, H2D 3.4 s (43.6 GB) of which only
0.2 s hidden under compute, GPU idle 1.2 s. By class: MMQ 9.9 s (39.7%), **FlashAttention 9.4 s (37.5%)**, delta-net 2.5 s,
mmvq 1.3 s (the 128 decode tokens), elementwise 1.0 s. MMQ now runs ~5.6 TOPS (~30% of dp4a peak); FA at ~1 TFLOPS on the MMA
kernel = the same tensor-core-on-TU116 defect as MMQ had -> tu1's GGML_CUDA_FA_NO_MMA arm is the next lever, then upload overlap
(3.2 s serialized = ~11% of the window, now worth building).

### 2026-09-23 06:01 — mmqdp1: dp4a MMQ is a free 7.7x on long-prompt prefill; pmux4 floor is exactly 0
Unit: test-backend-ops MUL_MAT_ID 929/929, MUL_MAT 1297/1297 (CUDA vs CPU). Long prompt (9,279 tok, cache 22, MTP): ubp 2048
324.6 t/s (MMA 152.3), ubp 4096 349.6 (MMA 167.1). Decode (specbench, served vs dp4a build + prefill mode ubp 2048, two
interleaved rounds): code -1.0% (spread 1.68), reason +0.9% (1.16), edit -0.7% (0.31), long +1.4% (0.95) = unchanged; prefill on
the served config -> dp4a+prefill mode: long 48.5 -> 372.4 t/s (7.7x), edit 48.2 -> 141.8, reason 38.6 -> 73.5, code 34.8 -> 55.6;
long-prompt wall t/s 5.8 -> 23.8 (4.1x). pmux4 floor row (ub 128 vs ub 128): KLD 0.000000, same top 100% — the run is
deterministic, so ub 2048's 0.0055 / 96.9% is a real ubatch effect; pmux5 judges it against the Q6 truth.

### 2026-09-23 06:18 — tu1: FA without tensor cores +29% prefill; 0014 decode +3.4% (contaminated run), exact
Unit: FLASH_ATTN_EXT 3979/3979 under GGML_CUDA_FA_NO_MMA=1, MUL_MAT_ID 929/929 under EXPERT_COLS. Long prompt (9,279 tok, ubp
4096): old sched path 352.6, readback only 350.1, 0013+0014 default 347.5 (prefill: no effect, noise), **FA tile 448.9 (+29%)**,
FA tile + expert cols 456.1. ub 128 without prefill mode: 110.2 -> 119.9 with expert cols (+8.8%). Decode (2 rounds, every arm
flagged FOREIGN CPU kcompactd0): 0014 vs sync-before-copy +3.4% mean (code +3.9, reason +6.2, edit +3.4, long -0.4; sync spread
up to 5.5); FA tile -3.9% decode (reason -6.5) -> 0015 GGML_CUDA_FA_TILE_MIN_BATCH (tile only for big batches). Identity
0013+0014 vs old path under LLAMA_MOE_CACHE_SYNC=1: IDENTICAL x4. tu2 re-runs decode clean (3 rounds, memory compacted per arm).
LFU question (Andrei): lfu_variants.py — pure LFU 30-33% hit, global LFU 45-54%, TinyLFU-gated 46-54% (frozen hot set) vs
gate-lru 58.8%; O11 stays dead.

### 2026-09-23 06:58 — pmux5: prefill mode, dp4a MMQ and FA tile are NON-INFERIOR vs the Q6 truth
K2 vs Q6_K_P reference (wikitext, c 2048, 12 chunks, cache off). Mean KLD / same top: ub128 MMA (today) 0.200676 / 80.78% |
ub2048 MMA (prefill mode) 0.200063 / 80.89 | ub2048 dp4a 0.199973 / 80.85 | ub2048 dp4a + FA tile 0.199801 / 80.91 | ub128 dp4a
0.200507 / 80.69 | ub128 dp4a + MoE expert cols **bit-identical** to ub128 dp4a (0016 is exact; tu1 +8.8% at ub128). Gate
(+-2% KLD, +-0.5 pt same top vs ub128 MMA): all PASS, every arm within 0.5% KLD. Left before promotion: tu2 (clean decode for
0018, 0019 split FA). Promotion goes to a NEW build dir so queued accuracy jobs (race1 pinned; hq1/qx3/lq keep build75) stay
comparable with their earlier rows.

### 2026-09-23 07:11 — tu2 + ent1: split FA passes, 0018 parked, lossless compression dead
tu2 (3 interleaved rounds, memory compacted per arm): 0019 GGML_CUDA_FA_TILE_MIN_BATCH=32 keeps FA-tile prefill (446.9 t/s on the
9,279-token prompt) with decode equal to default (-0.4%, inside spread) -> PASS. 0018 (no host sync before stream-ordered copies):
-2.0% vs the sync path, inside noise, spreads 2x wider (edit 7.97 vs 3.69) -> NOT proven; made opt-in (GGML_SCHED_NO_COPY_SYNC=1,
commit reworded on tu116-served 1c54372). ent1 (K2 experts, 8 layers): entropy-coder ideal 90.1% (gate Q2_K) / 90.4% (up) /
93.8% (down Q3_K) of stored size, lzma 93.2 / 93.7 / 98.4% -> no kind below 0.85: lossless expert compression DEAD. Next: promo1
builds tu116-served in a new worktree (/ai/src/llama.cpp-v2) with -DGGML_CUDA_MMQ_NO_MMA=ON, unit + identity vs ov + headline.

### 2026-09-23 07:47 — PROMOTED: STABLE = /ai/src/llama.cpp-v2 (tu116-served 1c54372: prefill mode + dp4a MMQ + FA tile >= 32)
promo1: fresh worktree + build with -DGGML_CUDA_MMQ_NO_MMA=ON (9 min); unit MUL_MAT_ID 929 / MUL_MAT 1297 / FLASH_ATTN_EXT 3979
pass; IDENTICAL x4 vs the ov build under LLAMA_MOE_CACHE_SYNC=1. Old served config (build75, -ub 128 -b 256) -> STABLE serving
config (-b 2048 -ubp 2048, FA_TILE_MIN_BATCH=32), 2 interleaved rounds: decode code +0.9 / reason +0.2 / edit -0.8 / long -6.1%
(long: 49.07 49.14 -> 45.27 46.95, both rounds below — open item: cache state after a prefill-mode switch vs a token-by-token
warm cache; the 9,279-token run shows the opposite, 47.3 -> 49.8), prefill 1.55x / 1.91x / 2.99x / 8.13x. 9,279-token prompt:
47.3 -> 403.2 t/s, wall 198.8 -> 25.6 s. LEGACY build75 stays for race1 (pinned) and the accuracy series. opt2 (confidence-gated
MTP drafting): no arm beats the n-max-2 base beyond its spread (best n3 p0.6 +2.6% vs spread 3.05; n4 -12..-14%) -> dead.

### 2026-09-23 07:55 — prof18: 0018's mechanism is real (-18 us host overhead per MoE layer) -> default-on in STABLE (0020)
nsys, STABLE build, serving config, 300-token decode, 2 captures per arm. Host overhead per layer (graph launch + stream syncs +
copy calls + host gaps): sync 89.5 / 79.4 us, no-sync 62.7 / 71.0 us (mean -17.6 us, larger than the a/b gaps of 8-10 us); stream
syncs per graph launch 5.21 / 5.23 -> 4.58 / 4.63; device phase 414 / 420 vs 418 / 404 us (unchanged); total time in syncs
unchanged (the remaining syncs wait longer), round 975 -> 925 us mean but the CPU expert phase alone swings +-25 us between
captures — the gain (~1.8%) sits below specbench's noise, which is why tu1 / tu2 read +3.4 / -2.0%. Exact by construction ->
promotion rule met at the mechanism level: 0020 flips it on (GGML_SCHED_COPY_SYNC=1 = old), promo2 checks identity before race1.

### 2026-09-23 08:11 — dec1: promo1's long-decode -6% does NOT reproduce in-build; short prompts -3..-4.5% under prefill mode
One build (ov = STABLE code + env-gated warm-tail), 3 interleaved rounds. Decode P (prefill mode) / O (-b 256 path) / T (prefill
mode + LLAMA_MOE_WARM_TAIL=128): code 54.40 / 56.94 / 56.71, reason 56.00 / 58.16 / 57.13, edit 58.71 / 60.64 / 58.44, long 2.2k
47.16 / 45.81 / 45.76 (spreads 0.3-4.3). 9,279-token prompt (1 run): prefill 403.5 / 121.9 / 402.6 t/s, decode 46.82 / 49.06 /
50.20. Reading: prefill mode is not slower on long prompts (+2.9% vs O); short prompts trail O in all 3 by 3-4.5% (inside spreads);
T recovers code / partly reason and has the best 9.3k decode (1 run). NOT PROVEN -> dec2 (P vs T, 5 rounds + 9.3k x3,
pre-registered rule in the script). Mechanism noted: in prefill mode the warm-up counts span the whole prompt (step() is skipped
while the cache is suspended); on the -b 256 path they span the last batch.

### 2026-09-23 08:36 — dec2: NOT PROVEN under the pre-registered rule, and contaminated (dec3 re-runs it clean)
P (prefill mode) vs T (prefill mode + LLAMA_MOE_WARM_TAIL=128), 5 interleaved specbench rounds, ov build. Mean decode (spread):
code P 56.28 (3.37) / T 55.72 (3.80) -1.0% | reason 56.94 (3.04) / 56.08 (3.50) -1.5% | edit 60.63 (3.05) / 58.18 (2.98) -4.0% |
long 47.24 (0.95) / 45.79 (4.54) -3.1% -> 4-prompt mean P 55.27, T 53.94 (-2.4%, P spread 2.60) -> rule's first leg fails.
9,279-token prompt x3: decode P 47.32 / 49.65 / 49.01, T 50.10 / 49.96 / 50.11 -> T >= P in 3/3 (mean +2.9%); prefill 401.5-403.1 both.
Contamination: kcompactd0 averaged 54-75% of a core in 4 of 10 specbench arms (T_b, T_c, T_d, P_d) -> 3 of 4 flags on T, the
short-prompt comparison is biased against T. Not evidence either way; warm-tail stays OFF. Root cause and fix: c52fa30 (unmovable CUDA
pinned host memory + proactive compaction + watermark boost; preflight now sets both sysctls to 0). dec3 = same job, clean box.

### 2026-09-23 08:42 — ovl2: the upload-overlap divergence is the LAYOUT, not the second stream (ovl3 tests stale expert bytes)
t2 at dfbdf14, specbench identity (LLAMA_MOE_CACHE_SYNC=1, serving config). off_a == off_b and on_a == on_b (IDENTICAL x4 each:
deterministic, no race). off vs on, off vs plan (GGML_SCHED_MOE_PREFETCH=2: hoisted input copies, uploads in stream order, no second
stream) and off vs on_block (CUDA_LAUNCH_BLOCKING=1) all DIVERGE at the same chars (code 221, long 96, reason 78; edit IDENTICAL).
=> moving the device copy of the host expert tensor changes the output by itself. Hypothesis: below 8 routed tokens/expert only the used
experts are uploaded; the rest of the copy is whatever the region held (layout-dependent); something reads it. Diag switch
GGML_SCHED_MOE_DIAG_FILL (1ea8e04, upload-overlap) + ovl3 (off vs plan with the copy zero-filled; fill 0x00 vs 0xFF). Overlap stays OFF.

### 2026-09-23 08:55 — dec3 (clean box): warm-tail NOT PROVEN -> dead; the kcompactd fix alone lifted P
Same arms as dec2, compaction off (preflight compact=0/0), no FOREIGN CPU flag in any arm. Mean decode (spread): code P 58.08 (1.91)
/ T 58.04 (2.39) -0.1% | reason 58.68 (5.51) / 58.58 (2.74) -0.2% | edit 61.79 (1.04) / 60.90 (2.43) -1.4% | long 47.64 (2.53) /
47.19 (1.94) -0.9% -> MEAN P 56.55, T 56.18 (-0.7%, P spread 2.75). 9,279-token decode P 50.17 / 49.99 / 50.14, T 50.10 / 49.07 / 50.17
-> T >= P 1/3. NOT PROVEN: LLAMA_MOE_WARM_TAIL stays out of the series (research/patches/experimental/warm-tail.patch parked).
dec2's T wins on the 9.3k decode were noise in a contaminated job. Cross-job (same build, same arms): P's 4-prompt mean 55.27 (dec2,
kcompactd) -> 56.55 (dec3, clean) +2.3%, P's 9.3k decode 48.66 -> 50.10 +3.0% (inferred: the compaction fix, not a code change).
dec1's "prefill mode costs short prompts 3-4.5%" is suspect for the same reason (its P_b arm was flagged).

### 2026-09-23 09:00 — ovl3: stale expert bytes are NOT the overlap divergence; the baseline never reads unselected experts
t2 @ 1ea8e04 (GGML_SCHED_MOE_DIAG_FILL). off_plain == ovl2_off_a (diag inert). off_z vs plan_z (copy zero-filled) still DIVERGES at the
same chars -> not stale bytes. off_z vs off_n (fill 0x00 vs 0xFF) IDENTICAL, off_plain vs off_z IDENTICAL -> no kernel reads experts
the router did not select (good news for STABLE/LEGACY). Tensor dump (llama-eval-callback, reason prompt, 55 tokens, one ubatch):
off vs plan IDENTICAL over 2,747 tensors -> the prompt ubatch is not where it diverges. (eval-callback segfaults at exit after the
dump; the dumps are complete, the off_a vs off_b last-block mismatch is the truncated final print.)

### 2026-09-23 09:08 — ovl4: the divergence is in the MAIN model, not the MTP draft
No draft (-md dropped): off vs off IDENTICAL, off vs plan DIVERGES (code at char 15, reason 205, long 318; edit IDENTICAL).
The draft-on-GPU arm was void (-otd exps=CUDA0 left the draft's experts in CUDA_Host 272.81 MiB). ovl5 (t2 @ 4431c75: plan log +
GGML_SCHED_MOE_PREFETCH_MIN_IDS) logs which graphs get planned splits and bisects by batch size. race1 held behind it (Andrei:
optimization work may push race1).

### 2026-09-23 09:30 — scoreboard: prefill + STABLE vs LEGACY vs TEST (2109a89)
longpf runs now land in the ledger (kind longpf; 34 backfilled), every run records its GGML_*/LLAMA_* env, SCOREBOARD.md has prefill
2.2k / 9.3k columns and a per-build section (best + median). K2, STABLE vs LEGACY, medians, identity runs excluded: decode code -0.5%,
reason +2.2%, edit +1.3%, long -0.4%; prefill 2.2k 8.13x, 9.3k 8.52x.

### 2026-09-23 09:44 — att1: FA tile is 1.9x faster at 9.3k KV but the time becomes GPU idle; the critical path is the CPU miss phase
nsys decode after the 9,279-token prompt (MTP n=2, graphs off), STABLE: MMA flash_attn_ext_f16 630 us/launch (2.98 ms/token) vs
NO_MMA flash_attn_tile 334 us/launch (1.63 ms/token) — but GPU idle 5.93 -> 7.04 ms/token, total 22.04 -> 22.08 ms/token. Timing
(graphs on, 2 rounds): 9.3k decode served 47.54 / 50.43, NO_MMA 51.08 / 50.06 (+3.2%, spread 2.89); 2.3k served 46.16 / 50.97,
NO_MMA 51.97 / 54.52 (+9.6%, spread 4.81). No-draft decode after 9.3k: 39.81 t/s. The GPU already waits on the CPU experts
(DDR4 miss phase, cpu1bench: memory bound at 6 threads) -> GPU-only speedups turn into idle. Candidate patch GGML_CUDA_FA_MMA_MAX_KV
(fa-kv 3ea44a4, experimental/fa-mma-max-kv.patch) stays a secondary item. H2D placement (h2dov.py): 40% of the cache fills
(~10 MB/token) land inside the CPU-bound idle windows (DDR4 contention). Ledger + ranked moves: research/design-harmony-ledger.md.

### 2026-09-23 09:43 — mt1: multi-turn prefix reuse works on STABLE (no re-prefill); checkpoints cost RAM + ~0.2 s TTFT
3 turns (9,279-token doc, then two short follow-ups), serving config. default: turn 2 processed 26 tok / reused 9,406, TTFT 0.51 s;
turn 3 29 / 9,493, 0.53 s. --ctx-checkpoints 0: same reuse (26 / 9,406), TTFT 0.33 / 0.37 s — an appending conversation continues
from the slot's own state; checkpoints only matter when a prompt diverges (edit / regenerate). Each checkpoint 81.6 MiB of host RAM
(8 created = ~650 MiB on a box with ~3 GiB free). Decode per turn 48.0 -> 41.8 -> 39.2 (default), 48.7 -> 43.6 -> 41.1 (nocp): MTP
acceptance 0.88 -> 0.67 -> 0.63 on list-style answers, content not overhead. Serving note: cap -ctxcp (2-4) when STABLE is deployed.

### 2026-09-23 09:45 — ovl6-ovl9: the overlap divergence is NOT CUDA graphs, NOT asynchrony; the cache warm-up observer differs
ovl6: GGML_CUDA_DISABLE_GRAPHS=1 both arms still DIVERGE. ovl7: eval-callback's earlier dumps had 0 planned pairs (void); planning
one target type alone keeps code/reason IDENTICAL, only the full chain breaks them. ovl8: GGML_SCHED_DIAG_SYNC=1 (drain after every
split) still DIVERGES -> layout, not timing; dump with 240 planned pairs IDENTICAL for the prompt ubatch. ovl9: planned prompt ubatch
+ decode-chain ubatch both IDENTICAL in callback mode (no CUDA op fusion there). Server: first prompt's warm-up 1016 (off) vs 1017
(plan) uploads, hit rate 51.2 vs 52.4% -> different cached experts -> different GPU/CPU expert split -> different rounding.
ovl10: warm-up off + an id-checksum trace of the observer.

### 2026-09-23 09:50 — pg1: PRE-GATING WORKS here — layer L's MoE input predicts layer L+1's experts at 90.9% (top-16)
ov + pregate-diag (LLAMA_MOE_PREGATE_DIAG=1), serving config, MTP n=2, decode batches only, 256 steps: recall of the actual top-8 by
layer L+1's router applied to layer L's MoE input: top-8 74.8% (L1-9 67.6, 10-19 77.4, 20-29 79.3, 30-39 74.1), top-16 90.9%;
two layers ahead top-16 83.2%. P2's TOKEN-based predictors (68% / 49%) were the wrong predictor, not a dead idea. This is the
enabling fact for the harmony design (research/design-harmony-ledger.md move 1): upload predicted misses of L+1 during layer L,
compute them on the idle GPU, take the bytes off the CPU/DDR4 critical path. Next: recall restricted to MISSES (experts not cached).

### 2026-09-23 09:48 — ovl10: not the warm-up; the prompt pass's routing itself differs from layer 8 on (server only)
warm-up off (--moe-expert-cache-warm 0): off vs plan still DIVERGES. Observer id-checksum trace: layers 0-7 identical, 8+ differ in
the first prompt pass (n = 304 ids), no out-of-range ids. Callback mode (no CUDA fusion) was identical -> ovl11: fusion disabled.

### 2026-09-23 09:54 — ovl11: the upload-overlap divergence is a CUDA OP FUSION that is layout-sensitive
GGML_CUDA_DISABLE_FUSION=1 in both arms: off vs plan IDENTICAL x4 (with fusion: DIVERGES, ovl2-ovl10). Prime suspect: the MoE
weighted-reduction fusion (the only fusion that registers alloc deps in graph_optimize: experts/weights kept alive until the fused
node), interacting with the hoisted input copies in the down split. Parked behind the prefetch work; next step is a per-fusion
bisection (the overlap is worth ~15% of prefill once exact).

### 2026-09-23 09:57 — pg2: pre-gating on MISSES — top-8 covers 70% of them at 68% useful uploads
Per decode pass and layer (union over the pass's tokens): 9.94 actual misses. Top-8 prediction from layer L: covers 70.3% of L+1's
misses with 10.24 uploads (68.2% useful); top-16: 89.5% with 23.24 uploads (38.3% useful). Expert ~1.14 MB -> ~92 us over PCIe vs
~42 us on the CPU (DDR4) -> in parallel the balance is ~2-3 useful uploads per layer (~-24% on the miss phase before DDR4
contention). Built: branch pregate-prefetch bb65a4d (LLAMA_MOE_PREFETCH=N budget, _TOPK width, same-step publish on the compute
stream, pinned table stage); pf1 = inert check + budget sweep + 9.3k decode.

### 2026-09-23 10:11 — pf1: pre-gated prefetch is INERT when off, raises the hit rate, and is SLOWER (issue cost or DDR4)
t2 @ bb65a4d. A: prefetch unset vs the STABLE binary, identity mode: IDENTICAL x4. B (specbench, 2 interleaved rounds, top-8):
hit rate 48.1-48.5% (off) -> 59.6 / 63.3 / 66.5-67.1% (budget 2 / 3 / 4), prefetch 21 / 31 / 41 MiB/step; mean decode
off 55.85 (spread 0.64) -> b2 54.14 (-3.1%), b3 51.42 (-7.9%), b4 49.01 (-12.2%), monotone. C (9,279-token prompt, cache 22):
decode off 48.77 / 50.07 vs b3 45.33 / 45.30 (-8%). Two candidate causes: (1) the head op issued ~3 cudaMemcpyAsync per expert
on CPU thread 0 under the cache lock (critical path) -> f43cd33 moves issue to a helper thread joined at the split tail (pf2 A/B
async vs inline); (2) DDR4: a prefetch MOVES an expert read from the CPU to the DMA engine inside the same critical window
(+ ~32% useless uploads) — only a win if the CPU phase is not bus-bound. Review (pf-review agent): 1 critical / 2 high / 3
medium closed in 8bdf719 (MTP-draft backend takeover, pageable sources, multi-GPU asserts, ...); 4 lows left for the next commit.

### 2026-09-23 10:35 — pf2/pf3: same-step prefetch is dead on this box (PCIe too slow, DMA inside the DDR4 window); thr1: C dead
pf2 (f43cd33): helper-thread issue cut the head op 19.5 -> 2.3 us/layer and changed nothing (b4 async -13.3%, inline -13.1%;
b2 -5.3%; off spread 1.60). pf3 nsys (9.3k decode): b4 H2D 28 -> 75 MB/token (6.19 ms/token busy), 94% inside the CPU-bound idle
gaps (off 43%), idle gaps 6.32 -> 8.36 ms/token (median gap 39 -> 157 us): budget 4 = ~5 MB/layer/pass = ~410 us at 12.4 GB/s =
the whole host window, on the compute stream and reading the same DDR4. => lead A (window mode 880f74b: L+2 on a copy stream,
issued at L's tail = the DDR4-idle window, waited at L+1's tail) in pf4.
thr1 (lead C): -t 6 57.11 (spread 0.71), -t 8 57.21 (+0.2%), -t 12 52.23 (-8.5%) -> no winner; SMT competes, -t 6 stays.
LM head facts (pf3 trace): output.weight q5_K ~349 MB read at ~196 GB/s (roofline) 1.29 ms/token on the target + 0.81 ms/token
on the MTP draft head (grid 124160) -> lead B (FR-Spec draft vocab) = the only lever there.
Spectrum pass (leads A-E, ranked by effect on the CRITICAL path, research/design-harmony-ledger.md): A window prefetch, B draft
vocab reduction, C threads (dead), D A+FA tile harmony, E overlap re-framed (address-dependent fusion verdicts -> KLD class).

### 2026-09-23 10:48 — pf4 (leads A + D): window prefetch NOT PROVEN; FA tile's long-context +3% is the part of D that survives
t2 @ 880f74b (window mode: L+2 on a copy stream, issued at L's tail, waited at L+1's tail). Specbench, 2 rounds, mean decode:
off 56.33 (spread 1.09) | w2 55.77 (-1.0%) | w4 54.72 (-2.9%) | NO_MMA 56.19 (-0.2%) | w4+NO_MMA 55.73 (-1.1%) -> A NOT PROVEN,
D NOT PROVEN (short). 9,279-token prompt decode: off 49.27 / 49.77 | w4 47.65 / 48.54 | w4+NO_MMA 50.82 / 51.14 (+3.0%) = the
FA tile kernel's own long-context gain (att1 +3.2%). nsys w4 (9.3k decode): the mechanism half-worked — H2D inside the CPU-bound
gaps 94% (step) -> 36% (off 43%), idle 6.32 -> 5.97 ms/token — but GPU busy 14.92 -> 15.98 ms/token (+1.06: norm/rope/elementwise
2.6 -> 3.7 us per launch, more cache-chain work): the moved bytes cost the device about what they save the host. Hit rate 48 ->
57 / 63%. Verdict: on this box (PCIe = 1/3 of DDR4, DMA shares VRAM with the kernels) moving expert bytes to the GPU does not pay;
the pre-gating predictor stays an asset, prefetch parked (branch pregate-prefetch, review fixes 0b7a2ba). Follow-up for D:
fakv1 = GGML_CUDA_FA_MMA_MAX_KV threshold sweep (tile only above N KV), queued after fr1.
Leads status: A not proven (parked) | B fr1 queued | C dead (thr1) | D -> fakv1 | E ovl12 queued (fusion verdicts, KLD, speed).
