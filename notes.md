# local-ai notes

Repo: **https://github.com/Andrei-Dr/local-ai** (private; `~/dev/local-ai`). Commit + push after every recorded result: notes, `bench/` (harness, ledger, logs mirrored from the box via `rsync root@i5.local:/ai/bench/{*.log,ledger.jsonl} bench/box/`), `research/`. The llama.cpp work itself lives on the box (`/ai/src/llama.cpp` branch `i5-tuning`, worktree `/ai/src/llama.cpp-moecache` branch `moe-cache`); it is exported here as `research/patches/box-series/` (`git format-patch 9a9394a..moe-cache`) — re-export after every box commit. A GitHub fork of llama.cpp would be PUBLIC (forks of public repos cannot be private), so that waits for an explicit go.

## i5 box (`ssh root@i5.local`)

- GPU: GTX 1650 SUPER 4 GB — Turing cc 7.5 but **TU116: no tensor cores**, no bf16/FP8. ~3.45 GB usable. PCIe 3.0 x16 (idles at gen1, gen3 under load). Power cap already at max (100 W).
- CPU: i5-10400F 6c/12t — `avx2 fma f16c bmi2`, **no AVX-VNNI / AVX-512**. RAM 2x8 GB DDR4-2667 dual channel (~38 GB/s theoretical, ~30 real). 2 GB swap.
- Disk: `/` is nvme ext4 (155 G free). **`/opt` and `/mnt/md0` are the same spinning md0 btrfs+zstd — never put models or build trees there.** `/root`, `/tmp` are nvme.
- Layout: **`/ai/{src,models,bench,.venv}`** (nvme). tmux session `ai` (windows: build, dl, bench, mv).
- `/root/bin/docker-cleanup.py` (mirrored at `~/bin/docker-cleanup.py` on the Mac): `-a` now prunes anonymous 64-hex dangling volumes (named volumes always spared), `-y` skips the prompt (`-ay` for cron), refuses without a tty unless `-y`.

### Host settings changed (persisted 2026-09-19 via `/etc/systemd/system/ai-perf-tweaks.service`, oneshot, enabled; remove with `systemctl disable --now ai-perf-tweaks && rm` the unit)

| setting | was | now | why |
|---|---|---|---|
| cpufreq governor (all cores) | `powersave` | `performance` | CPU-resident layers are CPU-bound |
| `/sys/kernel/mm/transparent_hugepage/enabled` | `madvise` | `always` | fewer TLB misses on the ~5 GB of CPU-side weights (pairs with no-mmap) |

Box was hard-reset 2026-09-19 12:09 EEST (see RAM rule in the runbook); tmux session `ai` recreated (windows build, dl, bench).

## Runtime

Not ollama, not vllm. PQ2_0 / PTQ1_0 only load in **PrismML-Eng/llama.cpp, branch `prism`** (not `prism-v6`). vLLM has no MLX backend and no PQ2_0 support; ollama ships stock llama.cpp.

Local branch `i5-tuning` at `/ai/src/llama.cpp` = prism@9a9394a plus:
1. `fix:` BoldingBuilds `0001-qwen35-mtp-hadamard-inverse.patch` — without it `--spec-type draft-mtp` dies with "Hadamard-latent table 'token_embd.weight' is read without the inverse transform". Not upstream yet.
2. `perf:` **AVX2 path for `ggml_vec_dot_pq2_0_q8_0`**. Upstream has only a VNNI path and a scalar `#else`; this CPU has no VNNI so 60% of the model ran scalar. 38.5 -> 5.6 cycles/32 weights (6.9x). `test-quantize-fns` exit 0, `test-backend-ops MUL_MAT type_a=pq2_0` 45/45 CUDA-vs-CPU. Upstream candidate.

Build: `cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`.
Untested: the fork itself suggests `-DCMAKE_CUDA_ARCHITECTURES="61-virtual;80-virtual" -DGGML_CUDA_FORCE_MMQ=ON` for tensor-core-less Turing.

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
| Gemma4-26B-A4B IQ3_M, **cache 16** | 3.9 | 24.3 (22.6 in-run, 41 min sustained) | 100.0 | 87.8 ±5.1 | 44.3 ±5.9 (*) | 38 / 0 | |
| Gemma4-26B-A4B IQ3_M, cache off | 3.9 | 16.7 | 100.0 (50/50) | 87.8 (36/41) | partial (*) | | killed by systemd-oomd at item 145/161 (15:06), resumed 21:05 |
| Bonsai-27B PQ2_0-MTP | 2.13 | 5.15 | owed | | | | ~2.5 h run |
| Gemma4 Q3_K_M / Q2_K_P | 3.4 / ~2.8 | to be measured | owed | | | | |

(*) MMLU-Pro first pass is NOT usable as an absolute score: 34–39 of 70 answers hit the 350-token cap and every cut-off answer scored wrong (accuracy among finished answers: Qwen 27/31, Gemma 31/36). Caps raised to 768/1024/1024; `qual.py` re-runs cut-off rows automatically on the next pass (`/ai/bench/qual3.sh`, to be run with the new quants + Bonsai).

## Research (full reports in `research/`)

- `research/pr-scout-llamacpp-prs.md` — mainline + prism PRs/issues. Key: FFN-only-on-CPU placement (#26622), MTP shared-KV misdetection bug present in our checkout (#27781, `common/speculative.cpp:2129`), #24670 (GTX 1650 SUPER + Qwen3.6-35B-A3B: `draft-mtp` never drafts unless `--spec-draft-p-min 0.0`).
- `research/lit-scout-arxiv-survey.md` — arXiv survey re-ranked against our measurements. Expert cache design: async admission, compute misses on CPU, never stall on a fetch (FreeToken 2608.16157, WiSP 2606.21868); eviction policy barely matters, cache-aware routing (Cache-Prior 2412.00099) is the lever at small caches; do NOT quantize GDN state (DAMP 2608.27513).
- `research/moe-scout-abliterated-moe.md` — MoE shortlist. On the box: Gemma4-26B-A4B IQ3_M (12.39 GB) + `mtp-gemma-4-26B-A4B-it.gguf`, Qwen3.6-35B-A3B IQ2_M (11.66 GB). The QAT-MTP Gemma (16.8 GB) does not fit. Fork supports `gemma4`, `gemma4-assistant`, `qwen35moe`; AVX2 kernels exist for all their quant types.

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
