# Abliterated MoE GGUF shortlist for expert-offloaded llama.cpp

Date: 2026-09-19

## Hardware constraints used

- GPU: NVIDIA GTX 1650 SUPER, 4 GB VRAM (Turing, compute capability 7.5, no bf16, no FP8). ~3.3 GB usable for weights + KV after CUDA context and compute buffers.
- CPU: Intel i5-10400F, 6c/12t, AVX2 only (no AVX-512).
- RAM: 15 GB total (2x8 GB DDR4-2667 dual channel, ~38 GB/s effective), ~12 GB realistically usable for model weights. 2 GB swap. NVMe available (mmap spill possible but slow).
- Derived limits: total GGUF <= ~14 GB, ideally <= 12 GB. Non-expert portion + KV at 8K context must fit ~3.3 GB VRAM. Active parameters per token <= ~4B.
- Target launch shape: `-ngl 999 -ot "exps=CPU"`, so that an LRU/LFU expert residency scheme becomes meaningful later.

## Method

Read-only, web/API only. During the research pass nothing was downloaded in full and nothing was run on any machine (see the addendum for what happened afterwards).

- File sizes are exact bytes from `https://huggingface.co/api/models/<repo>?blobs=true`.
- The expert / non-expert split is **not** estimated from config. Each GGUF header was range-fetched over HTTP and tensor offsets were summed: `*_exps` = routed experts (RAM), `token_embd` (RAM), everything else (VRAM). Scanner: `research/scripts/ggufscan/scan.py`.
- Base architectures from `https://huggingface.co/<base>/raw/main/config.json`.
- Mainline architecture support from `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp` (latest release tag v0.4.1, published 2026-09-14).
- Decode ceiling = 38 GB/s / (expert_bytes x experts_used / expert_count).
- GB are decimal throughout.

## Single recommendation

- Repo: <https://huggingface.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced> (the **non-QAT** repo).
- File: `Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf`, 12,392,563,552 B (12.39 GB).
- Fallback in the same repo: `...-Q2_K_P.gguf`, 10.70 GB, if RAM is tight or the AVX2 i-quant kernels turn out compute-bound.
- Drafter: `mtp-gemma-4-26B-A4B-it.gguf`, 251,937,728 B, from <https://huggingface.co/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP>.
- Launch shape:

  ```bash
  llama-server \
    -m Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf \
    -md mtp-gemma-4-26B-A4B-it.gguf --spec-type draft-mtp \
    -ngl 999 -ot "exps=CPU" -fa on --jinja -c 8192
  ```

- Optional: pin the experts of 3 layers to GPU (0.364 GB each) to cut RAM from 11.53 GB to ~10.4 GB. This spends the VRAM a future expert cache would use.
- The user's original candidate (the QAT-MTP repo's Q4_K_M) does not fit; see Rejected.

## Ranked shortlist

All rows header-scanned unless noted. Sizes in GB.

| # | Repo / file | Total | Experts (RAM) | token_embd (RAM) | Non-expert (VRAM) | KV @8K f16 (inferred) | VRAM need | RAM need | Active expert bytes/token | Ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced, IQ3_M | 12.39 | 10.92 | 0.61 | 0.87 + 0.61 tied-output copy (copy is inferred) | ~0.48 | ~1.96 | 11.53 | 0.682 | 55.7 t/s |
| 1b | same repo, Q2_K_P (gate_up Q2_K, down Q4_0) | 10.70 | 9.28 | 0.61 | 0.81 + 0.61 | ~0.48 | ~1.90 | 9.89 | 0.580 | 65.5 t/s |
| 2 | HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive, IQ2_M | 11.66 | 10.41 | 0.22 | 1.03 (incl. 0.35 output.weight) | ~0.23 | ~1.26 | 10.63 | 0.325 | 116.9 t/s |
| 3 | mradermacher/gpt-oss-20b-heretic-ara-v3-i1-GGUF, MXFP4_MOE. Header scanned on HauhauCS/GPT-OSS-20B-Uncensored-HauhauCS-Aggressive MXFP4 (also 12.11), **not** on the heretic file itself | 12.11 | 10.18 | 0.62 | 1.32 (Q8_0, incl. 0.62 output) | ~0.22 | ~1.54 | 10.80 | 1.272 | 29.9 t/s |
| 4 | BoldingBuilds/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Uncensored-ShimQuant-GGUF, `...v2-ShimQuant-IQ3_XXS` | 12.63 | 10.52 | 0.37 | 1.75 (Q6_K / Q8_0) | ~0.12 | ~1.87 | 10.89 | 0.493 | 77.1 t/s |
| 5 | mradermacher/Huihui-LFM2-24B-A2B-abliterated-i1-GGUF, Q3_K_M | 11.35 | 10.88 | 0.11 | 0.37 + 0.11 tied copy | ~0.17 | ~0.65 | 10.99 | 0.680 | 55.9 t/s |

Fit verdict: all five fit VRAM (3.3 GB budget) with 1.3-2.6 GB margin, and all fit the ~12 GB RAM target. Gemma IQ3_M is the tightest on RAM (11.53 GB). Active parameters: Gemma 3.8B, Qwen ~3B, gpt-oss 3.6B, Nemotron ~3B, LFM2 ~2B, all within the 4B limit.

Expert tensor types observed:

| File | Expert tensors |
|---|---|
| Gemma IQ3_M | gate_up IQ3_S x30; down IQ4_NL x27 + Q5_0 x3; per-layer F32 down scale |
| Gemma Q2_K_P | gate_up Q2_K x30; down Q4_0 x30 |
| Gemma QAT Q4_K_M (rejected) | gate_up Q4_K x30; down Q8_0 x14 + Q5_0 x16 |
| Qwen3.6 IQ2_M | gate/up IQ2_S x40; down IQ2_S x37 + IQ3_S x3 |
| gpt-oss MXFP4 | gate/up/down MXFP4 x24, F32 biases |
| Nemotron ShimQuant | up IQ2_XXS x23, down IQ2_S x23, one layer Q8_0 (no gate tensor; non-gated experts) |
| LFM2-24B Q3_K_M | gate/up Q3_K x38; down Q4_K x38 |

Ceilings are bandwidth-only. Expect well under them (inferred): IQ2_S / IQ3_S kernels are compute-heavy on 6 AVX2 cores, and there are 30-40 per-layer PCIe syncs. K/legacy-quant experts (Gemma Q2_K_P, LFM2 Q3_K_M, gpt-oss MXFP4) are the safer choice if CPU compute becomes the limiter.

## KV arithmetic (inferred from config / GGUF metadata)

- **Gemma4-26B-A4B**: 25 sliding layers x 8 KV heads x 256 x 2 (K,V) x 2 B = 8,192 B/token/layer over a ~1,536-token window (1,024 + ubatch) = ~315 MB. 5 global layers x 2 KV heads x 512 x 2 x 2 B = 4,096 B/token/layer x 8,192 tokens = ~168 MB. Total ~0.48 GB.
- **Qwen3.6-35B-A3B**: 10 full-attention layers x 2 KV heads x 256 x 2 x 2 B = 20.5 KB/token, ~168 MB at 8K. Plus 30 linear-attention layers x (32 value heads x 128 x 128 x 4 B) = ~63 MB state. Total ~0.23 GB.
- **gpt-oss-20b**: 12 full layers x 8 x 64 x 2 x 2 B = 24.6 KB/token, ~201 MB. Plus ~16 MB for the 12 sliding layers (128-token window). Total ~0.22 GB.
- **Nemotron-3.5-Lightning**: 7 attention layers x 2 x 128 x 2 x 2 B = ~7 KB/token, ~59 MB. Plus ~61 MB Mamba state. Total ~0.12 GB.
- **LFM2-24B-A2B**: 10 attention layers x 8 x 64 x 2 x 2 B = ~20 KB/token, ~168 MB.
- Compute-buffer note: Gemma's 262,144-token vocab makes the logits buffer 262,144 x 4 B x ubatch. At `-ub 512` that is ~537 MB; `-ub 256` halves it. This is inside the "after compute buffers" assumption but is the first knob to turn if VRAM runs short.

## Per-pick reasoning

### 1. Gemma4-26B-A4B HauhauCS Balanced

Created 2026-05-14, 74,164 downloads, 277 likes. Author account dates from 2024-10-07 with 9,199 followers and 27 models. Architecture `gemma4` is in mainline. 128 experts, top-8, 30 layers, expert FFN 704, plus a dense 2112 MLP per layer that stays on GPU, so only ~1.43B of the 3.8B active parameters cross DDR4. At ~3.9 bpw it is the highest quality-per-byte that fits. Per-expert size is 2.84 MB, so ~1.3 GB of spare VRAM would cache ~420 of 3,840 experts (11%).

MTP drafter verified: architecture `gemma4-assistant`, 4 blocks, Q4_0, 252 MB, Unsloth-sourced per the card. `draft-mtp` verified in mainline `common/speculative.cpp`.

Red flags: abliteration method undisclosed. "0/465 refusals" and "lossless" are the author's own claims. No KL or capability evals published.

Open-method alternative on the same base: mradermacher/gemma-4-26B-A4B-it-ultra-uncensored-heretic-i1-GGUF (Heretic; Q3_K_S 12.22 GB, experts 10.82 GB, 186,564 downloads). Its upstream card yielded no eval numbers.

### 2. Qwen3.6-35B-A3B HauhauCS Aggressive, IQ2_M

1,049,986 downloads, 3,713 likes. Architecture `qwen35moe` in mainline. 256 experts top-8 x 40 layers = 10,240 experts at ~1.02 MB each: the finest granularity and lowest bytes/token here, so it is the best substrate for LRU/LFU residency research.

The cost is 2.69 bpw, a real quality hit, and nothing at IQ3 or above fits: mradermacher Qwen3.6-35B-A3B-abliterated-v2 IQ3_XXS is 13.62 GB (experts 12.33), HauhauCS Q2_K_P is 14.98 GB, IQ3_M is 15.44 GB.

Base config has `mtp_num_hidden_layers: 1`, but this GGUF carries zero nextn/MTP tensor bytes (verified), so there is no built-in draft head in this file. Same 0/465 self-claim, no evals.

Same-architecture siblings: cognitivers/Ornith-1.5-35B-A3B-Abliterated-12GB-GGUF IQ2_M 12.38 GB (experts 10.99; 972 downloads), orcarouter/Nex-N2.5-mini-Uncensored-GGUF Q2_K 12.94 GB.

### 3. gpt-oss-20b heretic

The only candidate at native precision (experts are MXFP4 as released, no quant loss), and it has transparent author evals. The p-e-w/gpt-oss-20b-heretic-ara-v4 card reports refusals 9/100 vs 98/100 for the original and PIQA acc_norm 0.7742 vs 0.7731 (author's own). Method: Heretic with Arbitrary-Rank Ablation and row-norm preservation.

Costs: 32 coarse experts at 13.3 MB each, top-4, so 1.27 GB/token and a 30 t/s ceiling, plus the worst residency granularity of the five. August-2025 model. No MTP. GGUFs confirmed for ara-v3 (mradermacher); v4 GGUF availability not checked. HauhauCS also ships GPT-OSS-20B MXFP4 Aggressive / Balanced at 12.11 GB.

### 4. Nemotron-3.5-Lightning ShimQuant

Best-documented abliteration by a wide margin. All numbers are the author's own, but artifacts are in the repo (expert list, routing diff, 1,210 held-out prompts, judged scores, verify script):

- Judged harmful refusal 77.9% -> 6.8% over 960 prompts; over-refusal 0/250 on XSTest-safe.
- HumanEval pass@1 0.9573 -> 0.9512 under `--reasoning-budget 2000` (McNemar p = 1.000, n = 164). No MMLU. Single-turn English only.
- Method: projects the refusal direction out of 384 of 3,072 route-selected experts plus non-expert writers, then a read-side pass on routers and up-projections.

"ShimQuant" is a llama.cpp patch at <https://github.com/JoshBolding/shimquant> (MIT, created 2026-08-28, last push 2026-08-30, 1 star, 0 forks, pinned to llama.cpp `e70802a`). It zero-pads tensors to multiples of 256 because n_embd 2688 and expert widths 1856 / 3712 block k-quants and i-quants. The file **will not load** in stock llama.cpp, LM Studio, or Ollama (explicit `check_tensor_dims` shape error, per the card).

Corroboration: every stock quant in mradermacher/Darkstar-Nemotron-3.5-Lightning-30B-A3B-Abliterated-BF16-i1-GGUF is >= 18.66 GB, even "Q2_K".

File mix: Q6_K base, IQ2_XXS up / IQ2_S down experts, 23 MoE layers, Q8_0 pinned on blk.52, nextn=1 included. The card reports MTP measured 28% slower when fully VRAM-resident.

Red flags: 8-follower account, 678 downloads, dependency on an unmaintained fork pinned to an old commit, hybrid Mamba architecture.

### 5. LFM2-24B-A2B huihui abliterated

Smallest GPU footprint (~0.65 GB), leaving room for ~9 of 38 expert layers in VRAM at 0.286 GB each. Architecture `lfm2moe` in mainline. 64 experts top-4, 40 layers. The base is from February 2026 and weaker, and no abliteration evals are published.

Speed-first fallbacks that fit trivially:

- mradermacher/Huihui-LFM2.5-8B-A1B-abliterated-i1-GGUF, Q6_K 6.96 GB (experts 6.36 GB, ceiling 47.8 t/s).
- mradermacher/Huihui-Ling-mini-2.0-abliterated-i1-GGUF, Q4_K_M 9.91 GB (experts 9.23 GB, `bailingmoe2`, ceiling 131.8 t/s).

### MTP caveat for all picks (inferred, unmeasured)

With experts on CPU, verifying k drafted tokens touches up to k x top-k distinct experts per layer, so DDR4 bytes scale nearly linearly with draft length. The Gemma card's "~35% faster" was measured with `-ngl 99` and will not transfer. The drafter was also trained against stock weights; acceptance against an abliterated IQ3 target is unmeasured.

## Rejected

| Candidate | Reason |
|---|---|
| HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP (user's candidate) | Ships only Q4_K_M at 16.80 GB; experts 15.13 GB exceed total RAM, meaning ~3-4 GB of permanent NVMe spill. Repo also holds mmproj BF16 1.19 GB and the 252 MB drafter. Needs mainline llama.cpp with `gemma4` + `gemma4-assistant`. QAT only pays off at ~4-bit and no QAT-based quant under 14 GB exists (the huihui QAT-abliterated GGUF repo and heretic Q4_0 at 14.49 GB are also too big). |
| Gemma4 IQ4_XS, 13.92 GB | Experts 12.37 + embd 0.61 = 12.98 GB RAM, over the 12 GB target. Closest step up; would need ~1 GB spill or 3 expert layers on GPU. |
| Qwen3.8-Flash-Next, any variant | ~84 GB at 3.78 bpw (soyaakinohara shards 42.64 + 40.91 GB). Qwen3.8 has no small MoE: the Qwen org lists only 27B dense, Flash-Next, and 2.4T-A95B. |
| GLM-5.3-Flash | ~157 GB at IQ4_XS (marcorez8 shards 49.99 + 49.61 + 49.49 + 7.73). |
| PrismML Bonsai | Every release is dense. Ternary-Bonsai-2-27B has base Qwen/Qwen3.8-27B (PQ2_0 7.21 GB, PTQ1_0 5.95 GB). No MoE Bonsai in prism-ml's 36 repos. A dense 27B has no experts to offload and will not fit 4 GB VRAM. |
| Muse-Glimmer-30B | Dense (hidden 6656, intermediate 19968, no expert fields in config). |
| Stock Nemotron-3.5-Lightning abliterated quants | All >= 18.66 GB. |
| gbuzhf/Ornith-1.5-35B-A3B-Abliterated-MTP-UD-APEX-GGUF, APEX-I-Mini with MTP | 14.37 GB, over the 14 GB cap. |
| OS-Software/llm-jp-4-32b-a3b-thinking-uncensored-heretic | No GGUF. |
| ERNIE-4.5-21B-A3B, Granite 4.x | No abliterated GGUF found in search. |
| Josiefied (Goekdeniz-Guelmez), mlabonne, TheDrummer, davetha | Nothing current in this size class (davetha is Qwen3.8-27B dense and Flash-Next only). |

## Verified vs inferred

### Verified from live sources

- File sizes, download / like counts, creation dates: `https://huggingface.co/api/models/<repo>?blobs=true` for each repo named above.
- Expert / token_embd / non-expert byte splits, tensor quant types, expert_count / used, KV head layout, nextn presence: GGUF headers via range requests on `https://huggingface.co/<repo>/resolve/main/<file>`.
- Base architectures: `config.json` for google/gemma-4-26B-A4B-it, Qwen/Qwen3.6-35B-A3B, ornith-ai/Ornith-1.5-35B-A3B, nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, openai/gpt-oss-20b, LiquidAI/LFM2-24B-A2B, LiquidAI/LFM2.5-8B-A1B, nex-agi/Nex-N2.5-mini, meta-models/Muse-Glimmer-30B, inclusionAI/Ling-mini-2.0, llm-jp/llm-jp-4-32b-a3b-thinking.
- Mainline architecture support (`gemma4`, `gemma4-assistant`, `qwen35moe`, `nemotron_h_moe`, `gpt-oss`, `lfm2moe`, `bailingmoe2`): <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp>
- `draft-mtp` spec type: <https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/speculative.cpp>
- ShimQuant repo metadata: <https://api.github.com/repos/JoshBolding/shimquant>
- Eval numbers quoted: model cards at `https://huggingface.co/<repo>/raw/main/README.md`. All author-reported; none independently verified.
- Account ages: <https://huggingface.co/api/users/HauhauCS/overview>, <https://huggingface.co/api/users/BoldingBuilds/overview>

### Inferred

- KV cache sizes at 8K (arithmetic above).
- That llama.cpp duplicates the tied `token_embd` onto the GPU as the output tensor (adds 0.61 GB for Gemma, 0.11 GB for LFM2).
- Real decode landing well below the bandwidth ceiling.
- MTP gains being much smaller with CPU-resident experts.
- Expert-cache coverage fractions (~11-14% of experts in spare VRAM).
- Ornith being a Qwen3.6-35B-A3B finetune (identical config and 35,951,822,704 parameter count).

## Not done

- No third-party refusal / capability leaderboard (e.g. UGI) was checked.
- gpt-oss heretic-ara-v4 GGUF availability not checked.
- The heretic-ara-v3 file's own header was not scanned (layout taken from the HauhauCS MXFP4 file of identical size).
- No model was run; no decode speed, MTP acceptance rate, or refusal behavior was measured.

## Addendum (2026-09-19, after the research pass)

Items marked "reported" were relayed by the team lead and were not checked by this research pass. Items marked "cross-checked" were confirmed against a live source.

- **Downloads (reported, sizes cross-checked).** Both recommended files plus the drafter are downloaded and byte-verified on the box at `/ai/models`. The byte counts match the Hugging Face API exactly:

  | File | Bytes | HF LFS sha256 |
  |---|---|---|
  | Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-IQ3_M.gguf | 12,392,563,552 | `75010d66e408d008fbe1122bf8c33548ed43d8a37e7056b6c0837eb21b5e1a4b` |
  | mtp-gemma-4-26B-A4B-it.gguf | 251,937,728 | `62bd3af7f66c9308de9a5454233852f8c7324c93767e8dfb824ed45b9179864a` |
  | Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-IQ2_M.gguf | 11,659,235,456 | `ba3a1d47a604f17ef74d913f9a9d9a2b456acc6925e7b168bfa2e6527246011c` |

  The sha256 values are from the HF API; whether the on-box check compared hashes or only sizes was not stated.

- **PrismML fork (reported).** The PrismML llama.cpp fork at `prism@9a9394a` supports `gemma4`, `gemma4-assistant`, and `qwen35moe`, and has AVX2 branches for iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_xs, q2_K, and q3_K. That covers most expert tensor types in both recommended files: Gemma IQ3_M uses IQ3_S (gate_up) and Qwen IQ2_M uses IQ2_S + IQ3_S. Not in the reported list: IQ4_NL and Q5_0, which Gemma IQ3_M uses for its `ffn_down_exps` (27 and 3 layers), and Q4_0, which Gemma Q2_K_P uses for down. Those are long-standing mainline types with AVX2 paths upstream, but their presence in this fork at this commit should be confirmed before treating CPU expert decode as fully vectorized.

- **Mainline issue ggml-org/llama.cpp#24670 (cross-checked: exists, open, created 2026-06-15).** Title: "draft-mtp speculative decoding not activating on Turing (sm_75) with hybrid SSM+attention model (Qwen3.6-35B-A3B)". <https://github.com/ggml-org/llama.cpp/issues/24670>. As reported: on a GTX 1650 SUPER with Qwen3.6-35B-A3B and `--n-cpu-moe`, `draft-mtp` never drafts unless `--spec-draft-p-min 0.0` is set. This is the exact GPU and one of the two recommended models. Two consequences:
  - The recommended Qwen3.6 IQ2_M GGUF contains no MTP tensors (verified above), so draft-mtp on that model needs a separate MTP source regardless of this issue.
  - Whether the same failure affects the Gemma4 + `gemma4-assistant` drafter pair on Turing is not established by the issue title (it names the hybrid SSM+attention model). Test with and without `--spec-draft-p-min 0.0` and check the server's draft acceptance counters before concluding MTP is working.
