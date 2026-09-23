# STABLE model files: where they come from

Names and sha256 hashes live in `stable/stable.env` (generated table below). Two of the three files are built by us from public
inputs; the recipes are the box job scripts named here.

<!-- BEGIN GENERATED: model-hashes (bench/docgen.py; edit the source, not this block) -->
| file | sha256 |
|---|---|
| `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-K2q6-denseQ4K.gguf` | `325bd72443ae9f6fb71a819e96d7eeffd7a7661982418712b981aded4564d727` |
| `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` | `606fca331adcbfbdadc107512ce6a7161e84e1646ba0e0018256426f6296877f` |
| `mtp-Qwen3.6-35B-A3B-vocab49k.bin` | `28f138754fb39d76af7a9e3f95bf2e4c6a209b2e54a8e66d503d56eab1b865e3` |
<!-- END GENERATED: model-hashes -->

## Draft head: `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` (public)
The standalone multi-token-prediction head of Qwen3.6-35B-A3B, Q4_0, 1,060,038,432 bytes, downloaded unchanged from Hugging
Face `ggml-org/Qwen3.6-35B-A3B-GGUF` (2026-09-19). Used with `-md ... --spec-type draft-mtp`.

## Model: `...-K2q6-denseQ4K.gguf` (built by us, 13,126,146,176 bytes)
Base weights: `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` on Hugging Face (an uncensored fine-tune of
Qwen3.6-35B-A3B), file `...-Q6_K_P.gguf` (~30.6 GB, the near-lossless reference we also measure KL divergence against).
1. **imatrix** (`bench/box/kq1.sh`): `llama-imatrix` over the public `...-IQ2_M.gguf` from the same repo, 200 chunks of 512
   tokens from a 900 KB calibration text = 450 KB of llama.cpp source files + the first 450 KB of wikitext-2 `wiki.test.raw`.
2. **K2** (`bench/box/kq1.sh`): `llama-quantize --imatrix` from Q6_K_P, experts gate/up Q2_K and down Q3_K, everything else the
   stock IQ2_M recipe (base type IQ2_M). Experts ~11.7 GB: the largest K-quant whose experts fit in 16 GB of RAM.
3. **K2q6** (`bench/box/k2q6.sh`): quantize Q6_K_P again with base type Q4_K, then `bench/box/gguf_splice.py` copies its dense
   tensors (attention, recurrent-layer output, shared expert: 310 tensors) into K2; experts and all other tensors stay
   byte-identical to K2. Why: the dense tensors at IQ2_S carried ~44% of K2's divergence (KLD vs Q6 0.204 -> 0.115) and Q4_K
   is faster on GPUs without tensor cores (notes.md, 2026-09-24 k2q6 / k2q6b).
Lesson baked in: always quantize from the Q6 source; re-quantizing an already-quantized file compounds the error (k2d).

## Draft vocabulary: `mtp-Qwen3.6-35B-A3B-vocab49k.bin` (built by us, 196,616 bytes)
The ~49k token ids the draft head scores (patch 0022, FR-Spec), made by `bench/box/fr2_vocab.py`: token frequencies over the
model's own benchmark answers, the benchmark prompts, prose, wikitext-2, llama.cpp source, our bench scripts and part of the
Python standard library; the 49,152 (48 Ki) most frequent tokens plus the chat-template special tokens. Optional: without it the draft scores all 248k
tokens (same output, a few percent slower).
