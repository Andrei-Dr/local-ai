# Quickstart: run STABLE

STABLE is the configuration we serve: a patched llama.cpp build plus one model file, tuned for a 4 GB NVIDIA GPU without tensor
cores (GTX 16xx) and 16 GB of RAM. Everything below comes from `stable/stable.env`, the single source of truth; the blocks marked
"generated" are rewritten from it automatically, so they are current by construction.

## 1. What you need
- NVIDIA GPU with 4 GB+ (built for compute capability 7.5; change `CMAKE_CUDA_ARCHITECTURES` in `stable/stable.env` for others),
  CUDA toolkit, CMake, a C++ compiler, git.
- 16 GB RAM (the experts, ~11.7 GB, stay in system memory) and ~15 GB of disk for the model files.

## 2. Build llama.cpp STABLE
```
stable/build.sh            # clones llama.cpp into ./llama.cpp, applies our patch series, builds llama-server
```
What it runs (generated):

<!-- BEGIN GENERATED: build (bench/docgen.py; edit the source, not this block) -->
```
git clone https://github.com/ggml-org/llama.cpp llama.cpp && cd llama.cpp
git checkout e613ef2
git am ../research/patches/mainline-series/*.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_MMQ_NO_MMA=ON -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build -j --target llama-server
```
`stable/build.sh` does exactly this and verifies the patched source tree against `LLAMA_CPP_STABLE_TREE`.
<!-- END GENERATED: build -->

## 3. Get the model files into ./models
Full provenance, recipes and sha256 hashes: [stable/MODELS.md](stable/MODELS.md).

<!-- BEGIN GENERATED: model-files (bench/docgen.py; edit the source, not this block) -->
| file | what | where from |
|---|---|---|
| `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` | draft (MTP) head for speculative decoding | public: HF ggml-org/Qwen3.6-35B-A3B-GGUF |
| `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-K2q6-denseQ4K.gguf` | the STABLE model | built by us from HF HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive (Q6_K_P); not published yet |
| `mtp-Qwen3.6-35B-A3B-vocab49k.bin` | the draft head's vocabulary (patch 0022) | built by us (bench/box/fr2_vocab.py); not published yet; optional |
<!-- END GENERATED: model-files -->

Until the built files are published, any Qwen3.6-35B-A3B GGUF works with the same build, for example the public
`...-HauhauCS-Aggressive-IQ2_M.gguf`: `MODEL=<file> stable/serve.sh`. Without the vocabulary file the draft reads the full
vocabulary: same output, a few percent slower.

## 4. Serve
```
stable/serve.sh                 # -c 4096; OpenAI-compatible API on http://localhost:8080
stable/serve.sh --ctx 12288     # longer context, fewer expert-cache slots
```
The exact command (generated):

<!-- BEGIN GENERATED: serving (bench/docgen.py; edit the source, not this block) -->
STABLE since 2026-09-24: llama.cpp `e613ef2` + patches 0001-0030, model `Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-K2q6-denseQ4K.gguf`, draft head `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` with vocabulary `mtp-Qwen3.6-35B-A3B-vocab49k.bin`. Source: `stable/stable.env`.

```
LLAMA_MTP_VOCAB_FILE=mtp-Qwen3.6-35B-A3B-vocab49k.bin GGML_CUDA_FA_TILE_MIN_BATCH=32 GGML_CUDA_FA_MMA_MAX_KV=4096 GGML_SCHED_MOE_PREFETCH=1 GGML_OP_OFFLOAD_MIN_BATCH=32 llama-server -m Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-K2q6-denseQ4K.gguf -md mtp-Qwen3.6-35B-A3B-Q4_0.gguf -c 4096 -ngl 999 -fa on -t 6 -ot exps=CPU --load-mode none --cache-ram 0 --spec-type draft-mtp --spec-draft-n-max 3 -ub 128 -ubp 2048 --moe-expert-cache 21 -b 2048
```
At `-c 12288`: `--moe-expert-cache 18 -b 4096` instead of `--moe-expert-cache 21 -b 2048`.
`stable/serve.sh` runs exactly this (`--ctx 4096 | 12288`).
<!-- END GENERATED: serving -->

## 5. Check it works
```
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":32}'
```
The server log reports the expert-cache hit rate and the draft acceptance. What to expect on a GTX 1650 SUPER (measured,
generated from the run ledger):

<!-- BEGIN GENERATED: headline (bench/docgen.py; edit the source, not this block) -->
prompt reading ~511 tokens/s at 9.3k tokens (LEGACY 47, 10.8x); writing 53-71 tokens/s (65-71 on short prompts, 53 after a 9.3k-token prompt)
<!-- END GENERATED: headline -->

Full table: the "Progress" section of [README.md](README.md).

## For contributors
- `git config core.hooksPath .githooks` once per clone: the pre-commit hook regenerates every generated block and refuses a
  commit whose patch series and `research/patches/series.toml` disagree.
- Changing STABLE = edit `stable/stable.env` (and add the promotion's patches + their `series.toml` entries); never edit a
  generated block by hand.
