# Third-party material

Everything in this repository is MIT-licensed (see `LICENSE`) except the third-party material below, which keeps its own
license. Licenses were read from each source's own license file or Hugging Face dataset card on 2026-09-24.

## Code

| what | where in this repo | source | license |
|---|---|---|---|
| llama.cpp (context lines and code inside our patches, and upstream PR diffs) | `research/patches/` | https://github.com/ggml-org/llama.cpp | MIT, Copyright (c) 2023-2026 The ggml authors |

Our patches are modifications of llama.cpp; the ggml authors' MIT notice applies to the llama.cpp code they contain, and our own
changes are released under this repository's MIT license. Upstream PR diffs (`research/patches/pr*.diff`) are unmodified copies
of llama.cpp pull requests, kept for reference.

## Benchmark datasets (item subsets in `bench/qual/data*`)

| set | file(s) | source | license |
|---|---|---|---|
| GSM8K | `gsm8k.jsonl` | HF `openai/gsm8k` (Cobbe et al., 2021) | MIT |
| HumanEval | `humaneval.jsonl` | HF `openai/openai_humaneval` (Chen et al., 2021) | MIT |
| MMLU-Pro | `mmlu_pro.jsonl` | HF `TIGER-Lab/MMLU-Pro` (Wang et al., 2024) | MIT |
| HumanEval+ | `humaneval_plus.jsonl` | HF `evalplus/humanevalplus` (Liu et al., 2023) | Apache-2.0 |
| XSTest | `overrefusal.jsonl` (`xstest/*` items) | HF `Paul/XSTest` (Röttger et al., 2024) | CC BY 4.0 |
| OR-Bench | `overrefusal.jsonl` (`orbench/*` items) | HF `bench-llm/or-bench` (Cui et al., 2024) | CC BY 4.0 |
| MATH-500 | `math_l5.jsonl` | HF `HuggingFaceH4/MATH-500`, a subset of MATH (Hendrycks et al., 2021; MIT) chosen by OpenAI's PRM800K work (MIT) | the subset's card declares no license; the underlying MATH set is MIT |
| AIME 2024 / 2025 | `aime.jsonl` | HF `HuggingFaceH4/aime_2024`, `yentinglin/aime_2025` | the cards declare no license; the problems are from the American Invitational Mathematics Examination (Mathematical Association of America), included here for non-commercial benchmarking |

The model answers in `bench/qual/results/` quote these items (and the models' responses to them); the same terms apply to the
quoted items. CC BY 4.0 items are attributed above; no changes were made to the item text.

## Used but not included
- wikitext-2 (`wiki.test.raw`, Merity et al., 2016; CC BY-SA 3.0): the text behind our KL-divergence measurements and part of the
  imatrix calibration text; fetched on the benchmark box, not redistributed here.
- Model weights (Qwen3.6-35B-A3B and the HauhauCS fine-tune, the ggml-org MTP head): not in this repository; see
  `stable/MODELS.md` for sources. Their licenses are set by their publishers on Hugging Face.
