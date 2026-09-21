# Dense-as-MoE: Running Dense LLMs with Contextual Neuron Subsets

Scout report, 2026-09-21. Target hardware: GTX 1650 SUPER 4 GB (Turing cc 7.5, PCIe 3 x16) + i5-10400F 6c/12t (AVX2) + 16 GB DDR4.

---

## 1. Training-Free Activation Sparsity (SwiGLU/SiLU, No Retraining)

| Method | Year | Sparsity | Quality (reasoning) | Measured Speedup | Hardware |
|--------|------|----------|---------------------|------------------|----------|
| **CATS** | 2024 | ~50% FFN-only | Minimal PPL hit; no hard-reasoning eval published | ~15% e2e | A100 |
| **TEAL** (arXiv 2408.14690, ICLR 2025) | 2024 | 40-50% model-wide | Near-zero degradation at 25%; slight GSM8K drop at 50% on Llama-3; no MATH eval | 1.53-1.8x decode (batch=1) | A100/H100 |
| **R-Sparse** (arXiv 2504.19449, ICLR 2025) | 2025 | 50% model-level | Llama-3-8B avg 66.2 vs 69.4 dense on 10 commonsense tasks at 50%; no GSM8K/MATH eval | 43% e2e decode | A100, custom CUDA kernels |
| **WINA** (arXiv 2505.19427, ICLR 2026, Microsoft) | 2025 | 50-65% | +2.94% over TEAL at same sparsity; retains GSM8K/ARC-C better; up to 63.7% FLOP reduction | No wall-clock speedup published (FLOP savings only) | Eval on Llama-2/3-7/8B, Phi-4-14B |
| **Polar Sparsity** (arXiv 2505.14884, NeurIPS 2025) | 2025 | Attention-head sparsity (batch-invariant) | "Without compromising accuracy"; no per-task breakdown in abstract | 2.2x e2e (batched) | A100 Triton kernels |
| **WiSparse** (arXiv 2602.14452) | 2026 | 50% | 97%+ of dense avg on Llama-3.1-8B; +3.95pp over TEAL on GSM8K/HumanEval | Not published | Llama-3.1-8B, Qwen-2.5-7B |

**Key gap for us**: all measured speedups use custom CUDA kernels on Ampere/Hopper. No published kernel for Turing cc 7.5. More critically, none implement the hot-GPU/cold-CPU neuron split we need -- they assume the full model fits in GPU memory and skip columns/channels within a single matmul. The speedup comes from reduced memory bandwidth, not from avoiding PCIe transfers.

[verified: opened TEAL ICLR 2025 PDF, R-Sparse ICLR 2025 PDF, WINA arXiv, Polar Sparsity arXiv]
[unverified: WiSparse specific numbers from secondary source (MarkTechPost/arXiv HTML)]

## 2. Methods Requiring Fine-Tuning

**ReLUfication**:
- **TurboSparse** (arXiv 2406.05955): dReLU replaces SiLU in both gate and up-proj. TurboSparse-Mistral-7B activates 2.5B/7B params, scores 63.65 vs 61.57 dense on Open LLM Leaderboard. Cost: continued pretraining (budget not quantified in paper but described as "extensive"). Released checkpoints: Mistral-7B, Mixtral-47B only. **No Qwen3 family checkpoints exist.**
- **ProSparse** / **"ReLU Strikes Back"** (ICLR 2024): similar approach, older models (Llama-2 era). No Qwen3 checkpoints.

**Dense-to-MoE Conversion**:
- **ExpertWeaver** (arXiv 2602.15521, ICML 2026): training-free GLU-pattern-based neuron partitioning into shared + routed experts. Tested on Llama-3-8B, Qwen-2.5-7B. At 50% sparsity (half neurons active): Llama-3-8B avg 60.4 vs 48.2 LLM-Pruner; GSM8K 34.9 vs ~0 for pruners. For downcycling (200B tokens FineWeb-Edu), outperforms OLMoE by 4.6pp. **No released converted checkpoints; calibration-only, code presumably forthcoming.**
- **LLaMA-MoE** / **PESC** (EMNLP 2024): FFN partitioning + QLoRA adapters. Supports Qwen1.5 (not Qwen3). Cost: moderate (adapter-only fine-tuning).
- **Drop-Upcycling** (arXiv 2502.19261): improved upcycling with diversity re-init. 5.9B-active matches 13B dense at 1/4 training FLOPs. No Qwen3 checkpoints.

**No released dense-to-MoE checkpoints exist for any Qwen3/3.5/3.6/3.8 dense model or Gemma 3/4 dense.**

[verified: ExpertWeaver arXiv HTML, TurboSparse arXiv, PESC GitHub]
[unverified: ExpertWeaver GSM8K numbers from secondary aggregator]

## 3. PowerInfer / PowerInfer-2 / SmallThinker Status

| Project | Last Active | Qwen3 Support | GGUF k/i-quants | CUDA cc 7.5 |
|---------|-------------|---------------|------------------|-------------|
| **PowerInfer** (Tiiny-AI/PowerInfer) | Issues open Mar 2026; team pivoted to Tiiny AI Pocket Lab (CES 2026) | **No** -- requires ReLU-sparsified models + offline neuron activation profiler; no Qwen3 predictor exists | PowerInfer-GGUF format (not standard GGUF k-quants) | Not explicitly documented; built on llama.cpp CUDA so *probably* compiles, but untested |
| **PowerInfer-2** | Jun 2024 paper; smartphone-focused (TurboSparse-Mixtral) | No | Own format | Mobile focus (Snapdragon) |
| **SmallThinker** | Jul 2025 release (21B-A3B, 4B-A0.6B) | These ARE the Tiiny models, not a framework for converting others | Own arch | Unknown |

**llama.cpp PRs/issues on activation sparsity**: no mainline PR implements neuron-level activation sparsity. The closest work is issue [#20757](https://github.com/ggml-org/llama.cpp/issues/20757) (Mar 2026) on two-tier GPU+RAM expert cache for MoE offload -- expert-level, not neuron-level. `--cpu-moe` and `--n-cpu-moe` flags already exist for routing MoE expert tensors to CPU.

[verified: Tiiny-AI/PowerInfer GitHub repo, llama.cpp issue #20757]

## 4. Whittle-Qwen-3.8-35B-A3B

- **HF repos**: [logic65/Whittle-Qwen-3.8-35B-A3B](https://huggingface.co/logic65/Whittle-Qwen-3.8-35B-A3B), [logic65/Whittle-Qwen-3.8-35B-A3B-GGUF](https://huggingface.co/logic65/Whittle-Qwen-3.8-35B-A3B-GGUF), [mradermacher i1-GGUF](https://huggingface.co/mradermacher/Whittle-Qwen-3.8-35B-A3B-i1-GGUF)
- **Architecture**: `qwen4_exp` (Qwen3.8-Flash-Next: hyper-connections, gated DeltaNet + attention, 10B n-gram memory). Runs on **stock mainline llama.cpp**, no patches.
- **GGUF sizes**: Q8_0 37.8 GB, Q6_K 29.3 GB, Q5_K_M 25.1 GB, Q4_K_M 21.3 GB, **Q3_K_M 16.7 GB**. The n-gram memory is ~10.5 GB at Q8 (scales with quant), is a lookup table, and should live in system RAM (`-ot per_layer_token_embd=CPU`).
- **Fit in 16 GB RAM**: Q3_K_M at 16.7 GB is tight -- memory portion would be ~5-6 GB at Q3, body ~11 GB. With GPU offload of active experts this might just fit, but it's marginal.
- **Quality** (self-reported, research preview by one person): GSM8K 44/50 (88%), MATH-60 46/60 (77%), stop/loop battery 23/24. Long-context gate 18/36. Teacher parity (top-1 agree) 87.5%.
- **Distilled from**: Qwen3.8-27B (thinking on), 2,861 steps on 1,840 teacher traces (math + code review).

[verified: opened both HF model cards directly]

## 5. Hot-Neuron Concentration in SiLU-Gated FFNs

SiLU does not produce natural zeros, but activation energy IS concentrated:

- **TEAL finding**: inputs to W_down and W_o follow Laplace distributions. Magnitude-based thresholding zeroing the bottom 50% of activations in SwiGLU FFNs causes negligible PPL change on Llama-3-8B, Mistral-7B, Phi-3-mini. This means the top ~50% of neurons carry essentially all output energy.
- **Sparsing Law** (arXiv 2411.02335): activation sparsity increases monotonically with model scale. Larger models have more concentrated neuron activation.
- **Universal top-p framework** (arXiv 2509.00454, ICLR 2026): systematic study confirms GLU+SiLU/GELU models have exploitable concentration, but practical ceiling is ~1.3-1.5x acceleration (lower than speculative decoding's ~4x).
- **NDP-DIMM study**: ~52% of "hot" neurons (profiled offline) show varied activation at runtime -- a static hot/cold split suffers 1.63x degradation vs oracle. Dynamic prediction is needed.

**Bottom line**: energy concentrates enough for ~50% sparsity within a single matmul (skip columns), but a *static* hot/cold GPU/CPU split without a predictor loses ~40% of the theoretical gain. The distribution is concentrated but not concentrated *enough* for a naive static partition to match MoE's hard routing.

[verified: TEAL PDF, Sparsing Law arXiv]
[unverified: NDP-DIMM 52% figure from arXiv abstract]

## 6. Bottom Line: Ranked Actions for Our Box

| Rank | Action | Expected Speedup | Quality Risk | Eng Cost | Kill Condition |
|------|--------|------------------|-------------|----------|----------------|
| **1** | **Whittle-Qwen-3.8-35B-A3B Q3_K_M** -- it IS a dense-to-MoE distillation, already in GGUF, runs on our llama.cpp+expert-cache stack | Same as Qwen3.6-35B-A3B (MoE, ~47-54 tok/s class) | Self-reported 88% GSM8K; one-person research preview, no independent eval; Q3_K_M "some loss on maths" per author | **~1 hour**: download, test with our bench suite | Q3_K_M too large for 16 GB (16.7 GB total); memory portion competes with expert cache. Verify actual RSS. |
| **2** | **ExpertWeaver on Qwen3.8-27B** -- training-free dense-to-MoE, calibration-only | Depends on sparsity ratio; at 50% active params should halve FFN compute | GSM8K 34.9 at 25% sparsity (Llama-3-8B); untested on Qwen3.8 | **High**: no code released yet (ICML 2026); would need to implement + convert to GGUF + integrate with llama.cpp expert offload | No released code; GGUF conversion path unknown; router integration with llama.cpp non-existent |
| **3** | **TEAL/WINA magnitude pruning as a diagnostic** -- profile Qwen3.8-27B neuron activation distribution, measure how concentrated energy is, decide if a custom neuron-level offloader is worth building | No direct speedup (diagnostic) | N/A | **~1 day**: run calibration pass, collect activation stats, plot top-p energy curves | Finding: if top-20% neurons carry <80% energy, the whole "dense-as-MoE" thesis fails for this model |
| **4** | **TurboSparse-style dReLU conversion of Qwen3.8-27B** | Could enable 70%+ sparsity + PowerInfer-style offload | Unknown quality loss; no one has ReLUfied a Qwen3 model; reasoning tasks are the first to degrade | **Very high**: needs continued pretraining (hundreds of B tokens), GPU cluster access | GPU cost; no guarantee quality survives on hard reasoning |
| **5** | **Hack llama.cpp expert cache to do neuron-level offload on a TEAL-sparsified dense model** | Theoretical 2-3x over current 1.38 tok/s dense | Depends on TEAL quality at 50% + accuracy of neuron predictor | **Weeks**: custom CUDA kernels for cc 7.5, neuron predictor training, integration | Custom kernel development for Turing; no existing codebase to start from; the 1.3-1.5x ceiling from the literature suggests diminishing returns vs just using a native MoE |

**Honest assessment**: the gap between "activation sparsity exists in SwiGLU models" and "a working hot/cold GPU/CPU neuron offloader that matches MoE throughput" is enormous. Every published speedup assumes the full model fits on one GPU and saves bandwidth within matmuls. Nobody has shipped a neuron-level PCIe offloader for SiLU models. Whittle (#1) is the only thing that might actually run on our box this week, and even that needs RSS verification at Q3_K_M.

---

## Verify Before Acting

1. **Whittle Q3_K_M actual RSS**: download and measure -- 16.7 GB file may exceed 16 GB RAM with expert cache overhead. Check if mradermacher i1 IQ3_M is smaller.
2. **Whittle `qwen4_exp` arch**: confirm our llama.cpp build recognizes this arch string (it's the Qwen3.8-Flash-Next format).
3. **Whittle quality**: run our standard bench suite (GSM8K, MATH, coding) -- all published numbers are self-reported by one author.
4. **ExpertWeaver code release**: check if code has appeared since ICML 2026 acceptance (paper: arXiv 2602.15521).
5. **WINA GitHub (microsoft/wina)**: verify it includes sparse CUDA kernels (not just evaluation scripts) and whether they compile for cc 7.5.
6. **mradermacher Whittle-Qwen-3.8-35B-A3B-i1-GGUF**: check for IQ3_M or IQ2_M quants that might fit in 16 GB.
7. **NDP-DIMM 52% dynamic neuron claim**: only seen in abstract; verify against full paper if pursuing neuron-level offload.
