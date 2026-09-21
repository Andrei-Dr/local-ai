# Bounded Sparse Attention for Long-Context Decode

Scout report, 2026-09-21. Target: Qwen3.6-35B-A3B on GTX 1650 SUPER 4 GB, q4_0 Hadamard-rotated KV, batch-1 decode at 120k-256k depth.

---

## 1. Query-Aware Block Selection with Provable Bounds

**Quest** (arXiv:2406.10774) is the canonical reference. Per 128-token KV block it stores `K_min[d]` and `K_max[d]` (element-wise min/max of keys). The score for query q against block j is `sum_d max(q_d * K_min_d, q_d * K_max_d)` -- a provable upper bound on the maximum q.k for any key in the block. At 2x16 bytes (2 vectors per block), the summary is 8 KB/block for head_dim=256 fp16. [verified: opened primary source]

**BLASST** (arXiv:2512.12087, MLSys'26 Best Paper) takes a different approach inside FlashAttention: it tracks the running row-max of q.k during the online-softmax sweep and skips a block if `block_max - running_max < ln(lambda)`. The output error is bounded: `||y - y_hat|| <= |S| * B_c * lambda * V_max`. No summary storage at all -- the bound is computed from already-available statistics during the forward pass. Reported ~58% sparsity on Qwen3-Coder-30B at 200k. [verified: opened primary source]

**RBS-Attention** (arXiv:2609.20971) stores a centroid c and radius r per block; the bound is `q.c + ||q|| * r` (Cauchy-Schwarz). The paper shows the axis-aligned box (Quest) and the centroid-ball capture different geometry -- neither is uniformly tighter. RBS adds a "rescue branch" for high-radius blocks to avoid displacing useful candidates. [verified: opened primary source]

**RoPE tightness problem**: Quest's per-channel min/max bound is computed on the *post-RoPE* keys. Because RoPE applies position-dependent 2x2 rotations per dimension pair, the per-channel range of keys in a 128-position tile is inflated by the rotation sweep, loosening the bound. At 120k+ positions the low-frequency RoPE dimensions barely rotate within a tile (tight bound), but high-frequency dimensions rotate through full cycles (loose bound). Qwen3.6 uses **partial_rotary_factor=0.25**: only 64 of 256 head dims are RoPE-rotated. The remaining 192 dims are position-free content channels where Quest's bound is exact. This is favorable: 75% of dimensions contribute a tight, position-invariant bound. [verified: config.json on HuggingFace; RoPE loosening is geometric reasoning, unverified empirically]

## 2. Interaction with Hadamard-Rotated Quantized KV

The Hadamard rotation in llama.cpp PR #21038 rotates K (and V) before q4_0 quantization. This **destroys per-channel min/max structure**: a Hadamard mixes all channels uniformly, so `K_min[d]` and `K_max[d]` in the rotated basis no longer bound the original channel-aligned dot product. Quest's bound in the rotated space is valid but extremely loose -- every channel now carries the full dynamic range. [unverified empirically, but mathematically certain from Hadamard's equal-mixing property; confirmed by KVLinC (arXiv:2510.05373) showing Hadamard increases key quantization scaling factors]

**Fixes**:
- **Centroid+radius (ball) bound**: Store centroid c and max-deviation radius r per tile *in the rotated space*. The bound `q_rot . c + ||q_rot|| * r` is valid regardless of basis and its tightness depends on the within-tile key variance, not on per-channel outlier structure. Cost: 2 * 256 * 2 = 1 KB/tile (fp16 centroid + scalar radius). [unverified: no paper tests this on Hadamard-rotated KV specifically]
- **Norm-based bound**: Store `max(||k||)` per tile. Bound: `||q|| * max_k_norm`. Cheapest summary (1 scalar/tile) but loosest -- ignores directional alignment. [unverified]
- **Unrotated-space summaries**: Store Quest-style min/max in the *unrotated* space (before Hadamard). On dequant+un-rotate to check the bound you pay the Hadamard inverse, but for the *summary* only (not all keys). Tight, but the un-rotation adds FLOPs to the selection pass. [unverified; no known implementation]

## 3. Measured Attention Sparsity at Long Context (GQA)

"Inference Time Context Sparsity: Illusion or Opportunity?" (arXiv:2605.24168) tested Qwen3.5 family models (same architecture as Qwen3.6) under extreme sparsity. Key finding: **restricting attention to 128 retrieved tokens (~250x sparsity) preserves strong performance** on larger hybrid models. The linear-attention (GDN) layers carry the bulk of "memory"; the 10 full-attention layers act as retrieval checkpoints and tolerate aggressive pruning. [verified: opened primary source]

**GQA group union problem**: With 16 query heads over 2 KV heads (group of 8), a tile must be kept if ANY of the 8 query heads needs it. Retrieval-head analysis of Qwen3-Coder-30B-A3B (arXiv:2605.24168 and related work) shows most query heads have low retrieval scores -- only a small subset exhibits strong long-range retrieval. But the union of 8 heads will include the one "retrieval head" in each group, which may attend broadly at retrieval layers. Expect per-group sparsity to be materially lower than per-head sparsity, especially in the ~10 attention layers that serve as retrieval checkpoints. [verified: retrieval-head finding; GQA union impact is inferred, not directly measured]

**Diffuse heads**: Some heads (especially early layers) distribute attention broadly. In GQA, one diffuse head per group kills block-skipping for that group. The BLASST per-layer/per-head heterogeneity plot (Figure 7) confirms substantial variance. Mitigating this requires per-head or per-KV-group thresholds, not a uniform one. [verified: BLASST Figure 7]

## 4. llama.cpp / ggml State of the Art

**Issue #28734** (ggml-org/llama.cpp) documents the exact problem: the QSA sparse-attention path for Qwen3.8-Flash-Next was running as a dense O(n_kv) scan due to an unfinished TODO in PR #27742. The fix -- passing `top_k->ne[0]` as `n_kv_max` to flash-attn -- yielded +280% throughput at 250k context (11 -> 41.8 tok/s). **However**, this applies to models with a native sparse-attention indexer (QSA); Qwen3.6-35B-A3B does not have one. [verified: opened GitHub issue]

**No known PR or fork implements post-hoc sparse/block-selected attention for the `vec` CUDA flash-attention kernel** used on non-tensor-core GPUs (GTX 16xx). The vec kernel is acknowledged as slow by maintainers. The MMA/WMMA kernels require tensor cores. On GTX 1650 SUPER (cc 7.5, no tensor cores), you are stuck on the vec path for quantized KV. [verified: llama.cpp discussion #13946, #15650]

**Vulkan sparse FA** is WIP (mentioned in weekly report Sep 2026) but irrelevant for CUDA. [unverified: weekly report summary only]

## 5. Hybrid GDN/Attention Layer Peakedness

Qwen3.6-35B-A3B: 40 layers, 30 GDN + 10 full attention (every 4th layer). The full-attention layers are explicitly described as "retrieval/precision checkpoints" in Qwen's blog and architecture papers. [verified: config.json, Qwen blog]

"Inference Time Context Sparsity" (arXiv:2605.24168) shows Qwen3.5 27B (same arch) tolerates 250x sparsity in attention layers with minimal degradation. Larger scale saturates faster (27B near-parity at K=4 retrieved tokens). This strongly suggests the attention layers are highly peaked at retrieval targets, with long diffuse tails that contribute negligible mass. [verified: opened primary source]

ConSA (arXiv:2606.18056) studies controllable sparsity in hybrid attention and confirms the attention layers in these hybrids concentrate mass on a small number of positions. [unverified: abstract only]

## 6. Bottom Line: Best Bounded-Error Design for Our Constraints

**Recommended approach**: BLASST-style inline thresholding, adapted for the vec kernel.

**Why**: No external summary storage needed. The bound (`block_max - running_max < ln(lambda)`) is computed from values already available in the FlashAttention inner loop. It works in any basis (survives Hadamard rotation). It works with quantized KV (the comparison is on dequantized q.k products). It naturally adapts per-head and per-layer. No top-k selection pass; the pruning is the attention loop itself, just with early-exit per tile.

**Summary to store per tile**: None for BLASST. If adding a pre-filter to avoid even loading tiles: one scalar `approx_max_qk` per tile (the max q.k seen when the tile was last written or a centroid-based estimate). Cost: 4 bytes/tile. At 120k context with 128-token tiles, that is 940 tiles * 2 KV heads * 4 bytes = 7.5 KB total. Negligible.

**Expected pruning rate**: At 120k+ context in the 10 attention layers, based on Qwen3.5 sparsity measurements, expect 70-90% of tiles to be skippable per query head. GQA group union will reduce this to ~50-70% per KV group (the retrieval head in each group attends more broadly). At the 5% selection-cost budget, BLASST's comparison-only overhead trivially fits.

**What would kill it**:
1. **GQA union with a diffuse retrieval head**: If one query head per group distributes attention uniformly at a retrieval layer, no tiles are skippable for that group at that layer. Mitigation: split the skip decision per query head (compute 8 separate masks, OR them for the KV load decision but skip the matmul per head).
2. **Branching cost in the vec kernel**: You already measured that per-position branching is 2x slower. The BLASST design branches per *tile* (128 positions), not per position, which is ~1000x fewer branch points. But the vec kernel's control flow may still suffer if the warp diverges on the skip predicate. This must be benchmarked.
3. **Accuracy at the GQA boundary**: Skipping a tile for one query head but not another in the same KV group requires loading the tile anyway (for the non-skipped head). The VRAM bandwidth saving is only realized when ALL 8 heads agree to skip. In practice the savings may be closer to the 50-70% range (all-skip tiles) than the 90% per-head rate.

---

## Verify Before Acting

1. **Qwen3.6-35B-A3B partial_rotary_factor**: Confirmed 0.25 from HuggingFace config.json. Verify it matches what llama.cpp reads (check `hparams.rope_ratio` or equivalent in the GGUF metadata).
2. **BLASST per-tile branching cost in the vec kernel**: The original BLASST targets tensor-core MMA kernels. The vec kernel has a fundamentally different control flow (scalar FMA loop over head dims). Benchmark the cost of a per-tile `if` guard around the inner loop -- this is the make-or-break measurement.
3. **GQA group union sparsity**: No paper measures the fraction of tiles skippable by ALL 8 query heads simultaneously. Profile this on Qwen3.6 at 120k context: for each KV tile, compute the max attention weight across all 8 query heads; count tiles where this max is below threshold. This determines real-world pruning rate.
4. **Hadamard rotation vs. centroid bound tightness**: Compute within-tile key variance in the rotated basis at representative layers. If the radius r is large relative to centroid norms, the ball bound will be loose and a pre-filter won't help much (rely on BLASST inline thresholding alone).
5. **Retrieval-layer peakedness per KV group**: Profile the 10 attention layers of Qwen3.6 at 120k depth. Measure per-KV-group attention entropy. If any group at any layer has entropy > 5 bits (roughly uniform over 32 tiles), that group/layer cannot be pruned.
