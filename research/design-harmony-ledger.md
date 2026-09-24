# The per-link byte ledger ("harmony") — decode after a 9.3k-token prompt, STABLE, MTP n=2

Why: every optimization moves bytes between five resources; a move that relieves one link can load another. Rank moves by
what they do to the CRITICAL link, not by how fast a kernel gets in isolation. (att1 09:38 proved it: the FA tile kernel is 1.9x
faster per launch, and the saved 1.35 ms/token became GPU idle — total 22.04 -> 22.08 ms/token.)

## The links (measured peaks on i5 + GTX 1650 SUPER)
| link | peak | used by |
|---|---|---|
| VRAM | 192 GB/s | mat-vec (dense + cached experts), flash attention (KV), everything on the GPU |
| PCIe 3.0 x16 | 12.4 GB/s H2D | expert-cache fills, split input copies |
| DDR4 | ~38 GB/s (6 streams reach ~27) | CPU expert misses (the miss phase), AND the DMA reads behind every cache fill |
| CPU compute | 6 cores AVX2 | expert misses (1 thread = compute bound, 6 threads = memory bound: cpu1bench / cpu1bench2) |
| GPU compute | TU116, no tensor cores | MMA paths slow (dp4a MMQ, tile FA win) |

## Decode ledger per token (att1_nsys_served, 128 tokens after the 9,279-token prompt, graphs off)
| component | ms/token | link | note |
|---|---|---|---|
| mat-vec kernels (mmvq) | 9.09 | VRAM | 152 launches/token, 60 us each — roofline not yet checked |
| flash attention (MMA) | 2.98 | VRAM | 630 us/launch for 19 MB of KV = ~30 GB/s (6x under peak); tile kernel 334 us |
| GPU idle (waiting on host) | 5.93 | DDR4 via CPU | the miss phase: ~0.47 ms per layer per pass at the DRAM wall |
| H2D busy | 2.33 | PCIe + DDR4 | 28.3 MB/token; 25.4 MB/token are cache fills (>= 256 KiB copies) |
| total | 22.04 | | 45.4 t/s under nsys; 47.5-50.4 t/s with graphs on |

H2D placement (h2dov.py): 46% of all H2D and 40% of the cache fills (~10 MB/token) run INSIDE the CPU-bound idle windows, i.e.
they read DDR4 while the CPU miss phase is saturating it. 30.6 idle gaps per token, median 42 us, p90 498 us.

## Critical path
Per layer: GPU computes attention + cache hits while the CPU computes the misses; the layer ends when both finish. The CPU miss
phase is longer -> the GPU idles 5.9-7.0 ms/token -> DDR4 during the miss phase is the critical link. Anything that only speeds
the GPU (FA, mmvq) turns into idle unless it also shortens the miss phase.

## Moves ranked by effect on the critical link
| # | move | critical-link effect | other links | exact? | est. gain | status |
|---|---|---|---|---|---|---|
| 1 | MISS SPLIT: per layer, stream ~1/3 of the misses over PCIe to the GPU while the CPU computes the rest | miss phase 0.47 -> ~0.33 ms/layer (both paths together approach the ~38 GB/s DDR4 peak) | PCIe +, VRAM scratch slots | rounding differs (GPU vs CPU expert) = same class as a cache hit; KLD gate | ~+10% decode | design |
| 2 | ADMITTED MISSES ON THE GPU: an expert the cache admits is today read from DDR4 twice in one step (CPU compute + DMA fill); compute it on the GPU from the fill instead | -25 MB/token of CPU-side DDR4 reads (~15% of the miss phase) | none added | same class as a cache hit | ~+3-5% | design (a subset of 1) |
| 3 | FILL TIMING: issue cache fills outside the CPU miss windows | ~10 MB/token less DDR4 contention in the miss phase | none | exact (timing only) | ~+1.5-2% | idea |
| 4 | FA tile at long KV (GGML_CUDA_FA_MMA_MAX_KV, fa-kv 3ea44a4) | none directly (GPU side) | VRAM - | tile vs MMA rounding (pmux5: non-inferior) | +3% @9.3k (noise), +9.6% @2.3k (spread 4.8) — confirm | patch ready |
| 5 | KV cache q8_0 | none (GPU side) | VRAM -, frees ~120 MB at -c 12288 -> ~+100 expert slots -> hit rate up | lossy (KLD) | small | idea |
| 6 | multi-turn prefix reuse (hybrid-state checkpoints) | removes whole re-prefills (~23 s TTFT per 9k turn) | all | exact | huge for real chat, zero for the bench | mt1 measuring |

## Next measurements
- mmvq roofline: bytes per launch from the GGUF tensor sizes vs 60 us each (is the dense/cached-expert mat-vec at VRAM peak?).
- Per layer: miss count, miss bytes, CPU phase duration (host-side timestamps) -> the split ratio for move 1.
- Scout report (research/scout-2026-09-23-outside-box.md): HybriMoE / Fiddler / ktransformers designs for move 1.

## Outcomes (2026-09-23 night, all pre-registered; details in notes.md)
| move | predicted | measured | verdict |
|---|---|---|---|
| 1-3 expert bytes over PCIe (same-step prefetch, window prefetch) | ~+10% | step: -3..-13%; window: -1..-3% (host idle -0.35 ms, device +1.06 ms/token) | the DMA shares DDR4 (step) and VRAM/PCIe with the kernels (window): parked |
| 4 FA tile above a KV threshold | GPU-side only | +6..7% decode after 9.3k (the device phase IS on the critical path at long KV) | PROMOTED (0021) |
| draft vocabulary (FR-Spec) | ~+3-5% | +5..10% specbench, +3.7% after 9.3k, acceptance unchanged | PROMOTED (0022/0023) |
| draft length re-tune after FR-Spec | n/a | n3 +4.1% specbench, -2.9% after 9.3k (noise) | WIN, config (keep n2 for long context) |
| upload/compute overlap for prefill | ~+15% prefill | +23% prefill at 9.3k, +19% at 2.2k, decode unchanged (after two fixes: plan-loop cost, plan gate) | PROMOTED (0024-0030) |
| 5 q8_0 KV | small | KLD ok, -3.6% after 9.3k | not proven |
| more cache slots (VRAM reclaim) | +2-3% | +2.2% at cache 30 (inside spread); the draft embedding cannot be moved by -ot; tune1 on the promo5b stack: c28 OOMs mid-bench, c30 at load, c24 at 9.3k | dead at today's footprint |
| dense weights Q4_K from the Q6 source (k2q6/k2q6b) | accuracy | KLD -44%, specbench +4.2% at c21, 9.3k +2.1% at c18, prefill +3% | WIN (model file: Andrei's call) |
| dense weights IQ2_S -> Q4_K (k2d) | ~1 ms/token off the ALU-bound mat-vecs | decode -0.2% (4 fewer cache slots, hit -4.4 pts), prefill +2.8%; KLD +5.8% (requantized from K2) | dead for decode; Q6-sourced variant = accuracy follow-up |
| n3 draft at long context | 0 | 9.3k decode n3 53.4 vs n2 54.4 (noise) | parity: n3 may be the single default |
| SMT threads | 0-8% | -t 8 +0.2%, -t 12 -8.5% | dead |
## Outcomes (2026-09-24, speculation line; research/spec-design-2026-09-24.md, notes.md spec0-spec3)
| move | predicted | measured | verdict |
|---|---|---|---|
| self-distilled MTP head (L1) | needs a depth collapse | per-depth acceptance 0.82 / 0.80 / 0.77 (T = 0): no collapse (spec0 H1) | demoted |
| siblings / hybrid tree (L3 / L4) | +width where experts are shared | a sibling = 2.40 NEW experts / layer vs 4.15 per token; target in draft top-2..4 after a miss 65%; simulator fit failed its 3% rule; n4 -3.6% on the box | closed for now (re-price after M1) |
| cache chain serves 5-token batches (P1) | removes the n4 cliff (47 t/s) | correct (batch-shape class, spec2b); n4 58.5 vs n3 60.7 t/s | correct but unused (parked) |
| confidence stop (--spec-draft-p-min 0.6 / 0.75) | fewer wasted drafts | -1.0% / -1.7% | dead |
| draft length n2 vs n3 | 0 | n2 +3.2% (spec2b), n2 >= n3 (spec1cal) | WIN candidate (serving flag: Andrei's call) |
| M1 draft inside the verify's CPU window | the draft share of ~9-13 ms / position | draft = 19% of the marginal position cost; verify GPU busy +5.2 ms / position = 41% (spec3) | BETWEEN; spec4 attributes the GPU growth per op |
Lesson: on this box the winners removed WORK (fewer draft-head rows, a faster kernel for the long-KV scan, a hidden upload) or
reordered it into idle windows on the SAME resource; every move that shifted bytes between DDR4, PCIe and VRAM lost to the shared
bottleneck it created.
| verify GPU growth attribution (spec4) | GDN >= 50% (H7) | cache chain 53% / dense 30% / GDN 10% per extra position | GDN not the decode target; next: skip the dummy-slot mat-vecs (P2) |
