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
| more cache slots (VRAM reclaim) | +2-3% | +2.2% at cache 30 (inside spread); the draft embedding cannot be moved by -ot | not proven |
| SMT threads | 0-8% | -t 8 +0.2%, -t 12 -8.5% | dead |
Lesson: on this box the winners removed WORK (fewer draft-head rows, a faster kernel for the long-KV scan, a hidden upload) or
reordered it into idle windows on the SAME resource; every move that shifted bytes between DDR4, PCIe and VRAM lost to the shared
bottleneck it created.
