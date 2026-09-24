# Better speculation for the i5: cost model, levers, and a pre-registered plan (2026-09-24)

Inputs: `research/mtp-code-map-2026-09-24.md` (our speculation code, tu116-served f5ddca176),
`research/scout-2026-09-24-spec-tree-lit.md` (literature, arXiv ids checked by the scout), and the measurements below
(`research/scripts/moe-cache-sim/verify_union.py`, traces `bench/box/traces/q36_{code,prose}_tok.bin`, 40 layers, top-8, ~40k tokens each).

## 1. What a verify pass costs on this box (measured)

The CPU expert phase sits on the DRAM wall (cpu1bench, 2026-09-21: 6 threads ~27 GB/s). So the price of a verify batch is the
number of expert weight reads that miss the GPU cache, not the token count.

Distinct experts per layer touched by K consecutive tokens (no cache):

| K | code | prose | vs one token |
|---|---|---|---|
| 1 | 8.0 | 8.0 | 1.0x |
| 4 | 22.1 | 20.7 | 2.6-2.8x |
| 8 | 35.9 | 32.8 | 4.1-4.5x |
| 16 | 56.3 | 51.0 | 6.4-7.0x |

CPU experts per verify (misses) with an LRU of 21 slots per layer (the simulation hits 49% at K = 1; the box measures ~48%):

| K | code: misses / check, per token | prose: per token |
|---|---|---|
| 1 | 4.1, 4.07 | 3.47 |
| 4 | 18.3, 4.57 | 3.87 |
| 8 | 35.8, 4.48 | 4.05 |
| 16 | 56.1, 3.51 | 3.17 |

Reading: the overlap between consecutive tokens is the same locality the LRU already harvests; batching does not make a
token's CPU expert work cheaper, and above K ~ 4 a verify batch evicts its own experts (the batch's union exceeds 21 slots).
**Every rejected draft token costs about one full token of CPU expert work.** Speculation pays here by amortizing the
fixed per-step cost (GPU dense layers + attention + draft + launch; ~21 ms of a 58 ms n = 2 round in the SPEC 4.1 fit) over
the accepted tokens.

Caveat: the LRU simulation is not the real admission policy (llama-moecache). How a verify batch's experts claim slots is
open; a big batch evicting what the next token needs is a hidden cost the real cache may or may not have.

## 2. What the code does today (read in source, see the code map)

- Draft: one MTP head, run autoregressively per depth (depth d feeds the head's own `h_nextn` from depth d-1), one
  `llama_decode(ctx_dft)` per depth, draft sampler top-k 10 then greedy. n = 3 at STABLE.
- Verify: a single sequence, consecutive positions, all logits. No tree, no custom mask anywhere (EAGLE-3 is linear too).
- Acceptance (`common/sampling.cpp:677-706`): the target samples its own token at each position and the draft token is kept
  while they match. With a greedy (one-hot) draft this is exactly standard speculative sampling (accept with prob
  p_target(draft token)), so it is already optimal for ONE candidate per position; the gain under sampling can only come
  from MORE candidates per position (multi-draft acceptance).
- Gated DeltaNet rollback: per-token state planes (`n_rs_seq` = draft n_max), zero-copy (the next decode reads the right
  plane). Measured: S = 180 MiB for 1 seq x 3 planes => **60 MiB per recurrent state plane** (+8.4 MiB R). One expert-cache
  slot is ~50 MiB (dense +203 MiB cost 4 slots), so every extra branch state costs ~1.2 slots per plane.
- Branches as sequences: `seq_cp` is metadata-only and forks on first decode, so K branches = K seq ids is possible without
  new graph code; attention layers share the prefix KV, the 30 recurrent layers run each branch on its own state.
  `n_seq_max` is fixed at context creation.

### The flattening observation (inferred, to verify)
Routing is a deterministic function of (context, token). If a tree is flattened into full root-to-leaf paths, a duplicated
interior node routes to the same experts in every copy, so duplicates add GPU dense work (the GPU idles 5.8-6.3 ms per token
waiting on the CPU) but no new CPU expert reads. Only the nodes that DIFFER cost CPU experts. The limit is VRAM for branch
states, not compute.

## 3. Per-depth acceptance: the MTP head collapses with depth
Early measurement (notes.md, IQ2 era): acceptance 0.75 / 0.60 / 0.50 / 0.18 as draft n grows (n = 2, 3, ..., 15);
STABLE today: 0.831 at n = 3 with the draft vocab. FastMTP (2509.18362) documents the same collapse for a recursively used
MTP head (~70 / 10 / 0% at depths 1 / 2 / 3) and fixes it with self-distilled fine-tuning (80 / 56 / 36%, 2.03x with a vocab
trim); MTP-D (2603.23911) +7.5% acceptance by self-distillation. Our head is used the same way (one head, recursive). A clean
per-depth measurement on STABLE is owed (Phase 0).

## 4. Levers, ranked by expected effect on the critical link

| # | lever | mechanism | exactness | cost / risk |
|---|---|---|---|---|
| L1 | **Self-distilled MTP head** (FastMTP / MTP-D recipe) | lift depth-2/3 acceptance: more accepted tokens per step, fewer wasted CPU expert reads | exact (drafts are only proposals) | training the head only (one decoder block) on Dave's MI210s; data = target's own generations; no box code |
| L2 | **Cost-aware draft length** (Cascade 2506.20675 / EVICT 2605.00342 style, our cost model) | stop drafting when p(accept) x fixed-cost-saved < predicted CPU misses of the next draft token (predictor 91% recall@16 + cache residency) | exact | host logic in the speculative loop; small |
| L3 | **Multi-candidate at depth 1** (K seq ids, recursive rejection sampling / SpecTr for T > 0) | catch the ~17% of positions where draft top-1 misses, and raise acceptance under sampling | exact with the multi-draft rule | +60 MiB state per extra candidate (~1.2 slots) |
| L4 | **Expert-aware hybrid tree** (Andrei: deterministic head to depth N + sampled tail; branches scored by p(accept) / new CPU experts incl. cache) | spend verify width only where it is nearly free in CPU experts | exact: T = 0 any tree; T > 0 needs multi-draft acceptance with the tail's proposal law (RheoSampling 2609.21827 separates construction and verification probabilities) | builds on L3; decided by the sibling-overlap number |
| L5 | **TreeWY-style GDN tree verify** (2608.20961) | no per-branch state planes: one triangular solve for all nodes, rebuild only the accepted state | exact | a CUDA kernel for sm_75 without tensor cores; only if L4 is worth it and VRAM is the wall |

Not for us: SpecExec-size trees (their verify is PCIe-bound streaming of dense weights; ours is CPU-expert-bound and each
distinct node costs misses); expert prefetch from draft routing (SP-MoE / MoE-SpeQ) is the dead pf1-pf3 class here (PCIe and
DDR4 contention), unless it only reorders the CPU phase.

Side finding (separate lead): server logs show `forcing full prompt re-processing due to lack of cache data (likely due to SWA
or hybrid/recurrent memory)`; multi-turn prompts on this hybrid model re-read the whole prefix unless context checkpoints
work. At 512 t/s a 9.3k prefix costs ~18 s per turn. Check what the fork's checkpoint support covers.

## 5. Phase 0: measure before building (pre-registered)

**P0a (offline, done above):** chain union and LRU misses vs K.

**P0b spec0 (box job; instrumentation behind env switches, off by default, output unchanged):**
- Dump per verified position: draft top-10 (token, prob) at each depth; target top-10 (token, prob) at the same position;
  accepted length; T.
- Dump routing (`ffn_moe_topk`, all 40 layers) for every verify token including rejected ones.
- Sibling pass: at each of N sampled positions, decode the target's top-2..4 alternatives as extra 1-token sequences
  (`seq_cp` of the prefix) and record their routing.
- Workloads: specbench prompts (code + prose), T = 0 and the card sampler; ~20 min.

Hypotheses and kill rules:
- H1 (per-depth collapse): acceptance at depth 2 and 3 conditional on depth-1 acceptance is < 0.8 x depth 1.
  If FALSE (no collapse) -> L1 is demoted.
- H2 (depth-1 alternatives): P(target token in draft top-2..4 | draft top-1 rejected) >= 0.30 at T = 0.
  If < 0.15 -> L3 / L4 are dead at T = 0 (they stay open for T > 0 only if the same number >= 0.30 under the card sampler).
- H3 (sibling overlap): mean NEW experts (not in the accepted path's set nor in the LRU) for a depth-1 sibling <= 2.0 per
  layer (a full token costs ~4). If >= 3.5 -> L4 collapses to L3 (width is never cheap).
- Offline simulator on the dump: expected accepted tokens per CPU miss for chain n = 1..5, L2, L3 (K = 2..4) and L4 shapes,
  using the measured step cost split. A lever proceeds to a box prototype only if the simulator shows >= +5% tokens/s over
  STABLE's chain n = 3.

## 6. Phase 0 results (spec0, 2026-09-24; full entry in notes.md)
H1 FALSE (0.82 / 0.80 / 0.77 per depth at T = 0: no collapse, L1 demoted); H2 TRUE (target in draft top-2..4 after a depth-1 miss:
0.65 at T = 0, 0.69 card; rank 2 alone 0.47); H3 NOT DECIDED (sibling 2.40 NEW experts / layer vs 4.15 for a full token).
Rejected draft tokens cost 3.50 NEW / layer. Prose accepts 44% of drafts vs 84% on code / reasoning.

## 7. Phase 1: the offline simulator (pre-registered 2026-09-24, before it is written)
Input: the spec0 dumps (T = 0 and card), H3 sibling costs by rank, per-token NEW costs. Step time model
t_step = t_fixed + t_draft x depth + c x (sum of NEW experts over the batch), with t_fixed / t_draft / c fitted on the box from a
short calibration run (STABLE, draft n = 1..4 and no speculation, specbench prompts) and reported with their fit error.
Counterfactuals are exact within the dumped information: a chain's acceptance at depth <= 3 follows from the draft top-1 vs the
target's token; a sibling at depth d is accepted iff the target's token equals that sibling; nothing after an accepted sibling is
credited (conservative: no dumped drafts beyond it).
Policies: chain n = 1..3; chain with a confidence stop (p_top1 < theta); L3 siblings at depth 1 (rank 2, ranks 2-3), always and
only when p_top1 < theta; L4 = at each depth, if p_top1 >= theta continue the chain, else add ranks 2..k as siblings and stop.
theta and k are swept on the code+reason+prose T = 0 dump and scored on the card dump (and vice versa); a policy is reported on
the dump it was NOT tuned on.
Gate: a policy goes to a box prototype only if its predicted tokens/s beats STABLE's chain n = 3 by >= 5% on the held-out dump
at BOTH T = 0 and the card sampler, AND the calibration fit predicts STABLE's measured n = 1..4 speeds within 3%.
Kill: no policy >= +5% -> the tree / sibling line is closed for this box; the confidence stop alone is kept if >= +2% (it is a
one-line change).
