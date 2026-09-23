# Outside-the-box speedup scout — 2026-09-23

Window respected (Sept 10-23 emphasis where noted; foundational systems predate the window and are labeled).
Method: web_search provider was down; every source below was individually fetched (arXiv export API, abs pages,
api.github.com, project pages) this session. Numbers tagged [paper claim] or [measured consumer HW]. Where a claim
could not be verified, it says "not verified". Prior-report citation errors corrected here.

## Ranked ideas

| # | Idea | exact/lossy | Expected gain on OUR box | Effort | Source |
|---|------|-------------|---------------------------|--------|--------|
| 1 | sm_75 FA tile-64 above ~10K KV (adaptive selection) | exact (fp16 accum-order noise only) | +19..50% on the 3.2 ms FA term at long KV (ours: 230K regime; ~0 at 8K) | S | issue 28761 [measured consumer HW, TU106] |
| 2 | Inter-layer expert prefetch (predict layer N+1's routing during layer N compute) | exact | attack the 6.2 ms CPU-expert stall directly; HybriMoE +70% decode vs SOTA hybrid [paper claim] | M | arXiv 2504.05897, 2401.14361 |
| 3 | SeqMoE-style predictive expert placement | exact | hit-rate 50% -> claimed 97%/45% residency class; even half-way halves the 6.2 ms + part of 2.5 ms upload | L | arXiv 2609.12978 (Sep 2026) [paper claim] |
| 4 | Cost-aware spec budget: cap draft branches by UNIQUE-expert union of verify batch | exact output; heuristic timing | protect MTP gains on MoE; paper reports verify latency ~linear in active experts | M | arXiv 2607.12696 [paper claim] |
| 5 | Marconi-style prefix caching for our GDN+attn hybrid (cache recurrent state only at reusable prefix boundaries; speculative insertion) | exact | kills inter-turn prefill, complements our slots (which pay 9.7 MB/token GDN snapshots); TTFT class-win in multi-turn | M | arXiv 2411.19379, SGLang/pytorch blog [paper claim + eng blog] |
| 6 | Batch 2 conversations (decode-only continuous batching) | exact | throughput ~2x target; per-token CPU-expert ops amortized; +~5% KV bytes at 230K (prior scout local analysis) | M | this repo scout-2026-09-19-state-restoration.md + vLLM SOSP'23 lineage |
| 7 | ktransformers-style CPU expert kernels / CPU+GPU split by expert size | exact | CPU experts are the floor (54.5 ms CPU/token ours); better AVX2 INT8/group-2 kernels could cut 10-25%; KTransformers-style balanced splits studied on 4090-class [paper claim] | M | ktransformers repo (details not verified this session) |
| 8 | MoE-Lightning CPU-GPU-I/O pipelining (CGOPipe) + roofline policy | exact | batch-mode orientation; single-stream lessons usable (paged weights + overlap) | L | arXiv 2411.11217 ASPLOS'25 [paper claim] |
| 9 | Fiddler CPU-compute-always-for-cold-experts (offline-computed CPU/GPU assignment) | exact | we already compute cold on CPU; their twist = static per-expert schedule by phase cost model — refine our LRU churn | S | arXiv 2402.07033 ICLR'25 [paper claim, 3090-class] |
| 10 | PowerInfer-2 neuron-cluster granularity (sub-expert processing unit for I/O-compute pipelining) | lossy (their accuracy story = negligible, but cluster approximations involved) | inspiration for finer than per-expert prefetch units (our 30 MB fills become rag of small pieces) | L | arXiv 2406.06282 [paper claim, smartphone HW] |

Not separately verified this session (named in the brief, existence corroborated only indirectly via the ISCA'26
survey snippet listing COMET / MegaScale-Infer / MoE-Lightning / Fiddler / Pre-gated MoE / LYNX / Sida): **ProMoE,
AdapMoE** — not verified. "Pre-gated MoE/GateNet" exists (referenced in that survey) but its paper page was not
fetched this session — headline claim not verified.

## Q1 — Hybrid CPU+GPU MoE systems: what they do that we do not

Our shape: cold experts always on CPU, GPU LRU for ~50% hits, GPU stalls 6.2 ms/token waiting CPU experts.

- **Fiddler** (arXiv 2402.07033, ICLR 2025, fetched): computes cold experts ON CPU too, but its differentiator is an
  offline profiler choosing per-(expert, stage) CPU-vs-GPU execution to MINIMIZE data movement; claims 1.26x single
  batch, 1.30x long prefill, 11.57x beam search vs offloading SOTA [paper claim, consumer GPU + Xeon]. Difference to
  us: our split is dynamic LRU-policy-driven; theirs is a static schedule — stable-routing layers stop thrashing the
  cache. Applicability 6/10 (we tried cache policies, not schedule-offlining).
- **HybriMoE** (arXiv 2504.05897, DAC 2025, fetched; implemented on kTransformers): THE thing we lack —
  (a) **impact-driven inter-layer prefetch**: while layer N executes (on either device), it predicts layer N+1's
  expert set and streams/uploads concurrently; (b) dynamic intra-layer CPU/GPU balancing by expert size and load;
  (c) impact-score cache under unstable activations. Claims 1.33x prefill / **1.70x decode** over SOTA hybrid
  frameworks [paper claim]. Applied to our stall ledger this is the most direct 6.2-ms killer. Applicability 8/10.
- **MoE-Infinity** (arXiv 2401.14361, fetched; repo EfficientMoE/MoE-Infinity): sequence-level activation TRACE +
  activation-aware prefetch/cache — predicts subsequent layers' experts from within-sequence history, 2.7-13.7x
  per-token vs vLLM/Ollama/DeepSpeed/BrainStorm [paper claim]. Their tracing DB ≈ our expert_log.jsonl but used
  ONLINE for prefetch, not offline analysis. Applicability 7/10.
- **MoE-Lightning** (arXiv 2411.11217, ASPLOS'25, fetched): batch inference focus; CGOPipe interleaves CPU compute,
  GPU compute and PCIe transfers of expert weights with paged weights; hierarchical roofline model picks batch/pipe
  policies. Lesson for us: at batch>1 (idea #6) the pipe schedule, not policy, decides throughput [paper claim].
  Applicability 5/10 (needs batching first).
- **SeqMoE** (arXiv 2609.12978, Sep 2026 window, fetched snippet): predictive + graph-aware placement; **96.97% hit
  rate at 45% expert residency, 80.22% of full-load performance**; compute-transparent placement + sync-free
  orchestration [paper claim, hardware class unclear]. Closest new answer to our 50% hit rate. Applicability 8/10.
- **PowerInfer / PowerInfer-2** (2312.12456 SOSP'24 title verified via search; 2406.06282 fetched = smartphone ed.):
  decomposition of matmuls into NEURON CLUSTERS as schedulable units; dense clusters on NPU/GPU, sparse on CPU;
  segment-based neuron cache overlaps IO; claims 27.8x over SOTA phone framework, 11.68 tok/s for 47B [paper claim,
  Snapdragon 8 Gen3 + RTX 4090 assist]. What we do not do: sub-expert granularity overlap. Applicability 4/10
  (engineering-heavy restructure), idea-value 8/10 for slicing our 30 MB/token fills.
- **Pre-gated (GateNet)**: existence confirmed via ISCA'26 "Patterns behind Chaos" survey list; paper page NOT
  fetched this session — claim not verified. The concept (cheap auxiliary net predicting gate outputs to preload
  experts pre-emptively) overlaps SeqMoE/HybriMoE prefetch above.
- **ktransformers**: repo not fetched this session (details not verified), but HybriMoE (built ON it, fetched) confirms
  its architecture: per-expert CPU(GPLLM/int8)/GPU CUDA split with CPU expert kernels — our CPU side is llama.cpp
  CPU kernels; if HybriMoE-on-kT beats that at decode (their 1.70x claim includes it as base + others), their CPU
  expert kernel/library matters beyond scheduling. Applicability: investigate 6/10.

Pipelining ACROSS tokens/layers to hide CPU-expert latency = exactly (a)+(c) in HybriMoE and MoE-Infinity prefetch;
our fork currently serializes (measure shows GPU idle 6.2 ms = exposed CPU wall, no overlap).

## Q2 — Decode attention on Turing with GQA 2 kv-heads, hd 256

- **Upstream issue #28761** (opened 2026-09-11 — INSIDE window; fetched in full): on sm_75 the FA mma-f16 path forces
  the 32-token Q tile; 64-tile variants are compiled as EMPTY STUBS (NO_DEVICE_CODE). Not an ISA boundary (tile size
  = register budgeting; same mma.sync.m16n8k8). ptxas data (shared, verifiable without GPU): head_dim 256 SPILLS at
  BOTH tiles (255-reg cap; 164B tile32, 348B tile64); head_dim 128 spills neither (<=221 regs, 0B). Runtime on
  TU106 (2x tensor-split, IQ3_S 27B, q4_0 KV) [measured consumer HW]: hd256 KV8K tile32 698 vs tile64 439 (-37%),
  32K 544 vs 650 (**+19%**), 230K 255 vs 384 (**+50%**); crossover 8-10K. Proposed fix in the issue: relax cap +
  host-side per-op threshold tile64 iff KV>8192; 4K outputs byte-identical, 17K crossing differs at char 106
  (accumulation order). **For us at 9K context: ~neutral (right at the knee, tile32 already best); at 32K+/230K
  long-session decodes: +20-50% of the attention term.** The issue measures PREFILL-style throughput per call;
  decode-step effect follows the same crossover analysis but nobody posted decode-only numbers — noted. Patch to our
  fork = S (two files as enumerated in the issue). NOTE: this issue alone makes Sept 10-23 productive.
- **Upstream PR #26404** (Aug 1, updated Sep 22; fetched): GQA ratio %8 requirement relaxed for (192,128) kernels via
  partially-filled 8-channel tiles, gated on turing_mma_available — confirms upstream awareness that Turing mma FA
  selection is fragile; our (hd 256, GQA 8:2 -> ratio 4... heads: 16 q/2 kv => ratio 8) unaffected directly but
  signals kernel-selection latitude.
- **Volta analysis issue #28037** (Aug 30; fetched, context): m16n8k8 mma (Turing+) vs m8n8k4 (Volta) gap is
  architectural; our FA sits on the GOOD side of that line — the 30 GB/s we measure for 3.2 ms is therefore NOT mma-
  starvation but consistent with a bandwidth-bound decode FA at GQA=2 kv (few heads to parallelize → poor SM
  utilization → split-K/flash-decoding territory). Upstream has NO open PR on split-K decode FA for sm_75 in the
  search window (search results above); flash-decoding literature exists (FlashDecoding++ 2311.01282 fetched:
  4.86x/2.18x vs HF, 1.37x avg vs SOTA engines [paper claim] — data-center GPUs A100/H100/MI210, no consumer datapoint).
  Realistic expectation for us: split-K over KV chunks would raise SM occupancy for our 2-kv-head decode; our 30 GB/s
  vs 192 GB/s ceiling suggests headroom up to ~4x IF parallelism fixed; no fetched source demonstrates it on sm_75 —
  flagged as un-landed opportunity, effort M (custom kernel), novelty high.
- head_dim 256 specifically: both llama.cpp FA vec/tile regimes penalized by register spills at hd256 per #28761 —
  our FA config deserves an experiment sweep (vec vs mma via GGML_CUDA_FA_QUANTS-era knobs) before concluding
  3.2 ms is physics.

## Q3 — Prefix/state caching for GDN hybrids

Correction of prior-report citations (both fetched today):
- **PR #15293** = "server: add SWA checkpoints" (merged 2025-08-14): checkpoints hold SWA-window KV+cells only
  (small, proportional to window); created on prompt completion; --swa-checkpoints N (default 3); branch-from-anywhere
  only with --swa-full. It is a sliding-window trick, NOT a general context checkpoint — and does not snapshot GDN
  recurrent state (semantics per fetched body).
- **"PR 16607" as cited in scout-2026-09-19-hunt2 was wrong**: #16607 (fetched) = "webui: reorganize settings
  layout" (Oct 2025). The general state persistence our slotlib rides is **context shards, PR #16285** (verified in
  the 2026-09-19 restoration scout) plus llama_state_seq_* APIs. Treat the 15879-followups chain in hunt2 with the
  same suspicion (numbers unverifiable to the described claims; "not verified").
- **Marconi** (arXiv 2411.19379, fetched): FIRST prefix caching for hybrid attn+SSM. Rules: cache SSM state only at
  points where BOTH attention-prefix AND recurrent state will be reused; chooses boundaries via SPECULATIVE
  insertions — prefill trials using radix-tree statistics of past requests decide what SSM states are worth keeping.
  Claims up to 4.9x TTFT reduction [paper claim, A100-class]. For our multi-turn server: our GDN state cost dominates
  any context-checkpoint store (9.7 MB/token — measured by us, prior scouting); Marconi's boundary-selectivity is
  what would make checkpoint-per-turn AFFORDABLE on 16 GB RAM. Applicability 8/10 to the proxy/ctx1 direction.
- **SGLang hybrid support** (pytorch.org blog "Hybrid Models Meet SGLang", fetched via search snippet): SSM states
  updated IN PLACE ⇒ a running request's prefixes are not recoverable — same trap our llama.cpp slotting works
  around by explicit seq snapshots; SGLang caches states selectively at request boundaries. **SGLang Unified Radix
  Cache** (lmsys blog 2026-08-11, fetched via search): one radix tree, per-cache-kind TreeComponents (KV vs SSM),
  matching/splitting/locking unified [eng blog claim].
- **vLLM RFC #55697** "[RFC]: Application-Directed Prefix Checkpoints for Mamba/Hybrid models (Qwen3.5, Nemotron-4,
  Jamba)" — surfaced by search (github), body not fetched → existence noted, details not verified. Direction
  (app declares semantic prefix boundaries via input-side control) = exactly our ctxproxy --turn-boundary marking.

Bottom: nobody has solved turn-level state caching on consumer HW; SGLang/vLLM do it on servers with 80 GB of HBM.
Our GDN snapshot cost is THE blocker for cheap multi-turn resumes; Marconi-style boundary caching is the mapped
solution shape.

## Q4 — Speculative decoding on MoE, bandwidth-starved

- **arXiv 2607.12696 "Less Experts, Faster Decoding: Cost-Aware ..." (Jul 2026; fetched via arXiv HTML + HF papers
  snippet)**: THE mechanism the question asks about: *"Verification latency scales linearly with the number of active
  experts"*; confidence-driven draft selection EXPANDS the per-step union of activated experts ("expert scattering")
  inflating weight traffic during verify [paper claim; figure quoted directly]. Their remedy: cost-aware tree shaping
  minimizing marginal expert count per added draft token. For us: our MTP(2) verification batch of ≤3 tokens routes
  ≤24 experts/layer union vs 8 serial — with GPU cache miss cost dominating, worst-case verify can exceed serial
  decode cost on cold unions. Concrete: gate branch acceptance by predicted union size from expert_log history
  (cheap predictor: parent-token→expert-set map), i.e., prune a branch whose marginal unique experts exceed its
  probability × saved-time [applicability 8/10 — direct fork change; we already own acceptance telemetry].
- **SpecMoE** (Bang & Cho; semanticscholar page fetched via search): batched MoE verify with per-layer self-speculated
  micro-drafts; introduces "broadcast hazard" (idle expert copies) motivating per-layer expert pools — batch-oriented,
  less directly applicable [paper claim, A100 dual, batch≥8]. Existence + concept verified; numbers not verified.
- Trees vs chains: the linear-in-unique-experts result implies wide trees are EXTRA costly on our box (chain of 2
  already tested neutral-positive; branching multiplier tuned here: keep n≤2, prefer depth over width). Chain-vs-tree
  empirical llama.cpp numbers on consumer MoE: not found in search (gap, not verified).

## Q5 — Other outside-the-box, 4 GB VRAM / 16 GB RAM

- **Two-conversation decode batching**: no fetched source inside window proposes exactly this on a single 4 GB card;
  nearest: vLLM lineage + our internal 2026-09-19 analysis (measure-grounded locally). Marked: local analysis, S-M
  effort, exact; biggest throughput lever, worsens per-token latency for interactive session #1 (priority inversion
  to manage).
- **NVMe**: irrelevant here (11 GB experts fit pinned RAM); nothing credible sought for this box class — honest
  blank, aligned with prior scouting ("no NN-based NVMe predictor worth pursuing").
- **CPU-side leverage**: AVX2 VNNI absent on Comet Lake (dp4a yes — per setup facts) — KTransformers-family INT8
  expert kernels lean on AMX/VNNI server features; honest expectation modest on i5-10400F (detail not verified —
  kernels repo not fetched this session). The bigger CPU lever is pipelining (HybriMoE), not kernel speed.
- **Routing-aware placement**: covered by SeqMoE/HybriMoE/MoE-Infinity rows (predicted-placement tables computed
  OFFLINE from our expert_log.jsonl — we HAVE 6-week traces; that corpus is unusually valuable as prefetch-training
  data; no external system offers that shortcut to anyone else on this box).
- **Exact layer skipping**: nothing fetched qualifies as exact skipping; nearest honest item = MoE expert-skip
  (top-p routing threshold) which is lossy by construction and upstream shows regressions below 0.75 (prior hunt,
  verified then). Left out of ranking deliberately.
- **Display/iGPU-less CPU**: with no iGPU, nothing offloads elsewhere; GPU busy-waits cost clocks only — pinning
  already tried per setup. Blank by evidence.

## Verifications ledger (fetched this session)
abs/export API: 2402.07069(control, unrelated RL paper — checked against Fiddler-ID confusion), 2402.07033,
2406.06282, 2504.05897, 2401.14361(+repo desc), 2411.11217, 2411.19379, 2607.12696(html/snippet), 2609.12978(snippet),
2309.06180(Paged), 2311.01282(FD++), SpecMoE scholar page. GitHub API: pulls 15293(body), 16607(title!), issues
search incl. 28761(full body), 24670(full body — Turing MTP-not-activating report on IQ1_M + n_rs_seq asymmetry in
draft ctx; NOTE: our box DOES run MTP — discrepancy worth knowing: draft-context SSM state gap surfaced upstream),
26404(PR body), 28037(issue). Blogs/repos via search-result pages: efeslab/fiddler, EfficientMoE/MoE-Infinity,
PKU-SEC-Lab/HybriMoE, lmsys unified radix (Aug 11), pytorch SGLang-hybrid, vLLM RFC 55697 listing, docs.vllm.ai APC.

OUTSIDE_BOX_DONE
