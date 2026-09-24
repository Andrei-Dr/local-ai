# MTP Speculation Code Path Report

Source: `llama.cpp (tu116-served f5ddca176)` (llama.cpp worktree, branch `tu116-served`).  
Architecture: Qwen3.6-35B-A3B (`qwen35moe`): 40 layers (30 Gated DeltaNet recurrent + 10 GQA attention), 256 experts top-8 + shared, served with `--spec-type draft-mtp`.

---

## 1. Full Speculative Loop

### 1.1 Draft Generation (MTP)

**Implementation**: `common_speculative_impl_draft_mtp` (`common/speculative.cpp:1330-1767`).

**Three modes** (set once in the constructor, `speculative.cpp:1342-1348`):
- `is_mem_shared` (Gemma4): shares target KV, all heads in one graph.
- `chain_heads` (step35): `n_mtp_layers > 1 && !is_mem_shared`. Multiple trained heads, one per draft step. Each step selects a different decoder layer via `llama_set_nextn_layer_offset(ctx_dft, i)` (`speculative.cpp:1650`).
- **Neither** (our config, qwen35/qwen35moe): a single trained MTP head. `n_mtp_layers == 1`, `chain_heads == false`, `is_mem_shared == false`.

**Draft generation is autoregressive per-depth** (`speculative.cpp:1602-1751`):

1. **Seed step**: For each drafting sequence, build a batch with `(dp.id_last, dp.pos0)` and attach the pending hidden state `pending_h[seq_id]` as the embedding row (`speculative.cpp:1624-1625`). This hidden state is the target model's `h_nextn` output at the last verified position.

2. **Decode**: `llama_decode(ctx_dft, batch)` — runs the MTP graph (`LLM_GRAPH_TYPE_DECODER_MTP`), which is a full Qwen3.5 attention-based decoder block (NOT recurrent) with MoE FFN and an LM head (`qwen35moe.cpp:642-839`). The MTP block uses the draft context's KV cache (attention-only, no recurrent state). It produces logits and a pre-norm hidden state `h_nextn`.

3. **Sample + chain**: For each sequence, sample from the draft logits at `i_last[seq_id]` using a top-k=10 greedy sampler (`speculative.cpp:1671`). Read the draft's `h_nextn` at the same position via `llama_get_embeddings_nextn_ith(ctx_dft, i_last)` (`speculative.cpp:1672`). If `cur_p->data[0].p >= p_min`, accept the draft token and add it to the result.

4. **Next step**: Build a new batch entry with the sampled token at `pos0 + i + 1`, attaching the just-read `h_nextn` as the embedding (`speculative.cpp:1723-1724`). Decode again. Repeat until `n_max` drafts or p_min cutoff.

**The MTP head runs on whatever device the draft model is loaded to.** For our config this is the GPU (the draft head GGUF is loaded with its own `n_gpu_layers`).

**Draft n / p_min / n_min**: Controlled by `--spec-draft-max` (n_max, default 3), `--spec-draft-p-min` (p_min, default varies), `--spec-draft-min` (n_min, minimum draft length to keep). These are in `params.draft.n_max/p_min/n_min` (`speculative.cpp:1367,1380`).

### 1.2 Verify Batch Construction

In the server (`tools/server/server-context.cpp:504-542`):

1. The slot's last sampled token (`slot.sampled`) is added to the batch at `pos_next`.
2. Each draft token is added at consecutive positions `pos_next + 1, pos_next + 2, ...`.
3. All tokens request logits (`true` in the `add()` call, `server-context.cpp:532-534`).
4. `spec_i_batch` records the batch indices for each position (sampled + n_draft = n_draft + 1 entries).
5. Each token in the batch is assigned to the slot's single `seq_id`. **No multi-sequence or tree batching is done.**

The entire verify batch is decoded in one `llama_decode(ctx_tgt, batch)` call through the target model's normal graph, producing logits at every position.

### 1.3 Acceptance Decision

`common_sampler_sample_and_accept_n()` (`common/sampling.cpp:678-706`):

This is **greedy token matching**, not rejection sampling:

```
for i in 0..draft.size():
    id = common_sampler_sample(smpl, ctx, idxs[i])  // sample from TARGET logits
    accept(smpl, id)
    result.push_back(id)
    if draft[i] != id:
        break  // REJECT: target sampled a different token
if all matched:
    sample one more bonus token from target logits at position n_draft
```

The sampler uses whatever sampling parameters the user configured (temperature, top-p, etc.). Under greedy (temp=0), this is exact: the draft is accepted iff the target model's argmax matches the draft token at each position. Under sampling (temp>0), this is NOT exact rejection sampling — it's greedy match of the sampled target token against the draft. A draft token that the target would accept with high probability can still be rejected if the sampler happens to draw a different token. **(inferred) This means the acceptance is statistically unbiased (each accepted token was drawn from the target distribution) but not optimal (rejection sampling would accept more tokens).**

**Synthetic acceptance** (`server_sample_and_accept_synth`, `server-context.cpp:57-93`) is a benchmarking-only mode that accepts drafts at preconfigured rates, bypassing the real comparison.

### 1.4 KV Cache / State Rollback on Rejection

**After acceptance** (`server-context.cpp:3924-3998`):

1. Compute `n_rollback = draft.size() + 1 - accepted.size()`.

2. **Two paths depending on `ctx_tgt_seq_rm_type`**:

   a. **`COMMON_CONTEXT_SEQ_RM_TYPE_RS`** (our config — hybrid memory with recurrent state, `common.h:1019`): If `n_rollback <= llama_n_rs_seq(ctx_tgt)`, use `seq_rm(slot.id, pos_next_after_accepted, -1)` to trim the KV cache (attention layers) and roll back recurrent state (GDN layers) via the snapshot mechanism. Otherwise, fall through to checkpoint restore.

   b. **`COMMON_CONTEXT_SEQ_RM_TYPE_FULL`**: Cannot partially seq_rm. Restore from a serialized checkpoint (`spec_ckpt`), then replay the accepted tokens.

3. `common_speculative_accept(spec, slot.id, n_accepted-1)` — tells the MTP impl how many were accepted. The MTP `accept()` (`speculative.cpp:1753-1766`) picks the correct hidden state from `verify_h[seq_id]` for the next round's `pending_h`:
   ```
   i_h = min(n_accepted, verify_h_rows - 1)
   pending_h[seq_id] = verify_h[seq_id][i_h]
   ```

4. The prompt token list is truncated to remove rejected drafts, and `seq_rm` removes the rejected positions from the target memory.

---

## 2. Recurrent (Gated DeltaNet) State Handling During Speculation

### 2.1 Memory Architecture

Qwen35moe uses `llama_memory_hybrid` (`llama-memory-hybrid.h:19`), which composes:
- `llama_kv_cache` for the 10 GQA attention layers
- `llama_memory_recurrent` for the 30 Gated DeltaNet recurrent layers

Both are accessed through a unified interface. `seq_rm`, `seq_cp`, `seq_add` etc. are forwarded to both sub-memories (`llama-memory-hybrid.cpp:143-154`).

### 2.2 Recurrent State Snapshots (n_rs_seq)

The recurrent memory allocates `mem_size * (1 + n_rs_seq)` rows per layer tensor (`llama-memory-recurrent.cpp:101-103`). The extra `n_rs_seq` copies are rollback snapshots.

**How n_rs_seq is computed** (`common.h:394-419`): For MTP, `n_rs_seq = draft.n_max` (e.g., 3). For n-gram, capped at 8.

**The snapshot mechanism**:
- During a decode batch, the split logic ensures the trailing `(1 + n_rs_seq)` tokens of each sequence stay in the same ubatch (`llama-memory-recurrent.cpp:443-445`, `llama-memory-hybrid.cpp:85-89`). This means all snapshots are written atomically.
- `find_slot()` stores each cell's `src0` field: the row index (in the widened tensor) from which state should be copied (`llama-memory-recurrent.cpp:696-700`).
- `s_copy()` computes the actual source row for the graph's state-copy input: `(rs_idx[seq_id] * mem_size) + src0` (`llama-memory-recurrent.cpp:1306-1324`). An `rs_idx` of 0 means "use the current state"; an `rs_idx` of N means "roll back N tokens."

**Rollback via `seq_rm`** (`llama-memory-recurrent.cpp:193-203`):
```
if 0 < p0 && p0 <= cell.pos && p1 > cell.pos:
    rollback = cell.pos - (p0 - 1)
    if !pending && rollback >= 1 && rollback <= n_rs_seq:
        set_rs_idx(seq_id, rollback)
        cell.pos = p0 - 1
        return true
    else:
        return false  // cannot rollback beyond snapshot depth
```

This is a **per-token state snapshot rollback**, bounded by `n_rs_seq`. The `rs_idx` is consumed (reset to 0) on the next `s_copy()` call during the following decode. **The rollback is single-use**: if `rs_idx[seq_id] != 0` (a pending rollback), another rollback is rejected.

**Cost of rollback**: Zero data copy at rollback time. The `rs_idx` redirect is just an index offset applied during graph construction. The actual state read is a gather from the snapshot plane during the next forward pass. **Bytes**: no explicit copy — the graph reads from `row = rs_idx * mem_size + src0` instead of `row = src0`.

### 2.3 Can a Verify Batch Contain Multiple Sequences (Branches)?

**seq_cp support**: `llama_memory_recurrent::seq_cp()` (`llama-memory-recurrent.cpp:249-284`) copies sequence metadata (tail pointer, seq_id sets) but does NOT duplicate the recurrent state data. It adds `seq_id_dst` to the same cell's seq_id set. This means two sequences share the same physical state until one of them is decoded (at which point `find_slot` gives it its own cell).

**The critical constraint**: Recurrent state is per-cell, one cell per sequence. A branching tree where multiple branches share a prefix would need `seq_cp` to fork the recurrent state. `seq_cp` does fork the metadata cheaply, but:

1. **The state snapshot (n_rs_seq) is per-sequence, not per-branch.** Two branches from the same prefix would share the same cell until they decode different tokens. At that point, `find_slot` would separate them. But the snapshot index `rs_idx` is shared across all sequences that share a cell — you cannot have different rollback depths for different branches.

2. **n_seq_max limits**: The constructor takes `n_seq_max` (`llama-memory-recurrent.h:26`). The server typically creates this equal to `n_parallel` (number of concurrent slots). Tree branches would need additional sequence IDs.

3. **Batch splitting constraint**: The `(1 + n_rs_seq)` trailing-tokens-in-same-ubatch rule means all snapshot tokens for a sequence must be decoded together. Multiple branches from the same sequence would need careful batching.

**Bottom line**: The recurrent memory does support `seq_cp`/per-seq state such that branches could theoretically be verified in one batch, BUT:
- Each branch needs its own seq_id (limited by n_seq_max).
- Branches sharing a prefix share the physical state cell until the first divergent decode, at which point they're separated by `find_slot`.
- The snapshot rollback is bounded by `n_rs_seq` per sequence and is single-use.
- **(inferred) A tree verify with K branches would need K seq_ids, K copies of the recurrent state (forked at decode time via find_slot), and the verify batch would need to process all branches' tokens. The recurrent layers would process each branch independently (no sharing of intermediate state between branches within a single batch).**

---

## 3. Draft Vocabulary Trim Patch

**Location**: `src/models/qwen35moe.cpp:7-82` (namespace-scope `mtp_subvocab` struct and `mtp_subvocab_get()` function).

**Mechanism**:
- Enabled by `LLAMA_MTP_VOCAB_FILE=<path>` environment variable.
- The file contains int32 little-endian token IDs (e.g., the 49,152 most frequent tokens).
- At the first single-output draft graph build, the function lazily constructs:
  - `w_sub`: a gathered subset of the LM head weight — shape `[n_embd, K]`, same quantization type as the full head. Only K rows are copied from the original head.
  - `ids`: I32 tensor of length K mapping sub-index -> full token ID.
  - `logits_full`: a persistent F32 `[1, n_vocab]` buffer initialized to `-inf` everywhere.
- During the MTP draft graph (`qwen35moe.cpp:828-836`): if the batch has only 1 output token and the head has no scale/LoRA:
  ```
  sub = mul_mat(w_sub, cur)         // [K, 1] — only K rows computed
  dst = set_rows(logits_full, sub, ids)  // scatter K values into -inf buffer
  cur = reshape(dst, [n_vocab, 1])   // full-vocab logits, -inf outside ids
  ```

**What it saves**: Reads K rows instead of n_vocab rows from the (quantized) head. For K=49,152 vs n_vocab=152,064, this is a ~3x reduction in head matmul. The head is read at approximately VRAM roofline, so the cost scales linearly with K.

**Mapping back to full IDs**: The sampler sees full-vocab logits. Tokens outside the sub-vocabulary have logits = `-inf`, so they can never be drafted. The `ids` tensor is only used internally for the `set_rows` scatter; sampled token IDs are already in the full vocabulary space.

**The TARGET always verifies against its full vocabulary** — the trim only affects which tokens the draft can propose, not the verification distribution.

---

## 4. Hooking Draft Probabilities

### 4.1 Where Draft Top-k Probs Are Available

At each draft depth `i`, after `llama_decode(ctx_dft, batch)`:

1. `common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true)` is called (`speculative.cpp:1671`).
2. `common_sampler_get_candidates(smpl, true)` returns `cur_p` — the sorted candidate list with probabilities (`speculative.cpp:1674`).
3. The top candidates are already logged at TRACE level (`speculative.cpp:1676-1679`):
   ```
   SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", ...)
   ```

**Cheapest hook**: After line `speculative.cpp:1674`, `cur_p->data[k].id` and `cur_p->data[k].p` give the draft head's top-k tokens and probabilities at draft depth `i`. The sampler's `cur_p` is already sorted by probability.

### 4.2 Where Target Top-k at the Same Position Are Available

During the verify batch, the target model produces logits at every position in `spec_i_batch`. After `llama_decode(ctx_tgt, batch)` in the server's sampling phase, `common_sampler_sample(smpl, ctx_tgt, spec_i_batch[i])` is called for each position.

**Cheapest hook for an offline acceptance simulator**: In `common_sampler_sample_and_accept_n()` (`common/sampling.cpp:678-706`), after sampling at each `idxs[i]`, call `common_sampler_get_candidates()` to get the target's sorted distribution. Pair this with the draft's `cur_p` recorded during drafting. Both are in the same token-ID space.

---

## 5. Expert Routing Visibility During Verify

### 5.1 Where Routing Is Computed

In `build_moe_ffn()` (`llama-graph.cpp:2126-2131`):
```cpp
selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
// [n_expert_used, n_tokens]
cb(selected_experts, "ffn_moe_topk", il);
```

This tensor is created in the compute graph. It holds `n_expert_used` (8) expert indices per token per layer.

### 5.2 How to Record Expert IDs for Non-Accepted Branch Tokens

The `selected_experts` tensor exists only within the compute graph and is consumed by `ggml_mul_mat_id`. It is NOT normally exported to the host after the graph runs.

**To record expert IDs during a verify batch**:
1. Register a callback via `cb(selected_experts, "ffn_moe_topk", il)` — the callback system already tags this tensor with a name.
2. Use `ggml_backend_tensor_get()` after `llama_synchronize()` to read the tensor data. The tensor has shape `[n_expert_used, n_tokens]` with I32 type.
3. For tokens at positions beyond the acceptance cutoff, the expert IDs tell you which experts that branch would have activated.

### 5.3 MoE Cache Warm Observation Path

The `llama_moe_cache_build_warm_obs()` function (`llama-moecache.h:102`) already receives `selected_experts` for batches >= `warm` tokens. For small verify batches (1-4 tokens per slot), the warm observation path is skipped (`llama-graph.cpp:2117`: only fires when `n_tokens >= 1 && n_tokens <= 4` — this is actually the cache-aware routing path, not the warm path).

**(inferred) The warm observation path fires for batches of >= 32 tokens. A verify batch with n_draft=3 has only 4 tokens per slot, so it does NOT trigger the warm path. The expert routing is still computed but only consumed by the graph internally.**

---

## 6. Tree Attention / Custom Mask Support

### 6.1 Current KQ Mask Construction

The KQ mask is built per-batch in `llama-graph.cpp:31-65` (`build_attn_inp_kq_mask`), with shape `[n_kv, n_tokens/n_stream, 1, n_stream]`. It is filled by `llm_graph_input_attn_kv_unified::set_input()` (`llama-graph.cpp:453-479`).

The mask is **causal by default**: each token can attend to all KV cache positions at or before its own position in the same sequence. There is NO explicit "tree mask" or custom per-token mask override in the current batch API. The mask is derived entirely from `(seq_id, pos)` pairs in the batch.

### 6.2 Existing Tree/EAGLE3 Mask Support

EAGLE3 (`src/models/eagle3.cpp`) does NOT implement tree-structured attention. It is a single-layer decoder that processes tokens sequentially within each sequence. The draft loop is linear (one candidate per depth), not branched.

**No existing tree attention infrastructure** was found in the codebase. There is no mechanism to specify a custom attention mask per-token in a batch. The kq_mask is always derived from the causal seq_id/pos structure.

### 6.3 Why Recurrent Layers Break Pure-Mask Tree Approach

A tree verify with pure attention masking would work like this: place all branch tokens in one batch at the same position with different seq_ids, use a custom mask to make each branch attend only to the shared prefix + its own path. Attention layers support this via `seq_cp` (fork the KV) + per-seq masking.

**The recurrent layers (30 of 40 layers for qwen35moe) fundamentally break this approach:**

1. **Recurrent state is sequential, not positional.** GDN state at token N depends on ALL previous tokens 0..N-1 in order. There is no "mask" that can make a recurrent layer compute multiple branches simultaneously — each branch must process its own token sequence through the recurrent state transition function.

2. **No parallel-branch recurrent processing.** The recurrent memory stores ONE state per (cell, seq_id). To process K branches sharing a prefix of length P, you need:
   - Fork the recurrent state at position P (via `seq_cp`, which is metadata-only — it shares the same physical cell until decode).
   - Decode each branch's divergent tokens through the recurrent layers independently, which creates K separate state cells.
   - This is K serial recurrent decodes (or one batched decode with K seqs, each advancing one token), NOT one batched decode with a custom mask.

3. **Snapshot depth limits branching.** With `n_rs_seq = 3`, you can roll back at most 3 tokens. A tree with depth > 3 cannot be fully rolled back. Each branch that gets rejected must roll back independently.

4. **Cost scaling.** For a tree with B branches of depth D:
   - Attention layers: one batched decode of B*D tokens (with tree mask) — O(1) decodes.
   - Recurrent layers: B separate state transitions of D tokens each — O(B) sequential decodes of the recurrent state (or B*D tokens in one batch, but each branch still processes independently through the state transition).
   - The 30 recurrent layers dominate compute. The tree verify would save little vs. B separate linear verifies.

**The only viable tree approach for a hybrid recurrent/attention model** would be to:
1. Fork the recurrent state at the prefix endpoint.
2. Batch all branches' tokens in one decode, with each branch having its own seq_id.
3. The attention layers see the shared KV prefix + branch-specific tokens (via seq_cp + causal mask per seq).
4. The recurrent layers process each branch's tokens against its own forked state cell.

This is architecturally possible with the current `seq_cp` + batch mechanism, but it is NOT a "tree mask" optimization — it is B parallel linear decodes that happen to share the attention prefix. The recurrent layers cannot be shared across branches.

---

## Summary of Key Findings

| Question | Answer |
|----------|--------|
| MTP draft: autoregressive per-depth? | Yes, one token per depth, chained via `h_nextn` hidden state |
| What feeds depth d? | `pending_h` = target's `h_nextn` at last verified position, then draft's own `h_nextn` at each subsequent step |
| MTP head on GPU? | Yes, runs on whatever device the draft model is loaded to |
| Acceptance method | Greedy token match: sample from target, compare to draft. NOT exact rejection sampling |
| Recurrent rollback | Per-token snapshot via `n_rs_seq` planes, O(0) copy cost, bounded by `n_rs_seq` (=draft n_max), single-use |
| Multi-branch verify? | Architecturally possible via seq_cp, but recurrent layers process branches independently — no mask-based sharing |
| n_seq_max constraint | Set at context creation time, equals n_parallel. Tree branches would need additional seq IDs |
| Vocab trim | LLAMA_MTP_VOCAB_FILE restricts draft head to K tokens; target still verifies full vocab |
| Tree attention support | None. No custom per-token mask API exists. Recurrent layers fundamentally prevent mask-based tree verify |
