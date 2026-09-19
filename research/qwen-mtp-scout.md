# Qwen3.6-35B-A3B MTP head scout

Date: 2026-09-19. Read-only: HF API, HF range requests (GGUF headers only), GitHub API / raw source.
Tags: **VERIFIED** = observed in header bytes / source / API output this session. **INFERRED** = derived, not observed.

Scanner used: `/tmp/mtpscan/mtpscan.py` (wraps `research/scripts/ggufscan/scan.py`, adds absolute byte offsets
+ per-tensor listing; `~/dev/.venv/bin/python /tmp/mtpscan/mtpscan.py -v <url>`). Absolute offset = aligned
header end + tensor offset; length = next tensor offset - this offset (includes alignment padding).
Source snapshots: `/tmp/mtpscan/src/{main,fork}/` (mainline master, fork @ 9a9394a895b96003ca842a6041cb28ac49a108f7).

## 0. Bottom line

1. **An uncensored <= 12.6 GB GGUF with the MTP head already inside exists**: `SassyDiffusion/Qwen3.6-35B-A3B-heretic-MTP-GGUF`
   `Qwen3.6-35B-A3B-heretic.IQ2_M.gguf` = 11,882,973,120 B, `block_count=41`, `nextn_predict_layers=1`, 20 `blk.40.*` tensors (VERIFIED, header).
   It is heretic (llmfan46 "Native-MTP-Preserved"), not the HauhauCS trunk.
2. **Graft onto the current HauhauCS IQ2_M trunk is a 322,557,952 B range fetch**: blk.40 is the contiguous tail of
   `unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (bytes 11560411424-11882969375). Result ~11.98 GB (INFERRED sum).
3. **No graft is strictly required**: fork @ 9a9394a has an `mtp_only` load path, so `-md mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp`
   (ggml-org standalone head, 1,060,038,432 B) should load as-is (VERIFIED in source, not run). Cost: a second `output.weight` (286 MB Q4_0) (INFERRED VRAM hit).
4. **Fork @ 9a9394a has full qwen35moe MTP graph support** (VERIFIED). Mainline added it in PR #22673, merged 2026-05-16, ~3 months before the 2026-08-25 merge-base.
5. **Issue #24670 is this exact GPU** (GTX 1650 SUPER 4 GB, Qwen3.6-35B-A3B UD-IQ1_M): draft-mtp "never activates" with `--spec-draft-p-min 0.75`;
   a commenter reports drafts only generate with p-min 0.0. Fork default p-min is 0.0 (VERIFIED). **Do not set `--spec-draft-p-min`.**

Recommended order: graft (2) is the target end state (keeps the vetted trunk, shares `output.weight`/`token_embd`, +31 MB non-expert).
Run (3) or (1) first only as the falsification test for "does MTP pay at all on sm_75 with exps=CPU" — #24670 makes that a live question, and it costs one download and zero file surgery.

## 1. Which GGUFs actually carry blk.40 (header-verified)

All rows VERIFIED by range-fetching the header. Every hit has exactly the same 20 tensors (list in 1.1), arch `qwen35moe`,
`qwen35moe.block_count = 41`, `qwen35moe.nextn_predict_layers = 1`, 753 tensors total (733 trunk + 20 MTP).
Non-MTP files have `block_count = 40`, no nextn key, 733 tensors, max blk index 39.

| repo / file | total bytes | MTP bytes | expert bytes | non-expert bytes |
|---|---:|---:|---:|---:|
| **Standalone MTP-only files** | | | | |
| ggml-org/Qwen3.6-35B-A3B-GGUF `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` (23 tensors: +output, output_norm, token_embd) | 1,060,038,432 | 476,956,672 | 452,984,832 | 23,971,840 |
| ggml-org/Qwen3.6-35B-A3B-GGUF `mtp-Qwen3.6-35B-A3B-Q8_0.gguf` | 1,990,649,632 | 899,008,512 | 855,638,016 | 43,370,496 |
| ggml-org/Qwen3.6-35B-A3B-GGUF `mtp-Qwen3.6-35B-A3B-BF16.gguf` | 3,735,545,632 | not scanned | | |
| bartowski/Qwen_Qwen3.6-35B-A3B-GGUF `mtp-Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` (23 tensors) | 1,190,098,816 | 475,904,000 | 452,984,832 | 22,919,168 |
| lym00/Qwen3.6-35B-A3B-MTP-ONLY-GGUF `Qwen3.6-35B-A3B-MTP-q4_0.gguf` (23 tensors) | 1,190,098,592 | 475,904,000 | 452,984,832 | 22,919,168 |
| a4lg/Qwen3.6-35B-A3B-MTP-ONLY-GGUF `...-MTP-ONLY-Q4_K_M.gguf` (23 tensors) | 1,621,550,880 | 529,909,760 | 486,539,264 | 43,370,496 |
| tomngdev/MTP-Qwen3.6-35B-A3B-GGUF `mtp-Q4_K_M.gguf` (23 tensors) | 1,621,550,976 | 529,909,760 | 486,539,264 | 43,370,496 |
| daanbanaan/...Genesis-Hermes-V7-GGUF-MTP-merge `35B-A3B-MTP.gguf` (**20 tensors, blk.40 only**, Q8_0, 3 KVs) | 897,957,280 | 897,955,840 | 855,638,016 | 42,317,824 |
| havenoammo/Qwen3.6-35B-A3B-MTP-GGUF `35BA3B-MTP.gguf` (**20 tensors, blk.40 only**, Q8_0) | 903,319,200 | 897,462,400 | 855,638,016 | 41,824,384 |
| **Full stock quants with MTP** | | | | |
| unsloth/Qwen3.6-35B-A3B-MTP-GGUF `UD-IQ1_M` | 11,366,414,624 | 322,557,952 | 291,504,128 | 31,053,824 |
| unsloth/Qwen3.6-35B-A3B-MTP-GGUF `UD-IQ2_XXS` | 11,819,399,456 | 322,557,952 | 291,504,128 | 31,053,824 |
| unsloth/Qwen3.6-35B-A3B-MTP-GGUF `UD-IQ2_M` | 11,882,969,376 | 322,557,952 | 291,504,128 | 31,053,824 |
| unsloth/Qwen3.6-35B-A3B-MTP-GGUF `UD-Q2_K_XL` | 12,574,128,416 | 322,836,480 | 291,504,128 | 31,332,352 |
| bartowski/Qwen_Qwen3.6-35B-A3B-GGUF `IQ2_XS` (MTP block kept at Q8_0) | 11,694,831,232 | 897,955,840 | 855,638,016 | 42,317,824 |
| mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF `APEX-MTP-I-Nano` | 11,686,645,344 | 369,342,464 | 346,030,080 | 23,312,384 |
| byteshape/Qwen3.6-35B-A3B-MTP-GGUF `IQ2_S-2.25bpw` (**smallest full file with MTP**) | 10,016,977,088 | 594,589,696 | 553,648,128 | 40,941,568 |
| ggml-org/Qwen3.6-35B-A3B-MTP-GGUF `MTP-Q8_0` | 37,801,096,544 | 897,955,840 | 855,638,016 | 42,317,824 |
| **No MTP (confirmed zero nextn tensors)** | | | | |
| HauhauCS/...-Aggressive `IQ2_M` (our file) | 11,659,235,456 | 0 | | |
| unsloth/Qwen3.6-35B-A3B-GGUF `UD-IQ2_M` (non-MTP repo) | 11,522,702,304 | 0 | | |
| mradermacher/Qwen3.6-35B-A3B-i1-GGUF `i1-IQ2_M` | 11,659,236,064 | 0 | | |
| ggml-org/Qwen3.6-35B-A3B-GGUF `Q4_K_M` | 20,419,565,568 | 0 | | |
| BlueBackup (=Loswen) heretic `IQ2_M` / `IQ2_M_HQ` | 11,659,239,456 / 12,326,723,616 | 0 | | |

Not checked: `Qwen/Qwen3.6-35B-A3B-GGUF` — HF API returns `Invalid username or password` (repo absent or gated) (VERIFIED response; existence INFERRED unknown).
HauhauCS's own repo has no MTP file in any quant (VERIFIED file listing: 11 quants + mmproj, none tagged MTP; IQ2_M header-verified zero).

### 1.1 The MTP block structure (identical across all hits)

VERIFIED. **The MTP block is itself a full MoE layer: 256 experts, top-8, expert FFN 512, plus shared expert, plus full (non-GDN) gated attention.**
~95% of its bytes are routed experts. There is **no** `nextn.embed_tokens` and **no** `nextn.shared_head_head` in any GGUF scanned — the head reuses the trunk's `token_embd` and `output`.

```
blk.40.attn_norm.weight            [2048]            F32
blk.40.attn_q.weight               [2048, 8192]      (Q + gate, 2x)
blk.40.attn_k.weight               [2048, 512]
blk.40.attn_v.weight               [2048, 512]
blk.40.attn_q_norm.weight          [256]             F32
blk.40.attn_k_norm.weight          [256]             F32
blk.40.attn_output.weight          [4096, 2048]
blk.40.post_attention_norm.weight  [2048]            F32
blk.40.ffn_gate_inp.weight         [2048, 256]       BF16 or F32
blk.40.ffn_gate_exps.weight        [2048, 512, 256]
blk.40.ffn_up_exps.weight          [2048, 512, 256]
blk.40.ffn_down_exps.weight        [512, 2048, 256]
blk.40.ffn_gate_inp_shexp.weight   [2048]            BF16 or F32
blk.40.ffn_gate_shexp.weight       [2048, 512]
blk.40.ffn_up_shexp.weight         [2048, 512]
blk.40.ffn_down_shexp.weight       [512, 2048]
blk.40.nextn.eh_proj.weight        [4096, 2048]
blk.40.nextn.enorm.weight          [2048]            F32
blk.40.nextn.hnorm.weight          [2048]            F32
blk.40.nextn.shared_head_norm.weight [2048]          F32
```

Placement (INFERRED from the numbers above): `-ot "exps=CPU"` matches `blk.40.ffn_{gate,up,down}_exps` too, so experts land on CPU automatically.
GPU cost of the head = non-expert bytes (31 MB at unsloth UD-IQ2_M, 23-24 MB at Q4_0, 42 MB at Q8_0) + one full-attention layer of KV
(2 KV heads x 256 x K+V x f16 = 2,048 B/token -> ~64 MB at 32k ctx) + a draft compute buffer. Active expert bytes per drafted token = 8/256 x 291.5 MB = ~9.1 MB (IQ2_M head).

### 1.2 unsloth UD-IQ2_M blk.40 per-tensor (the graft donor; contiguous file tail)

VERIFIED. File total 11,882,969,376; data_start 10,991,392.

| tensor | type | abs offset | length |
|---|---|---:|---:|
| blk.40.attn_k.weight | Q5_K | 11560411424 | 720896 |
| blk.40.attn_k_norm.weight | F32 | 11561132320 | 1024 |
| blk.40.attn_norm.weight | F32 | 11561133344 | 8192 |
| blk.40.attn_output.weight | Q5_K | 11561141536 | 5767168 |
| blk.40.attn_q.weight | Q5_K | 11566908704 | 11534336 |
| blk.40.attn_q_norm.weight | F32 | 11578443040 | 1024 |
| blk.40.attn_v.weight | Q5_K | 11578444064 | 720896 |
| blk.40.ffn_down_exps.weight | Q3_K | 11579164960 | 115343360 |
| blk.40.ffn_down_shexp.weight | Q6_K | 11694508320 | 860160 |
| blk.40.ffn_gate_exps.weight | Q2_K | 11695368480 | 88080384 |
| blk.40.ffn_gate_inp.weight | BF16 | 11783448864 | 1048576 |
| blk.40.ffn_gate_inp_shexp.weight | BF16 | 11784497440 | 4096 |
| blk.40.ffn_gate_shexp.weight | Q5_K | 11784501536 | 720896 |
| blk.40.ffn_up_exps.weight | Q2_K | 11785222432 | 88080384 |
| blk.40.ffn_up_shexp.weight | Q5_K | 11873302816 | 720896 |
| blk.40.nextn.eh_proj.weight | Q8_0 | 11874023712 | 8912896 |
| blk.40.nextn.enorm.weight | F32 | 11882936608 | 8192 |
| blk.40.nextn.hnorm.weight | F32 | 11882944800 | 8192 |
| blk.40.nextn.shared_head_norm.weight | F32 | 11882952992 | 8192 |
| blk.40.post_attention_norm.weight | F32 | 11882961184 | 8192 |

Sum = 322,557,952 = 11,882,969,376 - 11,560,411,424. The block is contiguous and runs to EOF.
SassyDiffusion heretic IQ2_M has a byte-for-byte same-shaped tail (same types, same lengths) at 11560415168-11882973119 (VERIFIED sizes/types; identical *values* INFERRED — it is a requant of the heretic BF16, whose MTP head may or may not be stock).

## 2. Abliterated / uncensored <= 12.6 GB with MTP already inside

All VERIFIED by header (`block_count=41`, `nextn_predict_layers=1`, 20 blk.40 tensors):

| file | bytes | MTP bytes | note |
|---|---:|---:|---|
| SassyDiffusion/Qwen3.6-35B-A3B-heretic-MTP-GGUF `heretic.IQ1_M.gguf` | 11,366,418,368 | 322,557,952 | |
| SassyDiffusion/... `heretic.IQ2_XXS.gguf` | 11,819,403,200 | 322,557,952 | |
| **SassyDiffusion/... `heretic.IQ2_M.gguf`** | **11,882,973,120** | 322,557,952 | closest match to current quant class |
| SassyDiffusion/... `heretic.EMB_BF16.IQ2_M.gguf` | 12,550,457,280 | 322,557,952 | BF16 embeddings variant |
| SassyDiffusion/... `heretic.Q2_K_XL.gguf` | 12,574,132,160 | 322,836,480 | just under the 12.6 GB line |

Header metadata of the Sassy files: `general.base_model.0.repo_url = https://huggingface.co/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved`,
`quantize.imatrix.file = imatrix_unsloth.gguf_file` (VERIFIED). File sizes mirror unsloth's UD recipe within ~4 KB, i.e. unsloth's dynamic quant recipe applied to the heretic weights (INFERRED).
Note Sassy `file_type=19` (IQ2_XXS) for both "IQ2_M" and "IQ2_XXS" names and both are different sizes — naming follows unsloth's UD labels, not llama.cpp ftype (VERIFIED values; interpretation INFERRED).

Over the limit (VERIFIED sizes + headers): huihui-ai abliterated-MTP Q2_K 13,246,129,376; mradermacher heretic-Native-MTP-Preserved i1-Q2_K 13,246,134,816;
SGLabs Pym-Q2-MTP 13,222,476,000; SC117 heretic APEX-I-MINI 14,273,277,376; morikomorizz HauhauCS-MTP Q2_K_P 15,879,222,784 (smallest HauhauCS+MTP anywhere);
wang-yang HauhauCS-Aggressive Q6_K_P-MTP 31,352,239,648; Black-Engineer APEX-I-Mini-MTP 15,214,029,952.

**There is no HauhauCS-trunk GGUF <= 12.6 GB with MTP.** The morikomorizz/wang-yang HauhauCS-MTP files carry an 897,955,840 B Q8_0 head — the exact size of the stock Q8_0 head in bartowski/ggml-org — so they are stock-head grafts (INFERRED from byte-identical size), which is direct precedent for grafting onto this trunk.

## 3. Smallest download for a usable head

| option | download | what you get |
|---|---:|---|
| Range-fetch blk.40 tail of unsloth UD-IQ2_M | **322,557,952 B** | raw tensor bytes, 2-3 bit experts / Q5_K attn / Q8_0 eh_proj, imatrix-quantized |
| Range-fetch blk.40 tail of ggml-org `mtp-...-Q4_0.gguf` | 476,956,672 B | Q4_0 everything (no imatrix) |
| Whole ggml-org `mtp-Qwen3.6-35B-A3B-Q4_0.gguf` | 1,060,038,432 B | loadable standalone draft file (token_embd + output + output_norm + blk.40) |
| Whole daanbanaan `35B-A3B-MTP.gguf` / havenoammo `35BA3B-MTP.gguf` | 897,957,280 / 903,319,200 B | blk.40-only Q8_0 GGUF with valid header — feeds a graft script directly |

HF serves range requests on `resolve/main` (VERIFIED: every scan above is a `Range: bytes=0-N` fetch answered with `Content-Range`). `curl -L` is required (302 to the CDN).

```sh
# donor header (tensor infos, to rebuild names/dims/types) — first 11 MB is enough (data_start=10,991,392)
curl -L -r 0-10991391 -o unsloth-udiq2m.header.bin \
  https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
# blk.40 tensor data, contiguous to EOF
curl -L -r 11560411424-11882969375 -o unsloth-udiq2m.blk40.bin \
  https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
# alternative, Q4_0 head from the ggml-org standalone file
curl -L -r 583081760-1060038431 -o ggmlorg-q4_0.blk40.bin \
  https://huggingface.co/ggml-org/Qwen3.6-35B-A3B-GGUF/resolve/main/mtp-Qwen3.6-35B-A3B-Q4_0.gguf
```

Offsets are for the file revisions live on 2026-09-19; re-run the scanner before fetching (unsloth re-uploads). Pin with `resolve/<commit-sha>/` for reproducibility.

Graft compatibility, trunk vs donor (VERIFIED headers): both `general.alignment` default 32; both `token_embd`/`output` are `[2048, 248320]`; both use **separate** `ffn_gate_exps`/`ffn_up_exps` (not fused `gate_up_exps`);
identical blk.39 tensor names and dims. HauhauCS trunk: `token_embd` IQ3_S, `output` Q5_K. Graft = append 20 tensor infos + 322.6 MB, set `qwen35moe.block_count` 40 -> 41, add `qwen35moe.nextn_predict_layers = 1` (uint32).
Grafted size = 11,659,235,456 + 322,557,952 + header growth ≈ 11.98 GB (INFERRED).

Whether a stock head drafts well against the HauhauCS-modified trunk is unmeasured. Precedent: BoldingBuilds stock head on abliterated trunk = 0.643 mean acceptance at n-max 2 (their card; section 5). The head consumes the trunk's final hidden state via `eh_proj`, so acceptance tracks how far abliteration + IQ2 quantization moved that state (INFERRED).

## 4. How llama.cpp loads MTP for qwen35moe

### 4.1 Fork @ 9a9394a — has it

VERIFIED: commit `9a9394a895b96003ca842a6041cb28ac49a108f7`, 2026-09-18, "build: fix Ubuntu ARM64 and Windows Vulkan release jobs (#198)".
GitHub compare vs ggml-org master: ahead 93, behind 433, merge-base `5ea87ddad22541a37053c7ba92b02ec1923617c6` dated 2026-08-25T05:08:06Z.

`src/models/qwen35moe.cpp` in the fork (VERIFIED, line numbers from that file):
- L19 `ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);` — key `qwen35moe.nextn_predict_layers`, optional. `block_count` must already include the MTP block (41); `hparams.n_layer()` = 40.
- L28 recurrent-layer marking: `is_recr[i] = (i < n_layer()) && ((i+1) % full_attention_interval != 0)` — the MTP block is always full attention, never GDN.
- L43 `mtp_only = n_layer_nextn > 0 && ml.get_weight("blk.0.attn_norm.weight") == nullptr` -> trunk tensors become `TENSOR_NOT_REQUIRED`. **This is the standalone-MTP-GGUF path.**
- L45 `mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0` — MTP tensors are only loaded when `--spec-type draft-mtp` is set (`common.cpp` L1711 sets `mparams.load_mtp`). Without the flag, a grafted file costs nothing at runtime.
- L110-143 `load_block_mtp`: **required** = attn_norm, post_attention_norm, attn_q/k/v (or fused qkv), attn_output, attn_q_norm, attn_k_norm, ffn_gate_inp, ffn_down_exps, gate/up exps, ffn_gate_inp_shexp, ffn_{gate,up,down}_shexp, `nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`.
  **Optional** (`TENSOR_NOT_REQUIRED`) = `nextn.embed_tokens`, `nextn.shared_head_head`, `nextn.shared_head_norm`.
- L154 `gtype == LLM_GRAPH_TYPE_DECODER_MTP` -> `graph_mtp`. L557 asserts exactly one MTP block. L586 embeddings fall back to `model.tok_embd`; L722-735 head norm falls back to `output_norm`, head falls back to `model.output`.
- Graph: `concat(enorm(tok_embd), hnorm(h))` -> `eh_proj` -> gated full attention with multi-section RoPE -> MoE FFN + sigmoid-gated shared expert -> `shared_head_norm` -> `output`.

All 20 tensors in every scanned GGUF match the required + `shared_head_norm` set exactly. Tensor name strings come from `LLM_TENSOR_NEXTN_*` in `llama-arch.cpp`: `blk.%d.nextn.{eh_proj,embed_tokens,enorm,hnorm,shared_head_head,shared_head_norm}`.

### 4.2 KV: separate, not shared

VERIFIED in fork `src/llama-model.cpp` L2722-2725 and L2820-2822: for `LLAMA_CONTEXT_TYPE_MTP` on QWEN3NEXT / QWEN35 / QWEN35MOE / BAILINGMOE3 the draft context gets
"a plain attention KV cache instead of the hybrid wrapper", filtered to `il >= hparams.n_layer()`. So: own KV cache, one layer, no recurrent state (this is the `n_rs_seq = 0` that #24670 flagged — it is by design, not the bug).
`common/speculative.cpp` L2047-2054 documents three modes: `is_mem_shared` (gemma4), `chain_heads` (step35), "neither (qwen35 / qwen35moe): a single trained MTP head".
Because KV is separate, the draft context runs a catch-up decode over accepted tokens each step (L2223-2224).

Two ways in, both in `common_speculative_init_result` (fork `speculative.cpp` ~L3140-3200) (VERIFIED):
- no `-md`: "creating MTP draft context against the target model" — `llama_init_from_model(model_tgt, cparams)` with `ctx_type = MTP`. Shares all weights. **This is the grafted-file path.**
- with `-md <file>`: loads the draft file with the same mparams (`load_mtp` on), then creates the MTP context on it — the `mtp_only` standalone path. Owns its own `token_embd` + `output`.
`llama-context.cpp` L3911 refuses an MTP context when `n_layer_nextn == 0` ("model doesn't contain MTP layers").

CLI (fork `docs/speculative.md`, `common/arg.cpp`): `--spec-type draft-mtp`, `--spec-draft-n-max N` (default 3), `--spec-draft-n-min` (0), `--spec-draft-p-min` (default **0.00**, `common.h` L330),
`-md/--spec-draft-model`, `--mtp` (auto-download a sidecar head with `-hf`). Draft `backend_sampling` defaults **true** (`common.h` L332).

### 4.3 Mainline history (VERIFIED via GitHub API)

| PR | merged | what |
|---|---|---|
| #22673 "llama + spec: MTP Support" | **2026-05-16** | the PR that added qwen35 / qwen35moe / qwen3next MTP; touches `src/models/qwen35moe.cpp`; author tested "Qwen3.6 27B and Qwen3.6 35BA3B", "steady-state acceptance of around 75% with 3 draft tokens" |
| #23269 MTP clean-up | 2026-05-19 | |
| #23287 backend sampling for MTP draft | 2026-05-20 | |
| #24025 qwen35: post-norm hidden state for MTP | 2026-06-03 | |
| #24986 quant: fix quantizing moe with mtp | 2026-06-25 | |
| #26177 mtp nextn offload (`--fit` counts nextn block; "~10% tg" on Qwen3.6 35B A3B) | 2026-07-27 | |
| #26296 load MTP tensors only if used (`load_mtp`) | 2026-07-31 | |
| #27005 spec: auto-detect mtp draft model type | 2026-08-13 | |
| #27400 common: fix draft-mtp with embeddings | 2026-08-22 | |
| — merge-base 2026-08-25 — | | everything above is in the fork (INFERRED from dates; #22673/#26296 VERIFIED present in fork source) |
| #28159 load `n_layer_nextn` before `n_layer()` calls | 2026-09-01 | not in fork (INFERRED from date) |
| #28068 fix GDN normalization `max` -> `rsqrt` | 2026-09-06 | not in fork (INFERRED from date; affects trunk numerics, not MTP specifically) |
| **#28549 Enable CUDA graph for MTP draft** | **2026-09-16** | **NOT in fork** (VERIFIED: `gf_res_prev_active` occurs 7x in mainline `llama-context.cpp`, 0x in fork). PR text: on master the MTP catch-up + first draft token never use CUDA graphs and the second recaptures every iteration; fix gives "4-5%" on RTX 5090 with Qwen3.6-35B-A3B UD-Q4_K_M. On an overhead-bound 1650S the relative gain should be larger (INFERRED) — worth cherry-picking (2 files, +42/-11). |

### 4.4 BoldingBuilds graft script — not public

VERIFIED: `BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF` contains only `.gitattributes`, `README.md`, the two GGUFs, and
`0001-qwen35-mtp-hadamard-inverse.patch` (a 14-line runtime patch to `src/models/qwen35.cpp` for PrismML's Hadamard-rotated `token_embd`; **irrelevant to us** — our `token_embd` is plain IQ3_S, and the patch targets dense `qwen35.cpp`, not `qwen35moe.cpp`).
GitHub `JoshBolding` has 6 public repos (fusion360-api-field-notes, gpu-vram-quarantine, headspace-spotify, llama.cpp, second-brain-starter, shimquant) — no graft script repo. JoshBolding = BoldingBuilds is INFERRED from the team lead's note, not verified.
Card credits the recipe to decent-jawfish, ProCreations, and `github.com/sudoingX/qwen38-mtp` (that repo: README, `probe.py`, `serve_mtp.sh`, sweeps — a recipe, no graft tool) (VERIFIED listing).
Card numbers (their claim): 15 tensors `blk.64.*`, RTX 3090, no-MTP 68.8 tok/s; n-max 2 = 94.0 tok/s @ 0.643 acceptance; n-max 3 = 84.7 @ 0.520; n-max 4 = 86.3 @ 0.445. Per-prompt acceptance 0.391 (free prose) to 0.833 (reasoning); **free prose was slower than no speculation**. The "0.75" in the brief does not appear on the card; closest is 0.79-0.83 on structured prompts.

**Reusable public graft scripts that target exactly this arch (VERIFIED present, fetched to `/tmp/mtpscan/`):**
- `havenoammo/Qwen3.6-35B-A3B-MTP-GGUF/convert.py` (443 lines, 16,419 B): `python convert.py <target.gguf> <source.gguf> <output.gguf>`; copies `blk.{target_block_count}.*` from source, overrides `{arch}.block_count`, adds `{arch}.nextn_predict_layers`, preserves on-disk tensor sizes (offset-delta sizing, so padding/row-meta survive), pads to `general.alignment`. Docstring: "Tested with ik_llama.cpp GGUF Python module". Uses `gguf.GGUFReader` — needs the source as a **valid GGUF file**, so feed it a blk.40-only GGUF (daanbanaan/havenoammo, 0.9 GB Q8_0) or a whole donor quant, not a raw range blob.
- `daanbanaan/...-MTP-merge/QWEN_MTP.py` (481 lines): Colab single-cell variant of the same code (same helper functions), adds an "extract MTP-only GGUF" step; hardcodes Colab `userdata` auth — needs de-Colab-ing.
To use the 322 MB unsloth range blob instead of a 0.9 GB Q8_0 donor, the script needs a small shim that synthesizes the 20 tensor infos from the donor header fetch (names/dims/types are in section 1.2). The Q8_0 head costs +575 MB RAM vs the IQ2_M head and +11 MB VRAM; neither is decisive — acceptance difference between head quants is unmeasured.

## 5. Reported numbers and known bugs

### 5.1 Throughput / acceptance (all third-party claims; VERIFIED only as "this text exists at this URL")

- PR #27861 comment by Syugakubusei, 2026-09-01 (https://github.com/ggml-org/llama.cpp/pull/27861#issuecomment-5492079949), Qwen3.6-35B-A3B embedded MTP, host-offloaded experts, **hardware not stated**:
  "plain decode, no MTP / no expert cache: ~17.1 tok/s; MTP (--spec-draft-n-max 2) without expert cache: ~23–24 tok/s; MTP2 + expert cache: ~30–32 tok/s". That is +35-40% from MTP alone with experts on CPU.
  Same thread (ChangXiang-SCU, 2026-09-04): with experts on a weak host, 2-4 token verify batches "fall back to computing every expert on the CPU, which on a weak host is slower than not drafting at all" — the opposing data point.
- PR #22673 (author am17an): ~75% steady-state acceptance at 3 draft tokens, ">2x" on DGX Spark, fully on-GPU. Notes PP takes a hit from D2H hidden-state copies.
- PR #28549: RTX 5090, Qwen3.6-35B-A3B UD-Q4_K_M, default draft-mtp, +4-5% from CUDA-graph fix.
- PR #26177: `--fit` + MTP GGUF left layer 0 on CPU -> fused GDN disabled; fix = "~10% tg" on Qwen3.6 35B A3B. We use `-ngl 999`, so not exposed, but watch the log for `fused Gated Delta Net ... set to disabled`.
- Issue #24670 (below): GTX 1050 laptop, same model class, "draft acceptance = 1.00, ~15 t/s" vs ~6 t/s without — acceptance of exactly 1.00 is implausible as a steady state; treat as a small-sample log line (INFERRED).

No report found of Qwen3.6-35B-A3B draft-mtp on a 4 GB Turing card that *worked*. Our 28 tok/s baseline is already above every exps-on-CPU number above, which supports the overhead-bound diagnosis (INFERRED).

Expert-side cost model (INFERRED): a verify batch of n+1 tokens routes up to 8(n+1) experts per layer on CPU. At IQ2_M that is ~9 MB/layer/token of reads; at n-max 2 the CPU side does ~3x the expert work per step for ~1 + 2a accepted tokens (a = acceptance). If the box is truly GPU/overhead-bound the CPU has headroom and this is nearly free; if CPU expert time is >~40% of the step, n-max 2 at a = 0.64 breaks even or loses. Sweep n-max 1/2/3 and measure — BoldingBuilds and Syugakubusei both landed on n-max 2.

### 5.2 Bugs

- **#24670** (open, 2026-06-15) — "draft-mtp speculative decoding not activating on Turing (sm_75) with hybrid SSM+attention model (Qwen3.6-35B-A3B)". **Reporter's rig = GTX 1650 SUPER 4 GB**, UD-IQ1_M, `--n-cpu-moe 38`, `--spec-draft-n-max 2 --spec-draft-p-min 0.75`, `-ctk/-ctv q8_0`, builds 9626/9660. Symptom: initializes, `statistics draft-mtp` never appears, ~6 t/s unchanged. Reporter blames `n_rs_seq=0` in the draft context — that is by design (4.2). Only comment (2026-07-27): drafts "only work when --spec-draft-p-min is set to 0.0. Any value above 0.0 ... no draft tokens are generated." Root cause unconfirmed; plausible mechanism is backend draft sampling not populating `cur_p->data[0].p` for the `p < p_min` gate at fork `speculative.cpp` L2391 (INFERRED). Action: leave p-min at default 0.0; if no `statistics draft-mtp` line appears, that is this bug, not a bad graft.
- **#27572** (open, 2026-08-22) — acceptance collapses to exactly 0.0 under `-np N` with multi-ubatch batches. Thread converged on an H2D input write in `process_ubatch()` racing a still-running graph on single-GPU (the existing sync was guarded by `pipeline_parallel`); proposed fix `ggml_backend_sched_synchronize()` before `set_inputs`, overlapping open PR #27311. Reported also to corrupt multi-slot output with MTP off. **Use `-np 1`.**
- **#27781** (open PR, 2026-08-27, no maintainer review, bot-flagged for template) — claims `is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt` is true for every MTP context, skipping catch-up decode on qwen35. **Source contradicts this on both mainline master and the fork** (VERIFIED): `llama_context` ctor sets `cparams.ctx_other = nullptr` and only assigns it for `LLM_ARCH_GEMMA4_ASSISTANT` (and EAGLE3/DFLASH without own embeddings) — fork `llama-context.cpp` L219-236, mainline L143-160 — and `llama_get_ctx_other()` returns `cparams.ctx_other`. For qwen35moe it returns nullptr, so `is_mem_shared = false` and catch-up runs. Not a concern for us unless it changes upstream.
- **#27897** (open PR) — combining `draft-mtp` with another draft type plus `-md` crashes at startup ("failed to create MTP context"). Do not mix `draft-mtp` with `draft-dflash`/`draft-simple` + `-md`. Plain `draft-mtp` + `-md <mtp-only gguf>` is a different path (4.2) but is untested by us.
- #25144 (open PR) — MTP draft crash on vision inputs. Avoid `--mmproj` with draft-mtp until checked.

## 6. Test plan implied by the above (not executed — read-only task)

1. Falsify first, zero surgery: current trunk + `-md mtp-Qwen3.6-35B-A3B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 -np 1`, no p-min, keep `-ngl 999 -ot "exps=CPU"`. Watch for `statistics draft-mtp` and acceptance. If VRAM is short, the duplicate `output.weight` (286 MB) is the reason — go straight to step 2.
2. Graft: `convert.py HauhauCS-IQ2_M.gguf <donor> out.gguf`, donor = daanbanaan `35B-A3B-MTP.gguf` (works with the script as-is) or the unsloth IQ2_M tail via a header shim. Verify with the scanner: `block_count=41`, `nextn_predict_layers=1`, 753 tensors; then byte-identical generation vs the ungrafted file with speculation off (BoldingBuilds' check).
3. A/B against `SassyDiffusion heretic.IQ2_M` to separate "head vs HauhauCS trunk mismatch" from "MTP does not pay on this box".
4. If it pays: cherry-pick mainline #28549 into the fork (CUDA graphs for the MTP draft).
