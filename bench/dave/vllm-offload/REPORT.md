# q38fn at 1M context: KV offload to system RAM in vLLM (status for Dave, 2026-09-25)

## The problem
The 1M-context config (YaRN x4, KV 8.3e9 per GPU = 1,021,882 tokens total, `max-model-len 1010000`) sizes the GPU KV pool for
ONE max-length request. On 2026-09-24 20:43-20:48 two long requests (a ~700k compaction + another agent) sat at 84-97% KV:
7 preemptions, decode ~2 t/s for 5 minutes; smaller requests waited on "capacity". Each preemption recomputes the evicted
request from scratch. A RAM tier for the KV (vLLM native offload, 267 GB free host RAM) turns a preemption into a PCIe reload.

## Why native offload does not start on this model (vLLM 0.27, your image)
1. `distributed/kv_transfer/kv_connector/v1/offloading/config.py:60` asserts every KV group's block size is a multiple of the
   hash unit (3216 = the hybrid Mamba/attention page). One group has 8-token blocks.
2. That group is a `CircularBufferSpec` (`v1/kv_cache_interface.py:599`): one block per request holding the raw keys of the QSA
   token group still being compressed (indexer_compress_ratio = 4). It is not prefix-cacheable, and
   `offloading/scheduler.py:125` only accepts full / sliding / chunked / Mamba specs. The main attention group (a
   `UniformTypeKVCacheSpecs` of full attention + compressed MLA) would trip the same assert.

## The patch (bench/dave/vllm-offload/patched/{config.py,scheduler.py}, diff: offload-ring-exclusion.diff)
- Groups that are not prefix-cacheable (only the ring) keep their slot but get no keys and no blocks in every store, load,
  touch and lookup; the partial-tail feature is forced off when such a group exists; config.py skips the hash-divisibility
  check for them. The uniform-type group is handled by checking its member specs.
- Why it is exact: restores land on 3216-token boundaries, divisible by the compress ratio 4, so the ring's open group is
  empty there; the model reads the ring only for positions before the chunk start and only on group-closing rows
  (`models/qwen4_exp/amd/ops/qsa.py:668-695`, `common/qsa_cache.py:181/293`). The GPU prefix cache relies on the same rule
  (`v1/core/kv_cache_utils.py:1918-1928`). No `--prefix-match-unit` change is needed.
- Deployed as bind mounts into OUR container only; nothing in your image or /home/dave/rocm10 is modified.

## A finding about the server itself
It is not deterministic at temperature 0: the same prompt twice, both cold (distinct `cache_salt`), diverges at token
1 / 11 / 0 (code / prose / reasoning, 13-24k-token prompts); a GPU prefix-cache hit also differs from cold at token 0-1
(results/base1m.e0.json). So "identical token ids" cannot prove the offload exact; the test compares next-token distributions.

## Test (pre-registered, bench/dave/vllm-offload/equality_test.py, rev 4) and the result so far
E2: per probe, A and B = two cold runs (noise floor), G = GPU prefix hit, C = the same prefix restored from the RAM tier after a
GPU-only cache reset. d = KL over the top-20 next-token logprobs. PASS needs p95 and max d(G,C) <= those of d(A,B), argmax
agreement G/C >= A/B, and every C must actually load from RAM at the same boundary as G.

Partial run on the patched server (20 of 24 probes; stopped to end the outage window; results/offdev.e2.log):
| criterion | cold vs cold (noise) | RAM restore vs GPU hit |
|---|---|---|
| p95 KL (nearest-rank, n=20) | 1.35 | 0.369 |
| max KL | 1.90 | 1.61 |
| top-1 agreement | 17/20 | 18/20 |
| RAM loads / boundary | - | 527 MB (code, prose) / 816 MB (reasoning); CPU-hit tokens == GPU-hit tokens (9,648 / 19,296) |
Every criterion holds on the 20 probes: restoring from RAM is inside the server's own noise. NOT yet run: the last 4 E2 probes
and E1 (a real preemption mid-decode restored from RAM, twice, plus a control with the RAM tier wiped).

## What is left
One ~25-minute window (the test boot needs vLLM dev mode for `/reset_prefix_cache`; it is published on 127.0.0.1 only, so
LiteLLM / agents lose the server for the window):
  cd /mnt/llm-storage/localai/q38fn-1m
  OLD=q38fn-lru-1m-plain PUBLISH_HOST=127.0.0.1 EXTRA_ENV=VLLM_SERVER_DEV_MODE=1 \
    EXTRA='{"--kv-offloading-size":"64","--kv-offloading-backend":"native"}' \
    EXTRA_BINDS=<the 2 patched files, see q38fn_1m.sh header> bash q38fn_1m.sh --apply
  python3 vllm-offload/equality_test.py e2 offdev2 ; python3 vllm-offload/equality_test.py e1 offdev2 results/offdev.e0.json (x2, then --external)
Then the final boot (public binding, no dev mode, offload on) or a rollback.
Caveat found tonight: `docker rm -f` inside the script's rollback failed silently once; the rollback had to be done by hand.

## Current state (2026-09-24 22:4x box time)
- q38fn-lru: plain 1M config (no offload, public 0.0.0.0:8057, no dev mode), restarted from q38fn-lru-1m-plain.
- q38fn-lru-262k: your original, stopped, untouched. Inspect backups: /mnt/llm-storage/localai/q38fn-1m/backup-*.
