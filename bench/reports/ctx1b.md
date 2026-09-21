# ctx1b — presave / extend pairs

| ctx | kv | pre cfg | pre prompt_n | prefill t/s | prefill wall | slot MiB | save ms | ext cfg | restore ms | ext prompt_n | cache_n | ext_n | decode t/s | wall s | speedup | reply | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 131072 | q4_0 | ub1024 slots0 | 119569 | 56.8 | 2110 s (0:35) | 723 | 405 | ub128 slots24 | 435 | 32 | 119632 | 31 | 12.43 | 6.6 | 319.7 | '1. **N1 Drafting Cliff:** The n-gram drafting implementation' | RESTORED |
| 16384 | f16 | ub512 slots24 | 12397 | 84.5 | 147 s | 305 | 149 | ub512 slots24 | 768 | 32 | 12397 | 31 | 30.64 | 3.2 | 46.1 | '1. **N1 Drafting Performance Risk:** The hybrid n-gram draft' | RESTORED |
| 16384 | f16 | ub512 slots24 | 12397 | 84.5 | 147 s | 305 | 149 | ub512 slots24 | 152 | 31 | 12397 | 31 | 24.65 | 3.6 | 40.7 | '1. **N1 Drafting Performance Risk:** The hybrid n-gram draft' | RESTORED |
| 16384 | f16 | ub512 slots24 | 12397 | 84.5 | 149 s | 307 | 147 | ub512 slots24 | 628 | 32 | 12460 | 31 | 30.45 | 3.0 | 49.6 | '1. **N1 Drafting Rejection:** The hybrid n-gram drafting (N1' | RESTORED |
| 16384 | f16 | ub512 slots24 | 12397 | 84.5 | 149 s | 307 | 147 | ub512 slots24 | 551 | 31 | 12460 | 31 | 28.88 | 3.0 | 48.8 | '1. **N1 Drafting Performance Risk:** The hybrid n-gram draft' | RESTORED |
| 262144 | q4_0 | ub512 slots0 | 239107 | 30.8 | 7779 s (2:09) | 1383 | 706 | ub128 slots8 | - | - | - | - | - | - | - | '' | died |
| 32768 | f16 | ub2048 slots0 | 26824 | 135.9 | 201 s | 589 | 340 | ub128 slots24 | 305 | 32 | 26887 | 31 | 28.55 | 3.1 | 63.9 | '1. **N1 Draft Rejection on GDN Models:** Using n-gram drafts' | RESTORED |
| 32768 | f16 | ub2048 slots0 | 26824 | 135.9 | 201 s | 589 | 340 | ub128 slots8 +MTP | 317 | 32 | 26887 | 31 | 17.16 | 4.6 | 43.4 | '1. **N1 Drafting Net Loss on GDN:** Implementing longer n-gr' | RESTORED |

## Two-phase verdicts

- `c131k`: WORKS
- `c262k`: FAILED
- `c32k`: WORKS

## Extend gate

- `extend gate: PASS with pre_gen=64 drop=0 => deep rungs ON`

## Dead rows

- `ctx1b_c262k_dec_extend`: died: out of memory

