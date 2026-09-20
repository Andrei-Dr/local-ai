# ctx1 — long-context ladder report

## Ladder

| ctx | kv | slots | prompt_n | prefill t/s | prefill wall | decode t/s | vram | KV | RS | slot | save ms | restore ms | restore prompt_n | restore wall | speedup | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16384 | f16 | 24 | 12397 | 84.5 | 149 s | 27.8 | 2678 | 320.0 | 62.8 | 307 | 158.1 | 592.0 | 12397 | 149.7 | 1.0 | save ok / restore ok **RESTORE DID NOT SKIP PROMPT** |
| 32768 | f16 | 24 | 26824 | 75.7 | 357 s | 28.1 | 3038 | 640.0 | 62.8 | 589 | 345.2 | - | - | - | - | save ok / restore skipped |
| 65536 | f16 | 24 | - | - | - | - | 3676 | 1280.0 | 62.8 | - | - | - | - | - | - | save died / restore skipped |
| 131072 | q4_0 | 24 | 119569 | 46.2 | 2595 s (0:43) | 11.5 | 3434 | 720.0 | 62.8 | 723 | 397.2 | - | - | - | - | save ok / restore skipped |
| 262144 | q4_0 | 12 | - | - | - | - | - | - | - | - | - | - | - | - | - | save died / restore skipped |

## 32k variants vs baseline `ctx1_c32k_save`

| variant | decode Δ% | prefill Δ% | vram ΔMiB |
|---|---|---|---|
| `ctx1_c32k_nkvo_save` | -67.8% | -1.4% | -698 |
| `ctx1_c32k_pf1024_save` | -36.8% | +42.1% | -870 |
| `ctx1_c32k_pf2048_save` | -14.4% | +78.9% | -522 |
| `ctx1_c32k_q4_save` | -21.3% | -0.1% | -460 |
| `ctx1_c32k_q8_save` | -8.5% | -0.1% | -300 |

## Ubatch fit

F = 3.963 s fixed per ubatch, m = 5.445 ms/token, asymptote = 183.6 tok/s

| ub | predicted t/s | expert-only 100k | 200k | 250k |
|---|---|---|---|---|
| 512 | 75.8 | 1318.5 s | 2637.1 s | 3296.3 s |
| 1024 | 107.4 | 931.5 s | 1863.0 s | 2328.8 s |
| 2048 | 135.5 | 738.0 s | 1476.0 s | 1845.0 s |
| 4096 | 155.9 | 641.3 s | 1282.5 s | 1603.2 s |

attention cost grows with depth and is NOT in this fit; see the depth-decay table

## Depth decay

| prompt_n | prefill t/s (% of 16k) | decode t/s (% of 16k) |
|---|---|---|
| 12397 | 84.5 (100%) | 27.8 (100%) |
| 26824 | 75.7 (90%) | 28.1 (101%) |
| 119569 | 46.2 (55%) | 11.5 (41%) |

## Two-phase verdict

two-phase (prefill config -> decode config) FAILED

## Gates and dead rows

- `restore gate: 16k restore did NOT skip the prompt => deep restores OFF`
- `ctx1_c131k_restore`: skipped: the 16k restore did not skip the prompt
- `ctx1_c262k_restore`: skipped: the 16k restore did not skip the prompt
- `ctx1_c262k_save`: died: 0.08.793.027 E srv  llama_server: exiting due to model loading error
- `ctx1_c32k_mtp_save`: died: out of memory
- `ctx1_c32k_restore`: skipped: the 16k restore did not skip the prompt
- `ctx1_c64k_restore`: skipped: the 16k restore did not skip the prompt
- `ctx1_c64k_save`: died: out of memory

