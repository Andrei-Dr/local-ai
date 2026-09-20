# kq1 — K-quant experts for Qwen3.6

K-quant (Q2_K) expert tensors roughly halve transferred expert bytes versus the IQ2_M expert store; prefill — transfer-bound on/offload — should gain near that ratio, while net decode gain depends on miss-cost falling faster than the per-expert dequant penalty grows, with MTP n=3 buying more once per-token work got cheaper.


## miss cost, no cache

**decode_tps**

| prompt | base decode t/s | test decode t/s | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 27.40 (n=1) | 30.26 (n=1) | +10.44% | - | WIN |
| reason | 27.68 (n=1) | 34.03 (n=1) | +22.94% | - | WIN |
| edit | 26.95 (n=1) | 34.12 (n=1) | +26.60% | - | WIN |
| long | 26.98 (n=1) | 33.67 (n=1) | +24.80% | - | WIN |
| ALL | 27.25 | 33.02 | +21.16% | - | WIN |
base: 1 records | acceptance code -, reason -, edit -, long - | cache hit - | vram 1434 MiB | commits 2582f5cb6
test: 1 records | acceptance code -, reason -, edit -, long - | cache hit - | vram 1436 MiB | commits 2582f5cb6

**prefill_tps**

| prompt | base prefill_tps | test prefill_tps | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 37.90 (n=1) | 36.60 (n=1) | -3.43% | - | LOSS |
| reason | 41.60 (n=1) | 40.20 (n=1) | -3.37% | - | LOSS |
| edit | 78.20 (n=1) | 89.30 (n=1) | +14.19% | - | WIN |
| long | 95.40 (n=1) | 104.80 (n=1) | +9.85% | - | WIN |
| ALL | 63.28 | 67.72 | +7.03% | - | WIN |
base: 1 records | acceptance code -, reason -, edit -, long - | cache hit - | vram 1434 MiB | commits 2582f5cb6
test: 1 records | acceptance code -, reason -, edit -, long - | cache hit - | vram 1436 MiB | commits 2582f5cb6


## best config

**decode_tps**

| prompt | base decode t/s | test decode t/s | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 48.88 (n=1) | 54.90 (n=1) | +12.32% | - | WIN |
| reason | 48.24 (n=1) | 55.29 (n=1) | +14.61% | - | WIN |
| edit | 49.56 (n=1) | 57.94 (n=1) | +16.91% | - | WIN |
| long | 37.75 (n=1) | 45.04 (n=1) | +19.31% | - | WIN |
| ALL | 46.11 | 53.29 | +15.58% | - | WIN |
base: 1 records | acceptance code 0.858, reason 0.876, edit 0.995, long 0.650 | cache hit 54.1% | vram 3460 MiB | commits 2582f5cb6
test: 1 records | acceptance code 0.839, reason 0.872, edit 0.980, long 0.659 | cache hit 48.7% | vram 3434 MiB | commits 2582f5cb6

**acceptance**

| prompt | base acceptance | test acceptance | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 0.86 (n=1) | 0.84 (n=1) | -2.21% | - | LOSS |
| reason | 0.88 (n=1) | 0.87 (n=1) | -0.46% | - | flat |
| edit | 0.99 (n=1) | 0.98 (n=1) | -1.51% | - | LOSS |
| long | 0.65 (n=1) | 0.66 (n=1) | +1.38% | - | WIN |
| ALL | 0.84 | 0.84 | -0.86% | - | flat |
base: 1 records | acceptance code 0.858, reason 0.876, edit 0.995, long 0.650 | cache hit 54.1% | vram 3460 MiB | commits 2582f5cb6
test: 1 records | acceptance code 0.839, reason 0.872, edit 0.980, long 0.659 | cache hit 48.7% | vram 3434 MiB | commits 2582f5cb6


## does n=3 pay now

**decode_tps**

| prompt | base decode t/s | test decode t/s | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 54.90 (n=1) | 51.83 (n=1) | -5.59% | - | LOSS |
| reason | 55.29 (n=1) | 55.50 (n=1) | +0.38% | - | flat |
| edit | 57.94 (n=1) | 61.49 (n=1) | +6.13% | - | WIN |
| long | 45.04 (n=1) | 41.60 (n=1) | -7.64% | - | LOSS |
| ALL | 53.29 | 52.61 | -1.29% | - | LOSS |
base: 1 records | acceptance code 0.839, reason 0.872, edit 0.980, long 0.659 | cache hit 48.7% | vram 3434 MiB | commits 2582f5cb6
test: 1 records | acceptance code 0.846, reason 0.805, edit 0.996, long 0.608 | cache hit 46.7% | vram 3408 MiB | commits 2582f5cb6

**acceptance**

| prompt | base acceptance | test acceptance | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 0.84 (n=1) | 0.85 (n=1) | +0.83% | - | flat |
| reason | 0.87 (n=1) | 0.81 (n=1) | -7.68% | - | LOSS |
| edit | 0.98 (n=1) | 1.00 (n=1) | +1.63% | - | WIN |
| long | 0.66 (n=1) | 0.61 (n=1) | -7.74% | - | LOSS |
| ALL | 0.84 | 0.81 | -2.84% | - | LOSS |
base: 1 records | acceptance code 0.839, reason 0.872, edit 0.980, long 0.659 | cache hit 48.7% | vram 3434 MiB | commits 2582f5cb6
test: 1 records | acceptance code 0.846, reason 0.805, edit 0.996, long 0.608 | cache hit 46.7% | vram 3408 MiB | commits 2582f5cb6


## Quality (job labels vs the reference run)

| label | gsm8k | humaneval | mmlu_pro |
|---|---|---|---|
| `q36_iq2m_cache48` | 100.0 ±0.0 | 95.1 ±3.4 | 77.1 ±5.0 |
| `q36_k2_cache26` | 96.0 ±2.8 | 92.7 ±4.1 | 78.6 ±4.9 |

## Runs seen

- `kq_q36iq2m_c30_mtp2` | vram 3460 | hit 54.1 | power_w_avg 51.9 | gpu_util_avg 89.5 | pcie_rx_gbs_max 11.49 | mem_avail_mib_min 3319 | swap_used_mib_max 513
- `kq_q36iq2m_off` | vram 1434 | hit - | power_w_avg 44.1 | gpu_util_avg 62.7 | pcie_rx_gbs_max 4.74 | mem_avail_mib_min 3723 | swap_used_mib_max 502
- `kq_q36k2_c24_mtp3` | vram 3408 | hit 46.7 | power_w_avg 53.0 | gpu_util_avg 89.2 | pcie_rx_gbs_max 11.64 | mem_avail_mib_min 2088 | swap_used_mib_max 524
- `kq_q36k2_c26_mtp2` | vram 3434 | hit 48.7 | power_w_avg 53.7 | gpu_util_avg 90.5 | pcie_rx_gbs_max 11.8 | mem_avail_mib_min 1947 | swap_used_mib_max 521
- `kq_q36k2_off` | vram 1436 | hit - | power_w_avg 49.0 | gpu_util_avg 68.0 | pcie_rx_gbs_max 5.76 | mem_avail_mib_min 2500 | swap_used_mib_max 517
