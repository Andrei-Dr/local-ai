# mtp1 — In-model MTP head

The standalone MTP head squats on ~727 MiB of VRAM (duplicate output.weight + its own 256 experts). Grafting blk.40 into the target file hands that VRAM back; spent on cache slots it should buy 2-4% near the hit-rate curve, and speculation stays lossless (textdiff gates every equal-slot pair). Kill: in-model within noise at equal slots AND no gain when the freed VRAM is spent => stay on -md.


## same 30 slots

**decode_tps**

| prompt | base decode t/s | test decode t/s | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 46.15 (n=1) | 48.26 (n=1) | +4.57% | - | WIN |
| reason | 46.92 (n=1) | 44.25 (n=1) | -5.69% | - | LOSS |
| edit | 49.77 (n=1) | 48.74 (n=1) | -2.07% | - | LOSS |
| long | 34.86 (n=1) | 36.51 (n=1) | +4.73% | - | WIN |
| ALL | 44.42 | 44.44 | +0.03% | - | flat |
base: 1 records | acceptance code 0.851, reason 0.851, edit 0.995, long 0.588 | cache hit 53.3% | vram 3444 MiB | commits 498696c3c
test: 1 records | acceptance code 0.859, reason 0.815, edit 0.995, long 0.653 | cache hit 54.5% | vram 2924 MiB | commits 498696c3c

**acceptance**

| prompt | base acceptance | test acceptance | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 0.85 (n=1) | 0.86 (n=1) | +0.94% | - | flat |
| reason | 0.85 (n=1) | 0.81 (n=1) | -4.23% | - | LOSS |
| edit | 0.99 (n=1) | 0.99 (n=1) | +0.00% | - | flat |
| long | 0.59 (n=1) | 0.65 (n=1) | +11.05% | - | WIN |
| ALL | 0.82 | 0.83 | +1.13% | - | WIN |
base: 1 records | acceptance code 0.851, reason 0.851, edit 0.995, long 0.588 | cache hit 53.3% | vram 3444 MiB | commits 498696c3c
test: 1 records | acceptance code 0.859, reason 0.815, edit 0.995, long 0.653 | cache hit 54.5% | vram 2924 MiB | commits 498696c3c


## freed VRAM spent: 42 slots

**decode_tps**

| prompt | base decode t/s | test decode t/s | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 46.15 (n=1) | 51.60 (n=1) | +11.81% | - | WIN |
| reason | 46.92 (n=1) | 48.40 (n=1) | +3.15% | - | WIN |
| edit | 49.77 (n=1) | 48.09 (n=1) | -3.38% | - | LOSS |
| long | 34.86 (n=1) | 34.46 (n=1) | -1.15% | - | LOSS |
| ALL | 44.42 | 45.64 | +2.73% | - | WIN |
base: 1 records | acceptance code 0.851, reason 0.851, edit 0.995, long 0.588 | cache hit 53.3% | vram 3444 MiB | commits 498696c3c
test: 1 records | acceptance code 0.884, reason 0.839, edit 0.985, long 0.659 | cache hit 62.1% | vram 3408 MiB | commits 498696c3c

**acceptance**

| prompt | base acceptance | test acceptance | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 0.85 (n=1) | 0.88 (n=1) | +3.88% | - | WIN |
| reason | 0.85 (n=1) | 0.84 (n=1) | -1.41% | - | LOSS |
| edit | 0.99 (n=1) | 0.98 (n=1) | -1.01% | - | LOSS |
| long | 0.59 (n=1) | 0.66 (n=1) | +12.07% | - | WIN |
| ALL | 0.82 | 0.84 | +2.50% | - | WIN |
base: 1 records | acceptance code 0.851, reason 0.851, edit 0.995, long 0.588 | cache hit 53.3% | vram 3444 MiB | commits 498696c3c
test: 1 records | acceptance code 0.884, reason 0.839, edit 0.985, long 0.659 | cache hit 62.1% | vram 3408 MiB | commits 498696c3c


## freed VRAM spent: 46 slots

**decode_tps**

| prompt | base decode t/s | test decode t/s | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 46.15 (n=1) | 51.69 (n=1) | +12.00% | - | WIN |
| reason | 46.92 (n=1) | 50.22 (n=1) | +7.03% | - | WIN |
| edit | 49.77 (n=1) | 53.76 (n=1) | +8.02% | - | WIN |
| long | 34.86 (n=1) | 41.64 (n=1) | +19.45% | - | WIN |
| ALL | 44.42 | 49.33 | +11.04% | - | WIN |
base: 1 records | acceptance code 0.851, reason 0.851, edit 0.995, long 0.588 | cache hit 53.3% | vram 3444 MiB | commits 498696c3c
test: 1 records | acceptance code 0.830, reason 0.867, edit 0.985, long 0.704 | cache hit 65.0% | vram 3570 MiB | commits 498696c3c

**acceptance**

| prompt | base acceptance | test acceptance | delta % | noise % | verdict |
|---|---|---|---|---|---|
| code | 0.85 (n=1) | 0.83 (n=1) | -2.47% | - | LOSS |
| reason | 0.85 (n=1) | 0.87 (n=1) | +1.88% | - | WIN |
| edit | 0.99 (n=1) | 0.98 (n=1) | -1.01% | - | LOSS |
| long | 0.59 (n=1) | 0.70 (n=1) | +19.73% | - | WIN |
| ALL | 0.82 | 0.85 | +3.07% | - | WIN |
base: 1 records | acceptance code 0.851, reason 0.851, edit 0.995, long 0.588 | cache hit 53.3% | vram 3444 MiB | commits 498696c3c
test: 1 records | acceptance code 0.830, reason 0.867, edit 0.985, long 0.704 | cache hit 65.0% | vram 3570 MiB | commits 498696c3c


## Runs seen

- `mtp1_q36_inmodel_c30` | vram 2924 | hit 54.5 | power_w_avg 53.1 | gpu_util_avg 89.6 | pcie_rx_gbs_max 11.73 | mem_avail_mib_min 3192 | swap_used_mib_max 1000
- `mtp1_q36_inmodel_c42` | vram 3408 | hit 62.1 | power_w_avg 53.9 | gpu_util_avg 89.8 | pcie_rx_gbs_max 11.47 | mem_avail_mib_min 3213 | swap_used_mib_max 1001
- `mtp1_q36_inmodel_c46` | vram 3570 | hit 65.0 | power_w_avg 53.3 | gpu_util_avg 91.9 | pcie_rx_gbs_max 12.08 | mem_avail_mib_min 3223 | swap_used_mib_max 1004
- `mtp1_q36_md_c30` | vram 3444 | hit 53.3 | power_w_avg 52.0 | gpu_util_avg 88.8 | pcie_rx_gbs_max 11.56 | mem_avail_mib_min 3294 | swap_used_mib_max 982
