# kld1 — KL divergence vs the Q6 reference

| candidate | GiB | mean KLD ± err | median | 99% | 99.9% | max | same top p % | PPL ratio | vs IQ2_M |
|---|---|---|---|---|---|---|---|---|---|
| K2-expQ2K-downQ3K | 12.0 | 0.21814 ± 0.00367 | 0.114 | 1.884 | 4.009 | 6.164 | 79.9 | 1.165 | x1.00 lower divergence from the Q6 reference |
| IQ2_M | 10.9 | 0.21880 ± 0.00389 | 0.109 | 2.001 | 4.372 | 6.283 | 79.8 | 1.180 |  |

VERDICT: K2-expQ2K-downQ3K BEST (lowest mean AND lowest 99% KLD)

