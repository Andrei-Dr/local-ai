# T0 supplement (CPU only, no GPU): host-side optimizer step cost per 1B BF16 params at 12 threads (one FSDP2 rank's
# share of cores 24-47). Tensors shaped like expert shards [128, 1024, 2048] + [128, 2048, 512] (the 3D expert params).
import sys, time, torch
torch.set_num_threads(12)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ps = []
for i in range(N):
    for shape in ((64, 1024, 2048), (64, 2048, 512)):
        p = torch.randn(shape, dtype=torch.bfloat16) * 0.02
        p.grad = torch.randn(shape, dtype=torch.bfloat16) * 1e-3
        ps.append(p)
n = sum(p.numel() for p in ps)
arms = {"Adafactor": lambda: torch.optim.Adafactor(ps, lr=1e-5),
        "Adafactor_foreach": lambda: torch.optim.Adafactor(ps, lr=1e-5, foreach=True),
        "SGD": lambda: torch.optim.SGD(ps, lr=1e-5),
        "SGD_mom_foreach": lambda: torch.optim.SGD(ps, lr=1e-5, momentum=0.9, foreach=True),
        "AdamW_foreach": lambda: torch.optim.AdamW(ps, lr=1e-5, foreach=True),
        "AdamW_fused": lambda: torch.optim.AdamW(ps, lr=1e-5, fused=True)}
for name, mk in arms.items():
    try:
        o = mk(); o.step()
        t = []
        for _ in range(3):
            s = time.perf_counter(); o.step(); t.append(time.perf_counter() - s)
        print(f"{name:18s} {sorted(t)[1] / (n / 1e9):7.2f} s per 1B params (n={n/1e9:.2f}B)", flush=True)
        del o
    except Exception as e:
        print(f"{name:18s} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
