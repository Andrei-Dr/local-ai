"""Recompute telemetry for every label from the raw 1 Hz files in /ai/bench/mon -> /ai/bench/mon_summary.json (for the ledger backfill).
CPU busy is not recoverable here (only the start sample was stored); the backfill takes it from the sweep logs."""
import glob, json, os, time
out = {}
for gp in sorted(glob.glob("/ai/bench/mon/*.gpu")):
    label = os.path.basename(gp)[:-4]
    rows = [l.split() for l in open(gp) if l.strip() and not l.startswith("#")]
    f = lambda i: [float(r[i]) for r in rows if len(r) > i and r[i] not in ("-", "N/A")]
    pwr, temp, sm, mem, rx, tx = f(1), f(2), f(4), f(5), f(10), f(11)
    pp = gp[:-4] + ".perf"
    def imc(ev):
        if not os.path.exists(pp): return []
        return [float(l.split(",")[1]) for l in open(pp) if ev in l and l.split(",")[1].replace(".", "").isdigit()]
    rd, wr = imc("data_reads"), imc("data_writes")
    m = lambda x: round(sum(x) / len(x), 2) if x else None
    mx = lambda x: max(x) if x else None
    g = lambda v, d: round(v / d, 2) if v is not None else None
    out[label] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(os.path.getmtime(gp))), "seconds": len(rows),
                  "gpu_util_avg": m(sm), "gpu_util_max": mx(sm), "gpu_mem_util_avg": m(mem), "power_w_avg": m(pwr), "power_w_max": mx(pwr), "power_cap_w": 100,
                  "gpu_temp_c_max": mx(temp), "pcie_rx_gbs_avg": g(m(rx), 1000), "pcie_rx_gbs_max": g(mx(rx), 1000), "pcie_tx_gbs_avg": g(m(tx), 1000), "pcie_tx_gbs_max": g(mx(tx), 1000),
                  "dram_read_gbs_avg": g(m(rd), 1024), "dram_read_gbs_max": g(mx(rd), 1024), "dram_write_gbs_avg": g(m(wr), 1024), "dram_write_gbs_max": g(mx(wr), 1024)}
json.dump(out, open("/ai/bench/mon_summary.json", "w"), indent=1)
print(len(out), "labels")
