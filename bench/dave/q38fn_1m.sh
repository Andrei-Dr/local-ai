#!/bin/bash
# q38fn_1m.sh — switch Dave's q38fn vLLM (container q38fn-lru, R9700s) to a 1M-token context with YaRN, with rollback.
# Run it on Dave's box as dave. Default is a dry run that prints the new `docker run`; pass --apply to switch.
#   bash q38fn_1m.sh            # dry run: writes run_1m.sh next to this script and prints it
#   bash q38fn_1m.sh --apply    # stop + rename the running container to q38fn-lru-262k, start the 1M one, roll back on failure
#   bash q38fn_1m.sh --rollback # remove the 1M container, rename q38fn-lru-262k back and start it
#
# What changes (everything else, binds / env / ports / devices / image / other flags, is copied from `docker inspect`):
#   --hf-overrides  YaRN factor 4 over the native 262,144 positions (keeps mrope_section / interleaved / partial rotary 0.25)
#   --max-model-len 1010000
#   --kv-cache-memory 8300000000   (KV=; was 3.9e9 per GPU; vLLM needs 7.63 GiB per GPU at 1.01M incl. mamba page padding + MTP)
#   --cpu-offload-gb 45            (OFF=; was 40: +5 GB of experts per GPU to host RAM pays for the bigger KV)
# Why it can work: 12 of 48 layers are full attention (2 KV heads x 256, fp8) -> ~12 KiB / token total; the 36 linear-attention
# layers keep a fixed state. The patched mrope.py (bind-mounted) supports scaling_factor and sizes its cos/sin table at
# 4 x 262,144 = 1,048,576 rows. vLLM 0.27's get_rope builds YaRN on mrope (rotary_embedding/__init__.py, rope_type "yarn").
# Checked elsewhere: the override JSON is accepted by vLLM 0.28 (the MI210 attempt got past config and failed only on FP8
# weights, which gfx90a cannot run).
# Caveats: static YaRN applies to every request (Qwen warns short-context quality can dip: spot-check); decode may slow a
# little from the extra 3 GB/GPU offload; LiteLLM's model_info (max_input_tokens) may still say 262144.
set -euo pipefail
cd "$(dirname "$0")"
C=q38fn-lru; OLD=${OLD:-q38fn-lru-262k}; PORT=8057   # OLD=<name> for tuning rounds: the original 262k container stays untouched
if [ "${1:-}" = --rollback ]; then
  docker rm -f $C >/dev/null 2>&1 || true
  docker rename $OLD $C && docker start $C && echo "rolled back: $C (262k) started"; exit 0
fi
[ "$(docker inspect -f '{{.Name}}' $C 2>/dev/null)" = "/$C" ] || { echo "no container $C"; exit 1; }
[ "${1:-}" != --apply ] || ! docker inspect $OLD >/dev/null 2>&1 || { echo "$OLD already exists: pick another OLD= name"; exit 1; }
docker inspect $C > $C.inspect.json
python3 - "$C.inspect.json" > run_1m.sh <<'EOF'
import json, shlex, sys
c = json.load(open(sys.argv[1]))[0]
h, cfg = c["HostConfig"], c["Config"]
args = list(cfg["Cmd"])
def setarg(k, v):
    if k in args: args[args.index(k) + 1] = v
    else: args.extend([k, v])
import os
setarg("--max-model-len", "1010000"); setarg("--kv-cache-memory", os.environ.get("KV", "8300000000")); setarg("--cpu-offload-gb", os.environ.get("OFF", "45"))
for k, v in json.loads(os.environ.get("EXTRA", "{}")).items():   # EXTRA='{"--flag": "value"}' tuning overrides
    setarg(k, v)
setarg("--hf-overrides", json.dumps({"text_config": {"rope_parameters": {
    "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144, "rope_theta": 10000000,
    "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    "max_position_embeddings": 1048576}}, separators=(",", ":")))
run = ["docker", "run", "-d", "--name", "q38fn-lru", "--ipc", h["IpcMode"], "--shm-size", str(h["ShmSize"])]
if h["RestartPolicy"]["Name"] not in ("", "no"): run += ["--restart", h["RestartPolicy"]["Name"]]
if h["NetworkMode"] not in ("default", "bridge"): run += ["--network", h["NetworkMode"]]
for b in h["Binds"] or []: run += ["-v", b]
for p, bs in (h["PortBindings"] or {}).items():
    for b in bs: run += ["-p", (b["HostIp"] + ":" if b["HostIp"] else "") + b["HostPort"] + ":" + p.split("/")[0]]
for g in h["GroupAdd"] or []: run += ["--group-add", g]
for s in h["SecurityOpt"] or []: run += ["--security-opt", s]
for d in h["Devices"] or []: run += ["--device", d["PathOnHost"] + ":" + d["PathInContainer"]]
for e in cfg["Env"]: run += ["-e", e]
if cfg.get("Entrypoint"): run += ["--entrypoint", cfg["Entrypoint"][0]]
run += [cfg["Image"]] + args
print("#!/bin/bash\n" + " ".join(shlex.quote(x) for x in run))
EOF
chmod +x run_1m.sh
if [ "${1:-}" != --apply ]; then echo "dry run: $(pwd)/run_1m.sh"; cat run_1m.sh; exit 0; fi
docker stop $C >/dev/null && docker rename $C $OLD
if ! ./run_1m.sh >/dev/null; then echo "docker run failed -> rollback"; exec bash "$0" --rollback; fi
echo "waiting for the 1M server on :$PORT (model load takes minutes)"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && { echo "UP: $(curl -s http://127.0.0.1:$PORT/v1/models | grep -o '"max_model_len":[0-9]*')"; exit 0; }
  docker ps -q -f name=^$C$ | grep -q . || break; sleep 15; done
docker logs --tail 40 $C > q38fn-1m.fail.log 2>&1 || true
echo "1M server did not come up (tail in $(pwd)/q38fn-1m.fail.log) -> rollback"; exec bash "$0" --rollback
