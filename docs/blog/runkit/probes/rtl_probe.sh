#!/usr/bin/env bash
# Minimal rocm-trace-lite check: does `rtl trace` capture GPU kernels in a given image?
#
#   bash docs/blog/runkit/probes/rtl_probe.sh <image> [rtl_wheel_url]
#
# Runs a 2-GPU bf16 matmul + all_reduce loop under each RTL mode and prints op types,
# GPU ids, and RCCL op counts per trace. A working setup shows KernelExecution ops on
# real GPU ids; on rocm/primus:v26.7 (ROCm 10.0.0 TheRock wheels) RTL v0.3.3/v0.3.7
# capture 0 ops, or only roctx UserMarker ranges inside TransformerEngine workloads.
set -euo pipefail
IMAGE=${1:?usage: rtl_probe.sh <image> [rtl_wheel_url]}
WHEEL_URL=${2:-https://github.com/sunway513/rocm-trace-lite/releases/download/v0.3.7/rocm_trace_lite-0.3.7-py3-none-linux_x86_64.whl}
WORK=$(mktemp -d)
curl -sfL -o "$WORK/$(basename "$WHEEL_URL")" "$WHEEL_URL"

cat > "$WORK/mm.py" <<'EOF'
import torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r)
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(50):
    b = a @ a
    dist.all_reduce(b)
torch.cuda.synchronize()
dist.destroy_process_group()
EOF

cat > "$WORK/inspect.py" <<'EOF'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
q = lambda s: c.execute(s).fetchall()
print("  op types:", q("select s.string, count(*) from rocpd_op o join rocpd_string s on o.opType_id=s.id group by 1"))
print("  gpu ids :", q("select gpuId, count(*) from rocpd_op group by 1"))
print("  rccl ops:", q("select count(*) from rocpd_op o join rocpd_string s on o.description_id=s.id where s.string like '%ccl%'")[0][0])
EOF

docker run --rm --network host --ipc=host --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined -v "$WORK:/probe" -w /probe "$IMAGE" bash -c '
pip install -q /probe/*.whl >/dev/null 2>&1
rtl --version 2>/dev/null || true
modes=$(rtl trace --help | grep -oE "\{[a-z,]+\}" | head -1 | tr -d "{}" | tr , " ")
for m in $modes; do
  [ "$m" = full ] && continue
  echo "== mode=$m"
  rtl trace --mode "$m" -o "out_$m/trace.db" torchrun --nproc_per_node 2 mm.py >"out_$m.log" 2>&1 || true
  if [ -f "out_$m/trace.db" ]; then python3 inspect.py "out_$m/trace.db"; else echo "  no trace (see out_$m.log)"; fi
done'
echo "artifacts: $WORK"
