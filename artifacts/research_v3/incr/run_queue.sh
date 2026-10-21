#!/bin/bash
cd /home/misa/projects/university/myildrim/segppd-cbir
for f in 0 1 2 3 4; do
  [ -f artifacts/research_v3/incr/runs/fold${f}_s0/result.json ] && continue
  .venv-gpu/bin/python -m segppd_cbir.rv2.incr_train $f 0 >> artifacts/research_v3/incr/queue.log 2>&1
done
touch artifacts/research_v3/incr/queue.done
