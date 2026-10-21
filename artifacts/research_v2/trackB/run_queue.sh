#!/bin/bash
cd /home/misa/projects/university/myildrim/segppd-cbir
while read -r n; do
  [ -z "$n" ] && continue
  [ -f artifacts/research_v2/trackB/runs/$n/result.json ] && continue
  .venv-gpu/bin/python -m segppd_cbir.rv2.plantseg_train artifacts/research_v2/trackB/configs/$n.json >> artifacts/research_v2/trackB/queue.log 2>&1
done < "$1"
echo QUEUE_DONE >> artifacts/research_v2/trackB/queue.log
