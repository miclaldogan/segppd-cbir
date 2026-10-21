#!/bin/bash
# Sequential GPU queue (one training at a time on the RTX 4050 6 GB). usage: run_queue.sh <queue_file>
cd /home/misa/projects/university/myildrim/segppd-cbir
while read -r n; do
  [ -z "$n" ] && continue
  [ -f artifacts/research_v2/trackA/runs/$n/result.json ] && continue
  .venv-gpu/bin/python -m segppd_cbir.rv2.train artifacts/research_v2/trackA/configs/$n.json >> artifacts/research_v2/trackA/queue.log 2>&1
done < "$1"
echo QUEUE_DONE >> artifacts/research_v2/trackA/queue.log
