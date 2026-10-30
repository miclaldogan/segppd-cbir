#!/bin/bash
# usage: run_queue.sh <config dir> <queue file> <log file>   (one training at a time on the 6 GB GPU)
cd "/home/misa/projects/university/myildrim/finished?/segppd-cbir"
export PYTHONPATH=src
while read -r n; do
  [ -z "$n" ] && continue
  .venv-gpu/bin/python -m segppd_cbir.rv4.train "$1/$n.json" >> "$3" 2>&1
done < "$2"
echo QUEUE_DONE >> "$3"
