#!/bin/bash
# usage: run_queue_generic.sh <python module> <config dir> <runs dir> <queue file> <log file>   (one training at a time on the 6 GB GPU)
cd /home/misa/projects/university/myildrim/segppd-cbir
while read -r n; do
  [ -z "$n" ] && continue
  [ -f $3/$n/result.json ] && continue
  .venv-gpu/bin/python -m $1 $2/$n.json >> $5 2>&1
done < "$4"
echo QUEUE_DONE >> $5
