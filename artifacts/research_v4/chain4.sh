#!/bin/bash
cd "/home/misa/projects/university/myildrim/finished?/segppd-cbir"; export PYTHONPATH=src
while ps -eo args | grep -q "[r]v4.train"; do sleep 15; done
artifacts/research_v4/run_queue.sh artifacts/research_v4/final/configs artifacts/research_v4/final/queue.txt artifacts/research_v4/final/queue.log
