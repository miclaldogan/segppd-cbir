#!/bin/bash
# after the DINOv2-B KD queue: small-backbone follow-up queue
cd "/home/misa/projects/university/myildrim/finished?/segppd-cbir"; export PYTHONPATH=src
until grep -q QUEUE_DONE artifacts/research_v4/kd/queue.log 2>/dev/null; do sleep 20; done
artifacts/research_v4/run_queue.sh artifacts/research_v4/followup/configs artifacts/research_v4/followup/queue.txt artifacts/research_v4/followup/queue.log
