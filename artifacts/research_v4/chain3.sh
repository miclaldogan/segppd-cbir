#!/bin/bash
# after the follow-up queue: finalize the 336 px ConvNeXt run from its checkpoint (embedding died once, host RAM)
cd "/home/misa/projects/university/myildrim/finished?/segppd-cbir"; export PYTHONPATH=src
until grep -q QUEUE_DONE artifacts/research_v4/followup/queue.log; do sleep 15; done
.venv-gpu/bin/python -m segppd_cbir.rv4.train artifacts/research_v4/followup/configs/F_dinov3_convnext_tiny_336.json >> artifacts/research_v4/followup/queue.log 2>&1
echo FINALIZE_DONE >> artifacts/research_v4/followup/queue.log
