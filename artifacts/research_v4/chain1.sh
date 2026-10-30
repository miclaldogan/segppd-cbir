#!/bin/bash
# after the sweep: TTA on the reference PA checkpoints (GPU inference), then the DINOv2-B distillation queue
cd "/home/misa/projects/university/myildrim/finished?/segppd-cbir"; export PYTHONPATH=src
until grep -q QUEUE_DONE artifacts/research_v4/sweep/queue.log; do sleep 20; done
.venv-gpu/bin/python -m segppd_cbir.rv4.tta artifacts/research_v2/trackA/runs/A_pro_alpha32_delta0.3_s0 artifacts/research_v2/trackA/runs/A_pro_alpha32_delta0.3_s1 artifacts/research_v2/trackA/runs/A_pro_alpha32_delta0.3_s2 > artifacts/research_v4/tta/log_PA.txt 2>&1
echo TTA_DONE >> artifacts/research_v4/tta/log_PA.txt
artifacts/research_v4/run_queue.sh artifacts/research_v4/kd/configs artifacts/research_v4/kd/queue.txt artifacts/research_v4/kd/queue.log
