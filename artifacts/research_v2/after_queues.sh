#!/bin/bash
# Sequential post-processing chain (no concurrent jobs: an earlier concurrent run was OOM-killed on the 15 GB host).
cd /home/misa/projects/university/myildrim/segppd-cbir
while ! grep -q QUEUE_DONE artifacts/research_v2/trackC/queue_seeds.log 2>/dev/null; do sleep 30; done
artifacts/research_v2/run_queue_generic.sh segppd_cbir.rv2.plantseg_train artifacts/research_v2/trackB/configs artifacts/research_v2/trackB/runs artifacts/research_v2/trackB/queue_seeds.txt artifacts/research_v2/trackB/queue_seeds_retry.log
.venv-gpu/bin/python -m segppd_cbir.rv2.local_rerank > artifacts/research_v2/trackC/local_rerank_PA.log 2>&1
.venv-gpu/bin/python -m segppd_cbir.rv2.trackC_eval > artifacts/research_v2/trackC/trackC_eval.log 2>&1
echo AFTER_DONE > artifacts/research_v2/after.done
