#!/bin/bash
cd /home/misa/projects/university/myildrim/segppd-cbir
L=artifacts/research_v3/cross/queue.log; PY=.venv-gpu/bin/python; mkdir -p artifacts/research_v3/cross/emb_single
for s in 0 1 2; do
  [ -f artifacts/research_v3/cross/emb_single/S0_s$s.npz ] || $PY -m segppd_cbir.rv2.cross_embed artifacts/research_v2/trackA/runs/A_pro_alpha32_delta0.3_s$s/best.pt artifacts/research_v3/cross/emb_single/S0_s$s.npz >> $L 2>&1
  [ -f artifacts/research_v3/cross/emb_single/T0_s$s.npz ] || $PY -m segppd_cbir.rv2.cross_embed artifacts/research_v2/trackB/runs/B_M3_metric_proxyanchor_s$s/best.pt artifacts/research_v3/cross/emb_single/T0_s$s.npz >> $L 2>&1
done
for v in X1 X2 X3; do for s in 0 1 2; do
  [ -f artifacts/research_v3/cross/runs/cross_${v}_s$s/result.json ] && continue
  $PY -m segppd_cbir.rv2.cross_train $v $s >> $L 2>&1
done; done
touch artifacts/research_v3/cross/queue.done
