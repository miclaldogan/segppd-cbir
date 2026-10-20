#!/bin/bash
cd /home/misa/projects/university/myildrim/segppd-cbir
P=".venv-gpu/bin/python -m segppd_cbir.train_embed --backbone dinov2s"
$P --tag D3_dino_p4_proxyanchor_lr3e-5_s0 --mode partial --lr_bb 3e-5 --loss proxyanchor --seed 0
$P --tag D4_dino_p4_supcon_lr3e-5_s0 --mode partial --lr_bb 3e-5 --loss supcon --seed 0
echo DONE
