#!/bin/bash
cd /home/misa/projects/university/myildrim/segppd-cbir
P=".venv-gpu/bin/python -m segppd_cbir.train_embed --backbone dinov2s --mode partial --lr_bb 3e-5"
$P --tag D4_dino_p4_supcon_lr3e-5_s1 --loss supcon --seed 1
$P --tag D4_dino_p4_supcon_lr3e-5_s2 --loss supcon --seed 2
$P --tag D3_dino_p4_proxyanchor_lr3e-5_s1 --loss proxyanchor --seed 1
$P --tag D3_dino_p4_proxyanchor_lr3e-5_s2 --loss proxyanchor --seed 2
$P --tag D5_dinoB_p4_supcon_lr3e-5_s0 --loss supcon --seed 0 --variant base
echo DONE
