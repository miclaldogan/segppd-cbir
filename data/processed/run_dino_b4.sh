#!/bin/bash
cd /home/misa/projects/university/myildrim/segppd-cbir
P=".venv-gpu/bin/python -m segppd_cbir.train_embed --backbone dinov2s --mode partial --lr_bb 3e-5 --loss supcon --variant base"
$P --tag D5_dinoB_p4_supcon_lr3e-5_s1 --seed 1
$P --tag D5_dinoB_p4_supcon_lr3e-5_s2 --seed 2
echo DONE
