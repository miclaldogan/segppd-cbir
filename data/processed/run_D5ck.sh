#!/bin/bash
# Same config as D5 (DINOv2-B, last 4 blocks, SupCon, lr 3e-5), retrained WITH checkpoints for external-set inference.
cd /home/misa/projects/university/myildrim/segppd-cbir
for s in 0 1 2; do
  .venv-gpu/bin/python -m segppd_cbir.train_embed --backbone dinov2s --variant base --mode partial --blocks 4 --lr_bb 3e-5 --loss supcon --seed $s --save_ckpt --tag D5ck_dinoB_p4_supcon_lr3e-5_s$s
done
echo ALLDONE > data/processed/D5ck.done
