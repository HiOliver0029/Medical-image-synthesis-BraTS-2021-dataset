#!/bin/bash
set -e

cd /home/user/DLMI_HW1

echo "========================================"
echo "Starting CycleGAN Training..."
echo "========================================"
/home/user/DLMI_HW1/.venv/bin/python scripts/train_cyclegan.py --config configs/cyclegan_brats.yaml --output-dir checkpoints 2>&1 | tee train.log

echo "========================================"
echo "Starting Evaluation..."
echo "========================================"
# The fastest training config we set earlier saves at epoch 20
/home/user/DLMI_HW1/.venv/bin/python scripts/evaluate_cyclegan.py --config configs/cyclegan_brats.yaml --checkpoint checkpoints/cyclegan_epoch_020.pt > eval.log 2>&1

echo "========================================"
echo "Pipeline Completed!"
echo "Evaluation results are in eval.log"
echo "========================================"
