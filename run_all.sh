#!/usr/bin/env bash
# Reproduces every experiment in the report, then writes the result tables.
#
#   bash run_all.sh                       # 4 GPUs, 3 runs per GPU
#   GPUS="0 1" JOBS=2 bash run_all.sh     # other GPUs / runs per GPU
#
# Finished runs are skipped, so after a failure (or an interruption) fix the
# cause and start the script again. Results:
#   runs/summary.csv, runs/summary.md         CV accuracy of every experiment
#   runs/comparisons.csv                      paired McNemar tests
#   runs/t4_r18_places_ft/per_class.csv       per-class accuracy of the best recipe
#   runs/t4_r18_places_ft/f0_s0/test_results.csv   test accuracy of the final model
#   runs/metadata_baseline.txt                accuracy from image size and colour alone
set -euo pipefail
cd "$(dirname "$0")"
GPUS=${GPUS:-"0 1 2 3"}
JOBS=${JOBS:-3}
launch() { python scripts/launch.py "$@" --gpus $GPUS --jobs-per-gpu "$JOBS"; }

mkdir -p runs
python scripts/metadata_baseline.py | tee runs/metadata_baseline.txt

# Download the pretrained weights once, before many runs need them at the same time.
python -c "from torchvision import models; from scenecnn.models import resnet18_places365
models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); resnet18_places365()"

# Each experiment: 5 folds x 3 seeds.
launch configs/scratch/a*.yaml configs/analysis/x2_scratch_crop.yaml configs/transfer/t[1-4]_*.yaml
launch configs/scratch/a4_adamw.yaml --grid optim.lr=0.003,0.001,0.0003 optim.weight_decay=0.05,0.005,0.0005
launch configs/scratch/a6_res128.yaml --grid optim.lr=0.001,0.003,0.005,0.01

python scripts/summarize.py
rm -f runs/comparisons.csv
python scripts/oof.py --compare a6_res128 a12_longer
python scripts/oof.py --compare a6_res128 a11_attnpool
python scripts/oof.py --compare a1_simple_cnn x2_scratch_crop
python scripts/oof.py --compare t2_r18_imagenet_ft t3_r18_places_linear
python scripts/oof.py --experiment t4_r18_places_ft

# Final model: the five fold models of the best recipe (seed 0), evaluated once on the test set.
python evaluate.py --checkpoint runs/t4_r18_places_ft/f{0,1,2,3,4}_s0/last.pt
