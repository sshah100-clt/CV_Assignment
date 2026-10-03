# ITCS 6169/8169 Assignment 1: 16-class scene recognition with CNNs

**Result:** 96.5% test accuracy (95% CI 94.7-98.3%; 96.3% on the 15 scene
classes; 96.5% without the 4 test images that also occur in the training set)
and 95.4 ± 0.2% cross-validation accuracy.

**Model:** average of five ResNet-18 networks pretrained on Places365 and
fine-tuned, one per cross-validation fold (`configs/best.yaml`).

**Checkpoints:** [GitHub Release v1.0](https://github.com/sshah100-clt/CV_Assignment/releases/tag/v1.0)
(five files, `fold0_last.pt` to `fold4_last.pt`).

## How to run

### 1. Get the code and install the dependencies

```bash
git clone https://github.com/sshah100-clt/CV_Assignment.git
cd CV_Assignment
pip install -r requirements.txt
```

### 2. Add the dataset

Download the dataset from the link in the assignment handout and put the
images in `data/` inside the repository:

```text
data/train/<class>/*.jpg     2,400 images, 16 class folders
data/test/<class>/*.jpg      400 images, 16 class folders
```

Check that the images match the cross-validation folds (prints `OK`):

```bash
python -c "import json,pathlib; f=json.load(open('splits/folds.json'))['assignment']; imgs={str(p.relative_to('data/train')) for p in pathlib.Path('data/train').glob('*/*.jpg')}; print('OK' if imgs==set(f) else 'MISMATCH')"
```

Pretrained weights (ImageNet and Places365) are downloaded automatically on
first use.

### 3a. Evaluate the released final model (minutes, one GPU or CPU)

Download the five checkpoints from the
[Release](https://github.com/sshah100-clt/CV_Assignment/releases/tag/v1.0) and place them like this:

```bash
for f in 0 1 2 3 4; do
  mkdir -p runs/t4_r18_places_ft/f${f}_s0
  mv fold${f}_last.pt runs/t4_r18_places_ft/f${f}_s0/last.pt
done
python evaluate.py --checkpoint runs/t4_r18_places_ft/f0_s0/last.pt runs/t4_r18_places_ft/f1_s0/last.pt \
    runs/t4_r18_places_ft/f2_s0/last.pt runs/t4_r18_places_ft/f3_s0/last.pt runs/t4_r18_places_ft/f4_s0/last.pt
```

This prints the test accuracy (96.5%) and writes
`runs/t4_r18_places_ft/f0_s0/test_results.csv`.

### 3b. Train only the final model (a few minutes on 4 GPUs)

`configs/best.yaml` is the selected recipe (T4, `configs/transfer/t4_r18_places_ft.yaml`).
This trains new fold models into `runs/best/` and evaluates them; expect a test
accuracy within a few tenths of a point of 96.5%.

```bash
python scripts/launch.py configs/best.yaml --seeds 0       # 5 fold models -> runs/best/
python evaluate.py --checkpoint runs/best/f0_s0/last.pt runs/best/f1_s0/last.pt \
    runs/best/f2_s0/last.pt runs/best/f3_s0/last.pt runs/best/f4_s0/last.pt
```

### 3c. Reproduce every experiment and table (about 1.5 hours on 4 GPUs)

The repository already contains the reported results in `runs/`, and
`run_all.sh` skips every run that has finished. Move them aside first, so
that everything is trained again:

```bash
mv runs runs_reported                  # keeps the reported results for comparison
bash run_all.sh                        # default: GPUs 0-3, 3 runs per GPU
GPUS="0 1" JOBS=2 bash run_all.sh      # or: choose the GPUs and runs per GPU
```

This trains all 465 runs (31 experiments x 5 folds x 3 seeds), then writes:

| File | Contents |
|---|---|
| `runs/summary.csv` | cross-validation accuracy of every experiment |
| `runs/comparisons.csv` | paired McNemar tests used in the report |
| `runs/t4_r18_places_ft/per_class.csv` | per-class accuracy and most common confusion of the final recipe |
| `runs/t4_r18_places_ft/f0_s0/test_results.csv` | test accuracy of the final model |
| `runs/metadata_baseline.txt` | test accuracy from image size and colour alone |

Finished runs are skipped, so after an interruption or a failure, run the
same command again. A failed run's log is in `runs/logs/`. Figure 1 of the
report is drawn from `runs/summary.csv` with `python scripts/plot_figure1.py`.

### Running single experiments

```bash
python train.py --config configs/scratch/a6_res128.yaml --fold 0 --seed 0    # one run
python scripts/launch.py configs/scratch/a6_res128.yaml                      # 5 folds x 3 seeds
python scripts/launch.py configs/scratch/a6_res128.yaml --grid optim.lr=0.001,0.003   # a grid
python scripts/summarize.py                                  # cross-validation table
python scripts/oof.py --compare a6_res128 a12_longer         # paired McNemar test
python scripts/oof.py --experiment t4_r18_places_ft          # per-class accuracy and confusions
```

`train.py --set key=value` overrides any config value. `scripts/launch.py`
skips runs that already have results in `runs/`; `train.py` always trains.

## Dataset notes

The 16 classes are the 15-Scene benchmark (Lazebnik et al., CVPR 2006) plus a
Flower class: 2,400 training images (150 per class) and 400 test images (25 per
class). Checking the data before training showed:

- **Only Flower is in colour.** All models convert images to grayscale, so
  colour cannot be used as a shortcut.
- **Image size reveals the class.** For example, 8 classes are always 256x256
  and Suburb is always 330x220. A lookup on image size and colour mode alone
  reaches 27% test accuracy (`scripts/metadata_baseline.py`; chance is 6.25%).
- **Duplicates.** The training set has 14 groups of identical images, and 4
  test images also occur in it (`splits/duplicates.json`).

## Validation protocol

Every configuration is trained with 5-fold cross-validation over the 2,400
training images (`splits/folds.json`; stratified by class, with duplicates in
the same fold) and 3 seeds. It is scored by the accuracy of the final-epoch
predictions on all 2,400 images, each predicted by the model that did not train
on it. Differences between configurations are tested with paired McNemar
tests. The test set was used once, for the final model. Results of every
experiment are in `runs/summary.csv`.

## Repository layout

```text
train.py              trains one model on one fold
evaluate.py           test accuracy of one model or an average of several
run_all.sh            reproduces every experiment and result table
configs/base.yaml     all settings and their defaults
configs/scratch/      CNNs trained from scratch (A0-A12), one change per config
configs/transfer/     ResNet-18 pretrained on ImageNet or Places365 (T1-T4)
configs/analysis/     aspect-ratio experiment (X2)
scenecnn/             data loading, models, training loop, config loading
scripts/              CV splits, job launcher, result summaries, metadata baseline, Figure 1
splits/               the CV folds and the duplicate list
runs/                 logs, metrics, predictions and result tables of every run
report/               the two-page report
```

## Environment

All reported results come from one run of `run_all.sh` on 4 NVIDIA RTX A5000
GPUs (24 GB each) with Python 3.12.3, PyTorch 2.12.1 and CUDA 13.0 (recorded
in every run's `metrics.json`). Seeds are fixed, but GPU training is not
bit-for-bit deterministic, so a rerun gives values within a few tenths of a
point.
