# AI usage

## Tool

I used Claude Code (Anthropic) as a pair programmer in the terminal. It wrote and tested code on my laptop. I ran every experiment on a university GPU server with 4 NVIDIA RTX A5000 GPUs and checked the results before using them.

## How it helped

1. **Turning the notebook into scripts.** Claude Code wrote the training and evaluation scripts, one config file per experiment, and a launcher that runs jobs on all 4 GPUs. I reviewed the structure before running it. The first version was more complex than the experiments needed, so I had it cut down to only what they use (from about 1,850 to about 1,050 lines), until I could explain every file.
2. **Checking the dataset.** I asked for a check of the data before any training. It found that only Flower images are in colour. It also found that image size and colour mode alone give 27% test accuracy. The training set has 14 groups of duplicate images, and 4 test images also appear in training. Because of this, all models use grayscale, duplicates always stay in the same fold, and I report test accuracy with and without those 4 images.
3. **Finding a bug in the starter code.** The starter used `random_split`, which gives the training and validation sets the same transform. If I had added augmentation there, the validation images would have been augmented too. The split was rewritten so the two sets use separate transforms.
4. **Statistical tests and reproducing the results.** Claude Code wrote the McNemar tests that compare two models on the same images. The tests read the saved prediction files, so every p-value in the report can be recomputed with `scripts/oof.py`. Before submitting, I had the whole pipeline re-run from scratch in a new folder with `run_all.sh` (465 runs). Every number in the report comes from that run.

## Where it was wrong

- **RGB input.** The first plan suggested colour images, because colour usually helps with scenes. The data check then showed that only Flower is in colour. A colour model could spot Flower just from that, which says nothing about scenes. So every model uses grayscale.
- **Changing too much at once.** The first config for "E2" changed the optimizer, the learning-rate schedule and the number of epochs together. That made it impossible to tell which change helped, so it was split into A2, A3 and A4.
- **A wrong prediction.** After the 60-epoch run, Claude Code said longer training would barely help. The 150-epoch run gained 0.9 points and was better in all three seeds.
- **Bugs found in review.** The frozen-backbone option still updated the BatchNorm statistics, so the backbone was not really frozen. The first validation split did not keep duplicates together. Picking the best epoch on validation made the accuracy look higher than it was. All three were fixed before the reported runs. After a frozen-backbone run, only the final layer (`fc`) differs from the pretrained weights.
- **A result that did not hold up.** In the first round of runs, keeping the aspect ratio looked 2.0 points better than squashing the image. In the full re-run it made no real difference, so the report says that instead.

## Decisions I made

- **Stopping at ResNet-18.** Claude Code suggested waiting for ResNet-50 and ConvNeXt. I stopped at ResNet-18. I did not test model size. I thought the pretraining comparison mattered more. A frozen Places365 ResNet-18 had already beaten a fully fine-tuned ImageNet ResNet-18 with the same architecture and the same folds.
- **Reporting only the re-run.** Instead of reporting the first round of runs, I re-ran everything with the final code and used only those numbers. That way the results and the code in the repository match.
