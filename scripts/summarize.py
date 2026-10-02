"""Cross-validation results per experiment, best first.

    python scripts/summarize.py
    python scripts/summarize.py --filter t    # only experiments whose name starts with 't'

CV acc: accuracy of the final-epoch predictions on all 2,400 training images,
each predicted by the model whose training folds did not contain it; mean and
standard deviation over seeds. Train acc: final training accuracy (a large gap
to CV acc means overfitting). Also writes runs/summary.md and runs/summary.csv.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def cv_accuracy(runs):
    """Accuracy over all folds of one seed."""
    correct = total = 0
    for run in runs:
        pred = np.load(run['dir'] / 'val_predictions.npz')
        correct += (pred['probs'].argmax(1) == pred['labels']).sum()
        total += len(pred['labels'])
    return correct / total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--filter', default='')
    args = p.parse_args()

    experiments = defaultdict(list)
    for path in Path('runs').glob(f'{args.filter}*/f*_s*/metrics.json'):
        run = json.loads(path.read_text())
        run['dir'] = path.parent
        experiments[run['experiment']].append(run)

    rows = []  # (cv acc, markdown line, csv row)
    for name, runs in experiments.items():
        seeds = sorted({r['seed'] for r in runs})
        accs = [cv_accuracy([r for r in runs if r['seed'] == s]) for s in seeds
                if sum(r['seed'] == s for r in runs) == 5]  # only seeds with all 5 folds done
        if accs:
            std = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
            train = np.mean([r['final_train_acc'] for r in runs])
            params = runs[0]['params'] / 1e6
            line = f'| {name} | {len(accs)} | {100 * np.mean(accs):.2f} ± {100 * std:.2f} | {100 * train:.1f} | {params:.2f} |'
            row = [name, len(accs), round(100 * np.mean(accs), 2), round(100 * std, 2), round(100 * train, 1), round(params, 2)]
            rows.append((np.mean(accs), line, row))
    rows.sort(key=lambda r: -r[0])

    lines = ['| Experiment | Seeds | CV acc (%) | Train acc (%) | Params (M) |', '|---|---:|---:|---:|---:|']
    lines += [line for _, line, _ in rows]
    print('\n'.join(lines))
    Path('runs/summary.md').write_text('\n'.join(lines) + '\n')
    with open('runs/summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'seeds', 'cv_acc', 'cv_acc_std', 'train_acc', 'params_millions'])
        writer.writerows(row for _, _, row in rows)


if __name__ == '__main__':
    main()
