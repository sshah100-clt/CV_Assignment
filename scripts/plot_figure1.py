"""Draw Figure 1 of the report: how CV accuracy grows as the from-scratch CNN
gains one change at a time. Reads runs/summary.csv (scripts/summarize.py).

    python scripts/plot_figure1.py      # writes report/figures/fig1_scratch_ladder.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# The cumulative from-scratch steps, in order, with their axis labels.
STEPS = [
    ('a0_baseline', 'Starter\nTNet'),
    ('a1_simple_cnn', 'Deeper\nCNN'),
    ('a2_epochs', '60\nepochs'),
    ('a3_cosine', 'Cosine\nLR'),
    ('a4_adamw', 'AdamW'),
    ('a5_aug', 'Crop +\nflip'),
    ('a6_res128', '128 px\ninput'),
    ('a12_longer', '150\nepochs'),
]


def main():
    with open('runs/summary.csv') as f:
        cv_acc = {row['experiment']: float(row['cv_acc']) for row in csv.DictReader(f)}
    acc = [cv_acc[name] for name, _ in STEPS]

    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(3.4, 1.95))  # inches; same size as in the report
    x_last = len(STEPS)

    # First and last bars are totals (grey); the steps between are gains (green).
    ax.bar(0, acc[0], color='#b3b3b3', width=0.65)
    ax.text(0, acc[0] + 1, f'{acc[0]:.1f}', ha='center', va='bottom', fontsize=7)
    for i in range(1, len(acc)):
        low, high = sorted((acc[i - 1], acc[i]))
        ax.bar(i, high - low, bottom=low, color='#4a9a5c', width=0.65)
        ax.plot([i - 1.33, i - 0.33], [acc[i - 1]] * 2, ':', color='grey', lw=0.6)
        ax.text(i, high + 1, f'{acc[i] - acc[i - 1]:+.1f}', ha='center', va='bottom', fontsize=7)
    ax.plot([x_last - 1.33, x_last - 0.33], [acc[-1]] * 2, ':', color='grey', lw=0.6)
    ax.bar(x_last, acc[-1], color='#b3b3b3', width=0.65)
    ax.text(x_last, acc[-1] + 1, f'{acc[-1]:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(range(x_last + 1), [label for _, label in STEPS] + ['Best\nscratch'], fontsize=5.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel('CV accuracy (%)', fontsize=7)
    ax.tick_params(axis='y', labelsize=7, width=0.6, length=2.5)
    ax.tick_params(axis='x', width=0.6, length=2.5, pad=1.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.6)
    fig.tight_layout(pad=0.2)
    out = Path('report/figures/fig1_scratch_ladder.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
