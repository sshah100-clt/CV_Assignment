"""How far can a classifier get without looking at the pixels?

Predicts each test image's class from its width, height and colour mode alone,
using the most common training class for each (width, height, mode)
combination. Accuracy above chance (1/16 = 6.25%) shows that image size and
colour reveal the class in this dataset.

    python scripts/metadata_baseline.py
"""
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def load(split):
    """(width, height, mode) and class of every image in data/<split>."""
    items = []
    for path in sorted(Path('data', split).glob('*/*.jpg')):
        with Image.open(path) as image:
            items.append((image.size + (image.mode,), path.parent.name))
    return items


def main():
    train, test = load('train'), load('test')
    votes = defaultdict(Counter)
    for key, label in train:
        votes[key][label] += 1
    most_common_class = Counter(label for _, label in train).most_common(1)[0][0]

    correct = 0
    for key, label in test:
        guess = votes[key].most_common(1)[0][0] if key in votes else most_common_class
        correct += guess == label
    print(f'metadata-only test accuracy: {correct}/{len(test)} = {correct / len(test):.3f} (chance 0.0625)')


if __name__ == '__main__':
    main()
