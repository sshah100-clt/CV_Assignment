"""Image transforms, cross-validation splits and data loaders."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def normalization(cfg):
    """Per-channel mean and standard deviation."""
    d = cfg['data']
    if d['normalize'] == 'imagenet':  # the statistics pretrained models were trained with
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return [0.5] * d['channels'], [0.5] * d['channels']


def build_transform(cfg, train):
    d = cfg['data']
    size = d['img_size']
    aug = d['augment']
    steps = []

    # Only Flower images are in colour. Converting all images to gray stops a
    # model from recognising Flower by its colour alone.
    if d['color'] == 'gray':
        steps.append(transforms.Grayscale(num_output_channels=d['channels']))

    if train and aug['random_resized_crop']:
        steps.append(transforms.RandomResizedCrop(size, scale=tuple(aug['random_resized_crop'])))
    elif d['resize_mode'] == 'squash':
        steps.append(transforms.Resize((size, size)))  # ignores the aspect ratio
    else:
        steps += [transforms.Resize(size), transforms.CenterCrop(size)]  # keeps the aspect ratio

    if train and aug['hflip']:
        steps.append(transforms.RandomHorizontalFlip())

    mean, std = normalization(cfg)
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


def relative_paths(dataset):
    return [str(Path(path).relative_to(dataset.root)) for path, _ in dataset.samples]


def split_indices(cfg, dataset):
    """Training and validation indices for the configured fold. The folds in
    splits/folds.json are made by scripts/make_splits.py."""
    folds = json.loads(Path(cfg['data']['split_file']).read_text())['assignment']
    fold_of_image = [folds[path] for path in relative_paths(dataset)]
    train_idx = [i for i, f in enumerate(fold_of_image) if f != cfg['data']['fold']]
    val_idx = [i for i, f in enumerate(fold_of_image) if f == cfg['data']['fold']]
    return train_idx, val_idx


def make_loader(dataset, cfg, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=shuffle,
                      num_workers=cfg['data']['num_workers'], pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed))


def build_train_val_loaders(cfg, seed):
    """Loaders for the training folds and the validation fold of data/train.
    Two ImageFolder objects over the same images let the two use different
    transforms (the starter notebook's random_split shared one transform)."""
    root = Path(cfg['data']['root']) / 'train'
    train_set = datasets.ImageFolder(root, transform=build_transform(cfg, train=True))
    val_set = datasets.ImageFolder(root, transform=build_transform(cfg, train=False))
    train_idx, val_idx = split_indices(cfg, train_set)
    train_loader = make_loader(Subset(train_set, train_idx), cfg, shuffle=True, seed=seed)
    val_loader = make_loader(Subset(val_set, val_idx), cfg)
    return train_loader, val_loader, train_set.classes


def build_test_loader(cfg, split='test'):
    """Loader for a labelled folder such as data/test, plus its class names and image paths."""
    dataset = datasets.ImageFolder(Path(cfg['data']['root']) / split, transform=build_transform(cfg, train=False))
    return make_loader(dataset, cfg), dataset.classes, relative_paths(dataset)
