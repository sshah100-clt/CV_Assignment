"""Training and prediction loops."""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def predict(model, loader, device):
    """Class probabilities and labels for all images, in loader order."""
    model.eval()
    probs, labels = [], []
    for images, batch_labels in loader:
        probs.append(model(images.to(device)).softmax(1).cpu())
        labels.append(batch_labels)
    return torch.cat(probs), torch.cat(labels)


def validate(model, loader, device):
    """Cross-entropy, accuracy and probabilities on a labelled loader."""
    probs, labels = predict(model, loader, device)
    loss = F.nll_loss(probs.clamp_min(1e-12).log(), labels).item()
    acc = (probs.argmax(1) == labels).float().mean().item()
    return loss, acc, probs, labels


def build_optimizer(model, head, cfg):
    """The new classification layer (`head`) and the rest of the network are
    separate parameter groups, so pretrained layers can get a smaller learning
    rate, or be frozen."""
    o = cfg['optim']
    head_params = list(head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    groups = [{'params': head_params, 'lr': o['lr']}]
    if cfg['model']['freeze_backbone']:
        for p in backbone:
            p.requires_grad_(False)
    else:
        groups.append({'params': backbone, 'lr': o['lr'] * o['backbone_lr_mult']})

    optimizer = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[o['name']]
    return optimizer(groups, weight_decay=o['weight_decay'])


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Learning-rate factor per step: linear warmup, then constant or cosine decay."""
    s = cfg['schedule']
    total = cfg['train']['epochs'] * steps_per_epoch
    warmup = s['warmup_epochs'] * steps_per_epoch

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        if s['name'] == 'cosine':
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def mix_batch(images, labels, t):
    """Mixup (Zhang et al., 2018) or CutMix (Yun et al., 2019), chosen at random per batch.
    Returns the mixed images, the two label sets and the weight lam of the first set."""
    mixup, cutmix = t['mixup_alpha'], t['cutmix_alpha']
    if mixup == 0 and cutmix == 0:
        return images, labels, labels, 1.0
    perm = torch.randperm(len(images), device=images.device)
    if cutmix == 0 or (mixup > 0 and random.random() < 0.5):
        lam = np.random.beta(mixup, mixup)
        return lam * images + (1 - lam) * images[perm], labels, labels[perm], lam

    # CutMix: paste a random box from another image; lam is the area left unchanged.
    lam = np.random.beta(cutmix, cutmix)
    h, w = images.shape[-2:]
    bh, bw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
    y, x = random.randrange(h - bh + 1), random.randrange(w - bw + 1)
    images = images.clone()
    images[:, :, y:y + bh, x:x + bw] = images[perm, :, y:y + bh, x:x + bw]
    return images, labels, labels[perm], 1 - bh * bw / (h * w)


def train_one_epoch(model, head, loader, optimizer, scheduler, cfg, device):
    """One pass over the training data; returns mean loss and accuracy."""
    t = cfg['train']
    criterion = nn.CrossEntropyLoss(label_smoothing=t['label_smoothing'])
    if cfg['model']['freeze_backbone']:
        model.eval()   # keeps the pretrained BatchNorm statistics fixed
        head.train()
    else:
        model.train()

    total_loss = correct = seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images, labels_a, labels_b, lam = mix_batch(images, labels, t)
        logits = model(images)
        loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred = logits.argmax(1)
        correct += (lam * (pred == labels_a) + (1 - lam) * (pred == labels_b)).sum().item()
        total_loss += loss.item() * len(images)
        seen += len(images)
    return total_loss / seen, correct / seen
