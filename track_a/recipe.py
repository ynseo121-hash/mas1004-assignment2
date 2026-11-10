"""Your Track A recipe: how the fixed model is trained on your selection."""

import torch
import torch.nn.functional as F
from torchvision import transforms as T

BATCH_SIZE = 64


def train_transform():
    """A function that turns one training photo (a PIL image) into a 224 x 224 one."""
    return T.Compose([
        # 1. 지나친 자르기를 막기 위해 scale 하한을 0.35 -> 0.5로 상향 (랜드마크/건물 훼손 방지)
        T.RandomResizedCrop(224, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        # 2. 날씨, 조명, 계절 차이를 반영하기 위해 ColorJitter 약간 강화
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        # 3. 블러(Blur)를 추가해 서소문/표지판 등 미세한 소음 및 화질 차이에 강건하게 변경
        T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    ])


def sample_weights(cities, paths):
    """How often each photo in your selection is drawn."""
    return None


def make_optimizer(model):
    # 가중치 감쇄(weight_decay)를 0.05 -> 0.01로 조정하여 일반화 능력 향상
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)


def make_scheduler(optimizer, total_steps):
    """Stepped once after every training step. total_steps is BUDGET // BATCH_SIZE."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4, total_steps=total_steps, pct_start=0.15
    )


def mix(images, labels):
    """Change a batch on the GPU before the model sees it."""
    return images, labels


def loss_fn(logits, targets):
    return F.cross_entropy(logits, targets, label_smoothing=0.1)


def on_step(step, total_steps, model, optimizer):
    """Called after every step, with the step number from 0. Returns nothing."""


def final_model(model):
    """The model to hand in, at the end of training. It must still be the same ResNet18."""
    return model
