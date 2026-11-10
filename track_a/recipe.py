"""Your Track A recipe: how the fixed model is trained on your selection."""

import torch
import torch.nn.functional as F
from torchvision import transforms as T

BATCH_SIZE = 64


def train_transform():
    """A function that turns one training photo (a PIL image) into a 224 x 224 one."""
    return T.Compose([
        # 1. 랜드마크 훼손을 막고 구도를 보존하기 위해 scale 하한을 0.45로 설정
        T.RandomResizedCrop(224, scale=(0.45, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        # 2. 도쿄 등의 이미지 특징 손실을 막기 위해 Blur는 제거하고, 적절한 색상/조명 변형만 적용
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    ])


def sample_weights(cities, paths):
    """How often each photo in your selection is drawn."""
    return None


def make_optimizer(model):
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
