"""Distilled image를 학습 루프에 넣기 위한 로더.

지원 형식은 두 가지다.
1. 원본 repo가 저장한 results.pth: label과 distilled learning rate까지 보존되므로 가장 권장한다.
2. class별 하위 폴더가 있는 이미지 폴더: 이미지 100장만 따로 있을 때 쓰는 fallback이다.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.247, 0.243, 0.261)


def cifar10_train_transform() -> transforms.Compose:
    """원본 CIFAR-10 train set에 사용할 기본 augmentation."""

    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def cifar10_test_transform() -> transforms.Compose:
    """CIFAR-10 test set과 이미지 폴더 fallback에 사용할 결정적 transform."""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


class ImageFolderDistilledDataset(Dataset):
    """이미지 폴더를 distilled dataset처럼 읽는다.

    폴더 구조는 아래처럼 class index를 하위 폴더명으로 두는 방식을 기대한다.
    distilled_png/0/*.png, distilled_png/1/*.png, ..., distilled_png/9/*.png
    """

    IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.transform = cifar10_test_transform()
        self.samples: List[Tuple[Path, int]] = []

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name)
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise ValueError(f"이미지를 찾지 못했습니다: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def load_distilled_steps(path: Union[str, Path], device: torch.device):
    """results.pth에서 distilled step 목록을 읽는다.

    각 step은 (data, label, lr) 형태다. data와 label은 학습에 바로 쓰도록 device로 옮긴다.
    lr이 없거나 이미지 폴더 기반 학습을 쓰는 경우에는 일반 SGD learning rate를 별도로 쓴다.
    """

    loaded = torch.load(path, map_location=device)
    steps = []
    for data, label, lr in loaded:
        data = data.to(device)
        label = label.to(device)
        lr = None if lr is None else torch.as_tensor(lr, device=device).float()
        steps.append((data, label, lr))
    if not steps:
        raise ValueError(f"비어 있는 distilled 결과입니다: {path}")
    return steps
