"""CIFAR-10 평가에 사용할 모델 모음.

이 파일의 목적은 모델 선택 로직을 한 곳에 모아 두는 것이다.
나중에 모델을 추가하고 싶으면 train_eval.py를 고치지 않고 이 파일만 확장하면 된다.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def _make_group_norm(channels: int) -> nn.GroupNorm:
    """수정: tiny distilled batch에서도 안정적인 normalization을 쓰기 위한 helper.

    BatchNorm은 batch 10장짜리 distilled step에서 running mean/var가 불안정해질 수 있다.
    그래서 CIFAR용 비교 모델은 pretrained 모델을 불러오지 않고, GroupNorm 기반으로 직접 정의한다.
    """

    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    """수정: CIFAR용 feature extractor에서 반복해서 쓰는 conv-normalization-activation block.

    nn.Linear를 대체하려는 블록이 아니다. ConvNormAct는 이미지의 공간 패턴을 뽑는 앞단이고,
    nn.Linear는 마지막 pooled feature를 10개 class logit으로 바꾸는 classifier로 계속 사용한다.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            _make_group_norm(out_channels),
            nn.ReLU(inplace=True),
        )


class SimpleCNN(nn.Module):
    """가장 먼저 sanity check를 하기 위한 작은 CNN.

    복잡한 모델을 돌리기 전에 데이터 로더, 학습 루프, 평가 루프가 정상인지 확인하는 용도다.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class AlexCifarNet(nn.Module):
    """원본 repo가 CIFAR-10 distillation에 사용한 AlexNet 계열 모델.

    README의 CIFAR-10 예시는 ImageNet용 AlexNet이 아니라 AlexCifarNet을 사용한다.
    따라서 distilled image를 만든 원래 모델과 비교하려면 이 모델명을 기준으로 삼는 것이 맞다.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        return self.classifier(x)


class CifarVGG(nn.Module):
    """CIFAR-10용 VGGNet.

    torchvision VGG는 ImageNet 224x224 입력을 기준으로 4096 차원 classifier를 사용한다.
    32x32 distilled image 실험에는 과하게 크고 불안정해서, VGG의 "3x3 conv stack" 성질은 유지하되
    입력 크기와 classifier만 CIFAR 친화적으로 바꾼다.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # 수정: torchvision ImageNet VGG를 불러오지 않고 CIFAR-10 32x32용 VGG block을 직접 정의한다.
            ConvNormAct(3, 64, kernel_size=3),
            nn.MaxPool2d(2),
            ConvNormAct(64, 128, kernel_size=3),
            nn.MaxPool2d(2),
            ConvNormAct(128, 256, kernel_size=3),
            ConvNormAct(256, 256, kernel_size=3),
            nn.MaxPool2d(2),
            ConvNormAct(256, 512, kernel_size=3),
            ConvNormAct(512, 512, kernel_size=3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


class CifarBasicBlock(nn.Module):
    """수정: CIFAR용 ResNet18/34 basic residual block.

    ImageNet ResNet을 불러오는 대신 3x3 stem, no maxpool, GroupNorm으로 직접 구성한다.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_group_norm(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _make_group_norm(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CifarBottleneckBlock(nn.Module):
    """수정: CIFAR용 ResNet50 bottleneck block."""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        width = out_channels
        self.conv1 = ConvNormAct(in_channels, width, kernel_size=1)
        self.conv2 = ConvNormAct(width, width, kernel_size=3, stride=stride)
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False),
            _make_group_norm(out_channels * self.expansion),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                _make_group_norm(out_channels * self.expansion),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CifarResNet(nn.Module):
    """수정: CIFAR-10 32x32 입력에 맞춘 ResNet.

    torchvision.models.resnet*는 ImageNet 224x224 기준의 7x7 stride-2 stem과 maxpool을 포함한다.
    여기서는 residual block이라는 ResNet의 핵심 성질은 유지하고,
    CIFAR-10에서 일반적으로 쓰는 3x3 stride-1 stem으로 시작한다.
    """

    def __init__(self, block, layers, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64
        self.stem = ConvNormAct(3, 64, kernel_size=3)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            blocks.append(block(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class SqueezeExcite(nn.Module):
    """수정: EfficientNet 계열의 channel attention block을 직접 정의한다."""

    def __init__(self, channels: int, squeeze_channels: int):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.fc1(scale), inplace=True)
        scale = torch_sigmoid(self.fc2(scale))
        return x * scale


def torch_sigmoid(x):
    """수정: 함수 형태로 분리해 EfficientNet block의 의도를 읽기 쉽게 둔다."""

    return x.sigmoid()


class MBConv(nn.Module):
    """수정: CIFAR용 EfficientNet의 mobile inverted bottleneck block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormAct(in_channels, hidden_channels, kernel_size=1))
        layers.extend(
            [
                ConvNormAct(hidden_channels, hidden_channels, kernel_size=3, stride=stride, groups=hidden_channels),
                SqueezeExcite(hidden_channels, max(1, in_channels // 4)),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                _make_group_norm(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class CifarEfficientNet(nn.Module):
    """수정: CIFAR-10 32x32 입력용 작은 EfficientNet-B0 스타일 모델.

    torchvision EfficientNet은 ImageNet 입력을 전제로 한다. 여기서는 pretrained를 전혀 쓰지 않고,
    MBConv + squeeze-excitation이라는 EfficientNet의 핵심 성질은 유지하되,
    stride를 보수적으로 써서 32x32 synthetic image 정보가 초반에 너무 빨리 사라지지 않게 했다.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvNormAct(3, 32, kernel_size=3, stride=1),
            MBConv(32, 16, stride=1, expand_ratio=1),
            MBConv(16, 24, stride=1, expand_ratio=6),
            MBConv(24, 24, stride=1, expand_ratio=6),
            MBConv(24, 40, stride=2, expand_ratio=6),
            MBConv(40, 40, stride=1, expand_ratio=6),
            MBConv(40, 80, stride=2, expand_ratio=6),
            MBConv(80, 80, stride=1, expand_ratio=6),
            MBConv(80, 112, stride=1, expand_ratio=6),
            MBConv(112, 192, stride=2, expand_ratio=6),
            ConvNormAct(192, 320, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """문자열 이름으로 모델을 만든다.

    수정: VGG/ResNet/EfficientNet은 torchvision pretrained 모델을 불러오지 않는다.
    이 파일 안에 직접 정의한 CIFAR-10용 scratch 모델을 반환한다.
    주의: "원 논문의 ImageNet용 모델 그대로"가 아니라 "CIFAR-10 입력에 맞춘 같은 계열의 모델"이다.
    """

    key = name.lower()
    if key in {"simplecnn", "simple_cnn"}:
        return SimpleCNN(num_classes=num_classes)
    if key in {"alexcifarnet", "alex_cifar_net", "alexnet"}:
        return AlexCifarNet(num_classes=num_classes)
    if key in {"vgg", "vgg11", "vggnet"}:
        return CifarVGG(num_classes=num_classes)
    if key == "resnet18":
        # 수정: pretrained/ImageNet 모델을 불러오지 않고 CIFAR용 ResNet18을 직접 만든다.
        return CifarResNet(CifarBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if key == "resnet50":
        # 수정: pretrained/ImageNet 모델을 불러오지 않고 CIFAR용 ResNet50을 직접 만든다.
        return CifarResNet(CifarBottleneckBlock, [3, 4, 6, 3], num_classes=num_classes)
    if key in {"efficientnet", "efficientnet_b0"}:
        # 수정: pretrained/ImageNet EfficientNet을 불러오지 않고 CIFAR용 EfficientNet 스타일 모델을 직접 만든다.
        return CifarEfficientNet(num_classes=num_classes)
    raise ValueError(f"지원하지 않는 모델입니다: {name}")


def apply_repo_initialization(model: nn.Module, init_type: str = "xavier", init_param: float = 1.0) -> nn.Module:
    """원본 dataset-distillation repo의 기본 초기화와 맞춘다.

    distilled image는 모델 초기화에 민감하다. 원본 repo의 기본값이 xavier이므로,
    여기서도 같은 분포에서 시작해야 AlexCifarNet 비교가 더 의미 있다.
    """

    def init_module(module: nn.Module) -> None:
        classname = module.__class__.__name__
        if classname.startswith("Conv") or classname == "Linear":
            if getattr(module, "bias", None) is not None:
                init.constant_(module.bias, 0.0)
            if getattr(module, "weight", None) is not None:
                if init_type == "xavier":
                    init.xavier_normal_(module.weight, gain=init_param)
                elif init_type == "kaiming":
                    init.kaiming_normal_(module.weight, a=init_param, mode="fan_in")
                elif init_type == "default":
                    module.reset_parameters()
                else:
                    raise ValueError(f"지원하지 않는 초기화 방식입니다: {init_type}")
        elif "Norm" in classname:
            if getattr(module, "weight", None) is not None:
                module.weight.data.fill_(1)
            if getattr(module, "bias", None) is not None:
                module.bias.data.zero_()

    model.apply(init_module)
    return model


def count_parameters(model: nn.Module) -> int:
    """결과 로그에 모델 규모를 함께 남기기 위한 helper."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
