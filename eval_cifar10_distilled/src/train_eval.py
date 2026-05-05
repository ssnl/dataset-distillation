"""CIFAR-10 distilled image와 원본 CIFAR-10 학습 결과를 비교하는 실행 파일.

설계 원칙:
- 기존 repo 코드는 건드리지 않는다.
- 한 번 실행할 때 한 모델과 한 train_source만 평가한다.
- 모든 결과는 CSV에 누적해서 나중에 표로 비교한다.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from distilled_dataset import (  # noqa: E402
    ImageFolderDistilledDataset,
    cifar10_test_transform,
    cifar10_train_transform,
    load_distilled_steps,
)
from models import apply_repo_initialization, build_model, count_parameters  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 distilled image 평가")
    parser.add_argument("--model", default="SimpleCNN", help="SimpleCNN, AlexCifarNet, VGG11, ResNet18, ResNet50, EfficientNet_B0")
    parser.add_argument("--train-source", choices=["distilled", "distilled_folder", "full_cifar10"], required=True)
    parser.add_argument("--distilled-path", default=None, help="results.pth 경로. train-source=distilled일 때 필요")
    parser.add_argument("--distilled-folder", default=None, help="class별 이미지 폴더. train-source=distilled_folder일 때 필요")
    parser.add_argument("--data-root", default="./data/cifar10", help="CIFAR-10 다운로드/저장 경로")
    parser.add_argument("--output-csv", default="eval_cifar10_distilled/results/accuracy_summary.csv")
    parser.add_argument("--epochs", type=int, default=1, help="학습 epoch 수")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 수. 로컬 디버깅 안정성을 위해 기본값은 0")
    parser.add_argument("--lr", type=float, default=0.01, help="full_cifar10 또는 distilled_folder 학습용 learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--init", choices=["xavier", "kaiming", "default"], default="xavier")
    parser.add_argument("--init-param", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 등")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    """사용자가 지정한 device를 torch.device로 변환한다."""

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    """모델 초기화와 dataloader shuffle을 최대한 재현 가능하게 만든다."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """CIFAR-10 test accuracy와 평균 loss를 계산한다."""

    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            loss_sum += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.numel()

    return correct / total, loss_sum / total


def train_on_full_cifar10(model, train_loader, args, device):
    """원본 CIFAR-10 train set으로 일반적인 supervised training을 수행한다."""

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        # non-blocking=True 옵션은 데이터 로딩과 GPU 연산을 병렬로 수행할 수 있게 해준다.
            # CPU에서 데이터를 준비하는 동안 GPU는 이전 배치에 대한 연산을 계속할 수 있다.
            # 이렇게 하면 전체 학습 시간이 줄어들 수 있다. 
            # 단, 이 옵션이 항상 성능 향상을 보장하는 것은 아니며, 시스템의 I/O 성능과 GPU 연산 속도에 따라 달라질 수 있다.
        model.train()
        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
        print(f"[full_cifar10] epoch {epoch + 1}/{args.epochs} 완료")


def train_on_distilled_steps(model, steps, args):
    """results.pth의 distilled step을 순서대로 replay해서 모델을 학습한다.

    원본 distillation 코드는 step마다 저장된 learning rate로 gradient descent를 수행한다.
    여기서도 같은 의도를 보존하기 위해 lr이 있으면 그 값을 사용하고, 없으면 --lr 값을 사용한다.
    """

    for epoch in range(args.epochs):
        model.train()
        for step_index, (data, label, distilled_lr) in enumerate(steps):
            lr_value = args.lr if distilled_lr is None else float(distilled_lr.reshape(-1)[0].item())
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_value)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), label)
            loss.backward()
            optimizer.step()
            print(f"[distilled] epoch {epoch + 1}/{args.epochs}, step {step_index + 1}/{len(steps)}, lr={lr_value:.6g}")


def train_on_distilled_folder(model, args, device):
    """PNG/JPG 100장만 있을 때 사용하는 fallback 학습 방식."""

    dataset = ImageFolderDistilledDataset(args.distilled_folder)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epochs):
        model.train()
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
        print(f"[distilled_folder] epoch {epoch + 1}/{args.epochs} 완료")


def append_result(args, accuracy, loss, param_count, elapsed_sec):
    """한 번의 실행 결과를 CSV에 누적한다."""

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = output_csv.exists()

    fieldnames = [
        "model",
        "train_source",
        "epochs",
        "seed",
        "test_accuracy",
        "test_loss",
        "parameters",
        "init",
        "init_param",
        "elapsed_sec",
        "distilled_path",
        "distilled_folder",
    ]

    with output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "model": args.model,
                "train_source": args.train_source,
                "epochs": args.epochs,
                "seed": args.seed,
                "test_accuracy": f"{accuracy:.6f}",
                "test_loss": f"{loss:.6f}",
                "parameters": param_count,
                "init": args.init,
                "init_param": args.init_param,
                "elapsed_sec": f"{elapsed_sec:.2f}",
                "distilled_path": args.distilled_path or "",
                "distilled_folder": args.distilled_folder or "",
            }
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    start_time = time.time()
    model = build_model(args.model)
    apply_repo_initialization(model, init_type=args.init, init_param=args.init_param)
    model = model.to(device)
    param_count = count_parameters(model)
    # 수정: 사용자가 요청한 "모델을 어디에서 불러오는지"를 실행 로그에서 바로 확인하게 한다.
    # build_model은 eval_cifar10_distilled/src/models.py 안의 직접 정의 모델을 반환한다.
    print(f"built_class={model.__class__.__name__} from eval_cifar10_distilled/src/models.py")
    print(f"model={args.model}, parameters={param_count:,}, device={device}")

    test_dataset = datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=cifar10_test_transform(),
    )
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    if args.train_source == "full_cifar10":
        train_dataset = datasets.CIFAR10(
            root=args.data_root,
            train=True,
            download=True,
            transform=cifar10_train_transform(),
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        train_on_full_cifar10(model, train_loader, args, device)
    elif args.train_source == "distilled":
        if args.distilled_path is None:
            raise ValueError("--train-source distilled 에서는 --distilled-path 가 필요합니다.")
        steps = load_distilled_steps(args.distilled_path, device)
        train_on_distilled_steps(model, steps, args)
    elif args.train_source == "distilled_folder":
        if args.distilled_folder is None:
            raise ValueError("--train-source distilled_folder 에서는 --distilled-folder 가 필요합니다.")
        train_on_distilled_folder(model, args, device)

    accuracy, loss = evaluate(model, test_loader, device)
    elapsed_sec = time.time() - start_time
    append_result(args, accuracy, loss, param_count, elapsed_sec)

    print(f"test_accuracy={accuracy * 100:.2f}%, test_loss={loss:.4f}")
    print(f"결과 저장: {args.output_csv}")


if __name__ == "__main__":
    main()
