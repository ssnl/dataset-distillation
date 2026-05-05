# CIFAR-10 Distilled Image Evaluation

이 폴더는 기존 dataset distillation 코드를 건드리지 않고, distilled image 100장으로 여러 CNN 모델의 CIFAR-10 test accuracy를 비교하기 위한 독립 평가 도구입니다.

## 목표

- distilled image를 만든 원래 모델인 `AlexCifarNet`의 test accuracy를 확인합니다.
- 같은 distilled image를 `VGG11`, `ResNet18`, `ResNet50`, `EfficientNet_B0`에도 적용해 test accuracy를 비교합니다.
- 원본 CIFAR-10 train set으로 학습한 baseline과 distilled image만 사용한 결과를 같은 CSV에 저장합니다.

## 권장 입력

가장 권장하는 입력은 원본 repo가 저장한 `results.pth`입니다. 이 파일에는 이미지뿐 아니라 label과 distilled learning rate가 함께 들어 있어서 dataset distillation의 의도를 가장 잘 보존합니다.

PNG 100장만 있는 경우에는 class index별 폴더 구조를 사용합니다.

```text
distilled_png/
  0/*.png
  1/*.png
  ...
  9/*.png
```

## 실행 순서

먼저 가장 단순한 모델로 전체 파이프라인을 확인합니다.

```bash
python eval_cifar10_distilled/src/train_eval.py \
  --model SimpleCNN \
  --train-source distilled \
  --distilled-path path/to/results.pth \
  --epochs 1
```

로컬 환경에서 DataLoader worker 문제가 나면 `--num-workers 0`을 명시합니다. 기본값도 디버깅 안정성을 위해 `0`입니다.

distilled image를 만든 원래 모델 기준 결과를 확인합니다.

```bash
python eval_cifar10_distilled/src/train_eval.py \
  --model AlexCifarNet \
  --train-source distilled \
  --distilled-path path/to/results.pth \
  --epochs 1
```

원본 CIFAR-10 train set으로 baseline을 만듭니다.

```bash
python eval_cifar10_distilled/src/train_eval.py \
  --model AlexCifarNet \
  --train-source full_cifar10 \
  --epochs 200
```

다른 모델도 같은 방식으로 반복합니다.

```bash
python eval_cifar10_distilled/src/train_eval.py --model VGG11 --train-source distilled --distilled-path path/to/results.pth --epochs 1
python eval_cifar10_distilled/src/train_eval.py --model ResNet18 --train-source distilled --distilled-path path/to/results.pth --epochs 1
python eval_cifar10_distilled/src/train_eval.py --model ResNet50 --train-source distilled --distilled-path path/to/results.pth --epochs 1
python eval_cifar10_distilled/src/train_eval.py --model EfficientNet_B0 --train-source distilled --distilled-path path/to/results.pth --epochs 1
```

결과 비교표를 출력합니다.

```bash
python eval_cifar10_distilled/src/compare_results.py
```

## 결과 파일

실행 결과는 기본적으로 아래 CSV에 누적됩니다.

```text
eval_cifar10_distilled/results/accuracy_summary.csv
```

컬럼은 `model`, `train_source`, `epochs`, `seed`, `test_accuracy`, `test_loss` 등을 포함합니다.

## 주의

- CIFAR-10 distillation의 원래 모델은 ImageNet용 `AlexNet`이 아니라 `AlexCifarNet`입니다.
- `distilled` 모드는 `results.pth`의 step과 learning rate를 replay합니다.
- 기본 초기화는 원본 repo와 맞추기 위해 `xavier`입니다.
- `distilled_folder` 모드는 이미지 100장만 있을 때 쓰는 fallback이므로, `results.pth` 기반 결과와 완전히 같은 의미는 아닙니다.
