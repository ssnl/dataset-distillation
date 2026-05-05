"""accuracy_summary.csv를 사람이 읽기 쉬운 비교표로 출력한다."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 평가 결과 비교")
    parser.add_argument("--csv", default="eval_cifar10_distilled/results/accuracy_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"결과 CSV가 없습니다: {csv_path}")

    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            row["test_accuracy"] = float(row["test_accuracy"])
            rows.append(row)

    # 같은 모델 안에서 train_source별 최고 accuracy를 모아 비교한다.
    best_by_model = defaultdict(dict)
    for row in rows:
        model = row["model"]
        train_source = row["train_source"]
        current = best_by_model[model].get(train_source)
        if current is None or row["test_accuracy"] > current["test_accuracy"]:
            best_by_model[model][train_source] = row

    print("model, distilled_acc, full_cifar10_acc, gap(full-distilled)")
    for model in sorted(best_by_model):
        distilled = best_by_model[model].get("distilled") or best_by_model[model].get("distilled_folder")
        full = best_by_model[model].get("full_cifar10")

        distilled_acc = None if distilled is None else distilled["test_accuracy"] * 100
        full_acc = None if full is None else full["test_accuracy"] * 100
        gap = None if distilled_acc is None or full_acc is None else full_acc - distilled_acc

        def fmt(value):
            return "NA" if value is None else f"{value:.2f}%"

        print(f"{model}, {fmt(distilled_acc)}, {fmt(full_acc)}, {fmt(gap)}")


if __name__ == "__main__":
    main()
