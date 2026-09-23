"""Evaluate the model and enforce the accuracy quality gate."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model_utils import get_transform, load_model, log_metric


def evaluate(model_dir, test_data, metrics_out, min_accuracy=0.70):
    torch.set_num_threads(2)
    model, classes = load_model(model_dir)
    dataset = datasets.ImageFolder(test_data, transform=get_transform())
    matrix = [[0 for _ in classes] for _ in classes]
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=32):
            for real, predicted in zip(labels.tolist(), model(images).argmax(1).tolist()):
                matrix[real][predicted] += 1
    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(len(classes)))
    accuracy = correct / total if total else 0.0
    print(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    print("Per-animal accuracy:")
    per_animal = {}
    for index, name in enumerate(classes):
        count = sum(matrix[index])
        per_animal[name] = matrix[index][index] / count if count else 0.0
        print(f"  {name}: {per_animal[name]:.4f} ({matrix[index][index]}/{count})")
    print("Confusion matrix (rows=real, columns=predicted):")
    print("        " + " ".join(f"{name:>8}" for name in classes))
    for name, row in zip(classes, matrix):
        print(f"{name:>8} " + " ".join(f"{value:8}" for value in row))
    output = Path(metrics_out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps({"accuracy": accuracy, "samples": total, "classes": classes, "per_animal_accuracy": per_animal, "confusion_matrix": matrix}, indent=2) + "\n")
    log_metric("test_accuracy", accuracy)
    if accuracy < min_accuracy:
        print(f"QUALITY GATE FAILED: {accuracy:.4f} < {min_accuracy:.4f}")
        raise SystemExit(1)
    print(f"QUALITY GATE PASSED: {accuracy:.4f} >= {min_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--min_accuracy", type=float, default=0.70)
    evaluate(**vars(parser.parse_args()))