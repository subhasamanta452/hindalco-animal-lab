"""Small shared helpers used by training, evaluation, and prediction."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_transform():
    """Return the one transform used for both training and prediction."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_model(num_classes, pretrained=True, freeze=True):
    """Build MobileNetV2 and replace its output layer for our classes."""
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    if freeze:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        model.features.eval()
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
    return model


def save_model(model, classes, model_dir):
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "classes.json").write_text(
        json.dumps(list(classes), indent=2) + "\n", encoding="utf-8"
    )


def _find_artifacts(model_dir):
    root = Path(model_dir)
    model_files = list(root.rglob("model.pt"))
    class_files = list(root.rglob("classes.json"))
    if not model_files or not class_files:
        raise FileNotFoundError(f"Could not find model.pt and classes.json below {root}")
    model_file = model_files[0]
    class_file = next((path for path in class_files if path.parent == model_file.parent), class_files[0])
    return model_file, class_file


def load_model(model_dir):
    torch.set_num_threads(2)
    model_file, classes_file = _find_artifacts(model_dir)
    classes = json.loads(classes_file.read_text(encoding="utf-8"))
    model = build_model(len(classes), pretrained=False, freeze=False)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()
    return model, classes


def predict_image(model, classes, pil_image):
    image = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
    with torch.no_grad():
        scores = torch.softmax(model(get_transform()(image).unsqueeze(0)), dim=1)[0]
    result = {name: float(scores[index]) for index, name in enumerate(classes)}
    index = int(scores.argmax())
    return {"animal": classes[index], "confidence": float(scores[index]), "all_scores": result}


def log_metric(name, value, step=None):
    """Use Azure ML's MLflow only when an Azure ML URI is configured."""
    if os.environ.get("MLFLOW_TRACKING_URI", "").startswith("azureml"):
        import mlflow
        mlflow.log_metric(name, float(value), step=step)
