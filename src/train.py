"""Train a MobileNetV2 classifier using torchvision's ImageFolder."""
import argparse
import random

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from model_utils import build_model, get_transform, log_metric, save_model


def train(train_data, model_dir, epochs=10, lr=0.001, batch_size=16, from_scratch=False):
    # A small thread count keeps CPU-only Azure jobs responsive.
    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    dataset = datasets.ImageFolder(train_data, transform=get_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = build_model(len(dataset.classes), pretrained=not from_scratch, freeze=not from_scratch)
    if not from_scratch:
        # Frozen features never change, so compute them once for fast CPU training.
        feature_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        cached_features, cached_labels = [], []
        with torch.no_grad():
            for images, labels in feature_loader:
                features = functional.adaptive_avg_pool2d(model.features(images), (1, 1))
                cached_features.append(torch.flatten(features, 1))
                cached_labels.append(labels)
        train_inputs = torch.cat(cached_features)
        train_labels = torch.cat(cached_labels)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        if not from_scratch:
            model.features.eval()
        correct = total = 0
        running_loss = 0.0
        batches = zip(train_inputs.split(batch_size), train_labels.split(batch_size)) if not from_scratch else loader
        for images, labels in batches:
            optimizer.zero_grad()
            if from_scratch:
                outputs = model(images)
            else:
                outputs = model.classifier(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        loss_value = running_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{epochs} - loss: {loss_value:.4f} - train accuracy: {accuracy:.4f}")
        log_metric("train_loss", loss_value, epoch + 1)
        log_metric("train_accuracy", accuracy, epoch + 1)
    save_model(model, dataset.classes, model_dir)
    print(f"Saved model to {model_dir}/model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--from_scratch", action="store_true")
    train(**vars(parser.parse_args()))