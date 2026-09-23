"""Predict one local image."""
import argparse
from PIL import Image
from model_utils import load_model, predict_image

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True)
parser.add_argument("--image", required=True)
args = parser.parse_args()
model, classes = load_model(args.model_dir)
result = predict_image(model, classes, Image.open(args.image))
print(f"Prediction: {result['animal']}  (confidence {result['confidence'] * 100:.2f}%)")