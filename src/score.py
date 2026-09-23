"""Azure ML online endpoint scoring script."""
import base64
import io
import json
import os
from PIL import Image
from model_utils import load_model, predict_image

model = None
classes = None


def init():
    global model, classes
    model, classes = load_model(os.environ["AZUREML_MODEL_DIR"])


def run(raw_data):
    try:
        payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        result = predict_image(model, classes, Image.open(io.BytesIO(base64.b64decode(payload["image"]))))
        print(f"Prediction: {result['animal']} ({result['confidence'] * 100:.2f}%)")
        return result
    except Exception as error:
        print(f"Prediction error: {error}")
        return {"error": str(error)}