# Animal Predictor

This project trains a dynamically labelled animal image classifier. Animal
names come from folder names, so adding another folder needs no code change.

## Install

```bash
python -m pip install -r requirements.txt
```

## Check the data

```bash
python -m pytest tests -v
```

## Prepare a split

```bash
python src/prep.py --raw_data data/animals --train_out split/train --test_out split/test
```

The command validates every image, requires at least two animals and ten
images per animal, and creates a deterministic 80/20 split.

## Train

```bash
python src/train.py --train_data split/train --model_dir models
```

This uses ImageNet-pretrained MobileNetV2 with frozen feature layers. For a
fully trainable experiment, add `--from_scratch`.

## Evaluate

```bash
python src/evaluate.py --model_dir models --test_data split/test --metrics_out metrics
```

The command prints overall/per-animal accuracy and a real-by-predicted
confusion matrix, writes `metrics/metrics.json`, and fails below 70% accuracy.

## Local prediction

```bash
python src/predict.py --model_dir models --image demo_images/cat_demo.jpg
```

## Test the Azure scoring script locally

```bash
python src/make_request.py --image demo_images/cat_demo.jpg
AZUREML_MODEL_DIR=models PYTHONPATH=src python -c 'import score; score.init(); print(score.run(open("sample-request.json").read()))'
```

## Streamlit app

```bash
export ENDPOINT_URL="https://your-endpoint"
export ENDPOINT_KEY="your-key"
streamlit run app/app.py
```

The app resizes uploads to at most 512 pixels, sends JPEG base64 JSON with a
Bearer token, displays the prediction/confidence/all scores, and warns below
60% confidence.