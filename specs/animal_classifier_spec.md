# Animal Classifier Specification

## Goal

Build an image classifier that predicts which animal appears in a photo and
returns both the predicted animal name and a confidence score.

## Data and labels

- Source images are under `data/animals/<animal_name>/` (or the equivalent
  input directory supplied to the preprocessing command).
- Each immediate subdirectory is one class; its directory name is the label.
- The implementation must discover class names from directories at runtime. It
  must not hard-code either the number or names of animals. Adding a sixth
  animal directory must require no code change.
- Image files are `.jpg`, `.jpeg`, or `.png`, matched case-insensitively.

## Preprocessing

- Split every class independently into 80% training images and 20% test
  images.
- Use fixed random seed `42` for the split. Ordering and selection must be
  deterministic for the same input directory.
- Preprocess images by resizing them to `224x224` and applying ImageNet
  normalisation with mean `(0.485, 0.456, 0.406)` and standard deviation
  `(0.229, 0.224, 0.225)`.
- The exact same inference transform must be used for training, evaluation,
  and prediction. No training-only augmentation is permitted.

## Model and training

- Use torchvision MobileNetV2 with ImageNet weights for transfer learning.
- Freeze every parameter in `model.features`.
- Replace `model.classifier[1]` with a new `Linear` layer whose output size is
  the discovered number of classes.
- Keep `model.features` in evaluation mode while training; only the classifier
  is trained.
- Save the trained model as a directory containing:
  - `model.pt`: the model `state_dict`.
  - `classes.json`: a JSON list of class/animal names in model output order.

## Evaluation and quality gate

- Evaluate on the generated test split and report accuracy and the number of
  test samples in JSON at the requested metrics output directory.
- The evaluation command accepts `min_accuracy` and exits with code `1` when
  test accuracy is less than that threshold. The default threshold is `0.70`.
- It exits successfully when accuracy is equal to or greater than the
  threshold.

## Command-line interface

The scripts must accept the arguments already used by the supplied Azure ML
configuration without changing `azureml/` or `azure-pipelines.yml`:

```text
python prep.py --raw_data PATH --train_out PATH --test_out PATH
python train.py --train_data PATH --model_dir PATH --epochs INTEGER
python evaluate.py --model_dir PATH --test_data PATH --metrics_out PATH --min_accuracy FLOAT
```

The serving script must load the model directory from `AZUREML_MODEL_DIR`
(with a direct model-directory fallback for local use), accept an image in the
Azure ML request JSON, and return an animal name and confidence score.

## Acceptance criteria

- **AC1:** Class names are discovered from immediate image-folder names, with
  no hard-coded animal list or class count.
- **AC2:** Preprocessing creates deterministic, per-class 80/20 train/test
  splits using seed 42, without mixing classes.
- **AC3:** Training and prediction use resize-to-224 and the specified
  ImageNet normalisation through the same shared transform.
- **AC4:** The model is torchvision MobileNetV2 with ImageNet weights,
  frozen `features`, a newly replaced classifier output layer, and frozen
  features kept in eval mode during training.
- **AC5:** Training writes `model.pt` as a state dict and `classes.json` as the
  ordered class-name list in the model directory.
- **AC6:** Prediction returns the selected discovered animal name and a
  confidence score derived from model probabilities.
- **AC7:** Evaluation writes JSON metrics including test accuracy and sample
  count to `metrics_out`.
- **AC8:** Evaluation exits with status 1 below `min_accuracy` and status 0 at
  or above it; the default quality threshold is 0.70.
- **AC9:** The supplied Azure files remain unchanged and their exact command
  arguments execute successfully against the implemented scripts.