# Validation Report

Validation was run on Linux from `/root/Downloads/animal_lab_starter`.

| Criterion | Status | Evidence |
|---|---|---|
| AC1: pytest data checks pass | PASS | `tests/test_data.py::test_animal_data_is_well_formed PASSED`; `1 passed in 0.05s` |
| AC2: prep finds 5 animals and creates train/test folders | PASS | Prep table reported `cat`, `chicken`, `cow`, `dog`, and `horse`, each `20` total, `16` train, `4` test; split contained 100 images. |
| AC3: train finishes and saves artifacts | PASS | Training output: `Epoch 10/10 - loss: 0.6167 - train accuracy: 1.0000`; `Saved model to models/model.pt`; `models/model.pt` and `models/classes.json` exist. |
| AC4: evaluation output and metrics file | PASS | Output included `Overall accuracy`, per-animal accuracy, and `Confusion matrix (rows=real, columns=predicted)`; `metrics/metrics.json` was written. |
| AC5: test accuracy >= 0.70 | PASS | `Overall accuracy: 0.8500 (17/20)` and `QUALITY GATE PASSED: 0.8500 >= 0.7000`. |
| AC6: local prediction works | PASS | `Prediction: cat  (confidence 67.88%)` for `demo_images/cat_demo.jpg`. |
| AC7: score.py works with generated request | PASS | `Wrote sample-request.json (15245 bytes)` followed by a score result containing `animal`, `confidence`, and `all_scores`; prediction log reported `cat (68.90%)`. |
| AC8: no hard-coded animal count or names | PASS | Classes are read from `torchvision.datasets.ImageFolder`; saved `classes.json` contains discovered classes; model output size uses `len(dataset.classes)`. |
| AC9: README explains every step | PASS | `README.md` documents installation, tests, preparation, training, evaluation, local prediction, scoring, and Streamlit usage. |

## Commands run

```bash
python -m pip install -r requirements.txt
python -m pytest tests -v
python src/prep.py --raw_data data/animals --train_out split/train --test_out split/test
python src/train.py --train_data split/train --model_dir models
python src/evaluate.py --model_dir models --test_data split/test --metrics_out metrics
python src/predict.py --model_dir models --image demo_images/cat_demo.jpg
python src/make_request.py --image demo_images/cat_demo.jpg
AZUREML_MODEL_DIR=models PYTHONPATH=src python -c 'import score; score.init(); print(score.run(open("sample-request.json").read()))'
```

The supplied `azureml/` directory and `azure-pipelines.yml` were not changed.