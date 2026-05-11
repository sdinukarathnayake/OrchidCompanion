# XGBoost Training

This folder contains a simple XGBoost training script for CSV-based datasets.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Train a classification model

```bash
python train_xgboost.py --data your_dataset.csv --target label --task classification
```

## Train a regression model

```bash
python train_xgboost.py --data your_dataset.csv --target price --task regression
```

## Optional arguments

- `--test-size 0.2` controls the validation split.
- `--n-estimators 300` sets the number of boosting rounds.
- `--max-depth 6` sets the maximum tree depth.
- `--learning-rate 0.05` sets the learning rate.
- `--output-dir artifacts` chooses where model files are saved.

## Output files

After training, the script saves:

- `xgboost_model.joblib` - trained preprocessing pipeline and model
- `metrics.json` - evaluation metrics for the test split
