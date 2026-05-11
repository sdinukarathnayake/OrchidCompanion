from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost model from a CSV file.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the CSV dataset.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        default="classification",
        help="Type of machine learning task.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows used for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of boosting rounds.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Boosting learning rate.")
    parser.add_argument("--subsample", type=float, default=0.9, help="Row sampling ratio.")
    parser.add_argument("--colsample-bytree", type=float, default=0.9, help="Feature sampling ratio.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Folder where the trained model and metrics will be saved.",
    )
    return parser.parse_args()


def build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [column for column in features.columns if column not in categorical_columns]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", build_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns were found in the dataset.")

    return ColumnTransformer(transformers=transformers)


def build_model(args: argparse.Namespace, num_classes: int | None = None):
    common_kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "random_state": args.random_state,
        "tree_method": "hist",
    }

    if args.task == "classification":
        if num_classes and num_classes > 2:
            return XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                eval_metric="mlogloss",
                **common_kwargs,
            )
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common_kwargs)

    return XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common_kwargs)


def save_outputs(output_dir: Path, payload: dict, metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_dir / "xgboost_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    if not args.output_dir.is_absolute():
        args.output_dir = script_dir / args.output_dir

    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    dataframe = pd.read_csv(args.data)
    if args.target not in dataframe.columns:
        raise ValueError(f"Target column '{args.target}' was not found in {args.data}")

    dataframe = dataframe.dropna(subset=[args.target]).copy()
    features = dataframe.drop(columns=[args.target])
    target = dataframe[args.target]

    if features.empty:
        raise ValueError("The dataset has no feature columns after removing the target column.")

    preprocessor = build_preprocessor(features)

    if args.task == "classification":
        label_encoder = LabelEncoder()
        encoded_target = label_encoder.fit_transform(target.astype(str))
        if len(label_encoder.classes_) < 2:
            raise ValueError("Classification needs at least two target classes.")
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                encoded_target,
                test_size=args.test_size,
                random_state=args.random_state,
                stratify=encoded_target,
            )
        except ValueError:
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                encoded_target,
                test_size=args.test_size,
                random_state=args.random_state,
            )
        model = build_model(args, num_classes=len(label_encoder.classes_))
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)
        class_names = [str(label) for label in label_encoder.classes_]
        report = classification_report(
            y_test,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        metrics = {
            "task": "classification",
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "report": report,
        }
        payload = {
            "pipeline": pipeline,
            "label_encoder": label_encoder,
            "task": "classification",
            "target_column": args.target,
            "feature_columns": features.columns.tolist(),
        }
    else:
        numeric_target = pd.to_numeric(target, errors="coerce")
        valid_rows = numeric_target.notna()
        features = features.loc[valid_rows]
        numeric_target = numeric_target.loc[valid_rows]
        if numeric_target.empty:
            raise ValueError("Regression target must contain numeric values.")

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            numeric_target,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        model = build_model(args)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        metrics = {
            "task": "regression",
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": rmse,
            "r2": float(r2_score(y_test, predictions)),
        }
        payload = {
            "pipeline": pipeline,
            "task": "regression",
            "target_column": args.target,
            "feature_columns": features.columns.tolist(),
        }

    save_outputs(args.output_dir, payload, metrics)
    print("Training complete.")
    print(f"Saved model to: {args.output_dir / 'xgboost_model.joblib'}")
    print(f"Saved metrics to: {args.output_dir / 'metrics.json'}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
