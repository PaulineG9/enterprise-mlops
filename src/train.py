"""
train.py
---------
Model training, experiment tracking, and registry.

Trains a classifier that predicts `failure_within_7_days` from the
engineered asset-risk features produced by `src/etl_pipeline.py`.

Every run logs parameters, metrics, and the fitted model artifact to
MLflow (Experiment Tracking + Model Registry). If MLflow isn't
installed or reachable -- e.g. a laptop with no tracking server, or a
CI runner -- the script falls back to writing the same metrics to a
local JSON file and the model to disk with joblib, so the pipeline
never silently no-ops.

Usage:
    python src/train.py --features data/processed/asset_features/asset_features.csv
    python src/train.py --features data/processed/asset_features/asset_features.parquet --model-type gboost
"""

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from mlflow.models import infer_signature
    INFER_SIGNATURE_AVAILABLE = True
except ImportError:
    INFER_SIGNATURE_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

REGISTERED_MODEL_NAME = "asset-failure-risk-classifier"

NUMERIC_FEATURES = [
    "vibration_mm_s", "heat_celsius", "pressure_kpa", "runtime_hours",
    "cycle_count", "age_days", "rolling_vibration_avg", "rolling_heat_avg", "risk_flag",
]
CATEGORICAL_FEATURES = ["asset_type", "region"]
TARGET = "failure_within_7_days"


def load_features(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_pipeline(model_type: str) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",
    )

    if model_type == "gboost":
        model = GradientBoostingClassifier(random_state=42)
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def train(
    features_path: str,
    model_type: str,
    model_dir: str,
    test_size: float = 0.2,
    seed: int = 42,
    register_from_local_file: bool = False,
):
    """
    register_from_local_file: when False (default), the model is registered
    via a runs:/<run_id>/... URI -- the standard MLflow pattern, and what
    Databricks' own model registry expects. Set this to True only when
    tracking to an Azure ML backend: Azure ML's own run IDs are GUID-based
    and exceed the 32-character limit its CreateModelVersion API validates
    against, so registration there needs to happen from a local file path
    instead (which carries no run_id at all).
    """
    df = load_features(features_path)
    missing_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature file is missing expected columns: {missing_cols}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipeline = build_pipeline(model_type)

    params = {
        "model_type": model_type,
        "test_size": test_size,
        "seed": seed,
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
    }

    os.makedirs(model_dir, exist_ok=True)

    if MLFLOW_AVAILABLE:
        # No mlflow.set_experiment() call here on purpose: the caller (the
        # notebook) already sets it, using whatever the active tracking
        # backend actually requires -- a full workspace path like
        # "/Shared/enterprise-asset-risk" on Databricks, or a bare name on
        # Azure ML. Calling set_experiment() again here with a hardcoded
        # name would silently override that and could point the run at an
        # invalid/unexpected experiment depending on the backend.
        with mlflow.start_run() as run:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # Log the model artifact first, then register it as a separate
            # step (rather than passing registered_model_name directly into
            # log_model). This two-call pattern is what Azure ML's own docs
            # recommend, and it avoids a run_id-length validation bug that
            # the combined call can trigger when tracking to an Azure ML
            # backend from a Databricks-hosted run (CreateModelVersion:
            # "run_id too long. Maximum length is 32 characters").
            #
            # A model signature (input/output schema) is attached whenever
            # possible: it lets a deployed endpoint validate incoming
            # requests against the expected columns/types automatically,
            # and it's a hard requirement if this ever gets registered to a
            # Unity Catalog model registry instead of Azure ML's.
            artifact_path = "model"
            log_model_kwargs = {"artifact_path": artifact_path}
            if INFER_SIGNATURE_AVAILABLE:
                signature = infer_signature(X_train, pipeline.predict(X_train))
                log_model_kwargs["signature"] = signature
                log_model_kwargs["input_example"] = X_train.head(3)
            else:
                print("mlflow.models.infer_signature unavailable -- logging model without a signature.")

            mlflow.sklearn.log_model(pipeline, **log_model_kwargs)
            run_id = run.info.run_id

            if register_from_local_file:
                # Azure ML path: register from a LOCAL FILE PATH rather than
                # a runs:/<run_id>/... URI, sidestepping Azure ML's own
                # run_id-length validation bug (see the docstring above).
                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_model_path = os.path.join(tmp_dir, "model")
                    save_model_kwargs = {"path": local_model_path}
                    if "signature" in log_model_kwargs:
                        save_model_kwargs["signature"] = log_model_kwargs["signature"]
                        save_model_kwargs["input_example"] = log_model_kwargs["input_example"]
                    mlflow.sklearn.save_model(pipeline, **save_model_kwargs)
                    mlflow.register_model(f"file://{local_model_path}", REGISTERED_MODEL_NAME)
            else:
                # Standard MLflow pattern -- what Databricks' own registry expects.
                mlflow.register_model(f"runs:/{run_id}/{artifact_path}", REGISTERED_MODEL_NAME)

            active_experiment = mlflow.get_experiment(run.info.experiment_id)
            print(f"[mlflow] Run {run_id} logged to experiment '{active_experiment.name}'.")
            print(f"[mlflow] Model registered as '{REGISTERED_MODEL_NAME}'.")
    else:
        print("MLflow not available in this environment -- falling back to local artifact logging.")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)
        run_id = datetime.now(timezone.utc).strftime("local-%Y%m%dT%H%M%SZ")

        model_path = os.path.join(model_dir, f"{run_id}.joblib")
        joblib.dump(pipeline, model_path)

        metadata = {"run_id": run_id, "params": params, "metrics": metrics, "model_path": model_path}
        metadata_path = os.path.join(model_dir, f"{run_id}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[local backend] Model saved -> {model_path}")
        print(f"[local backend] Run metadata -> {metadata_path}")

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics, run_id


def main():
    parser = argparse.ArgumentParser(description="Train and register the asset failure-risk model.")
    parser.add_argument("--features", type=str,
                         default=os.path.join("data", "processed", "asset_features", "asset_features.csv"))
    parser.add_argument("--model-type", type=str, choices=["random_forest", "gboost"], default="random_forest")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    # parse_known_args (not parse_args) so that stray arguments injected by
    # a Jupyter/Databricks kernel (e.g. "-f /databricks/kernel-connections/
    # ....json", present if this file gets executed via %run instead of
    # imported) are silently ignored rather than causing a SystemExit --
    # real CLI usage from an actual shell is unaffected.
    args, _unrecognized = parser.parse_known_args()

    train(args.features, args.model_type, args.model_dir, args.test_size, args.seed)


if __name__ == "__main__":
    main()