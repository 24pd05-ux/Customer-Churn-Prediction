"""Model training, hyperparameter tuning, evaluation, and serialization pipeline."""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_raw_data, clean_data, get_feature_target_split
from preprocessor import get_preprocessor_pipeline, get_feature_names_out

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "churn_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importances.json")


def build_candidate_pipelines() -> Dict[str, Pipeline]:
    """Instantiates candidate models wrapped with preprocessing pipeline."""
    preprocessor = get_preprocessor_pipeline()
    
    candidates = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                solver="liblinear"
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=7,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=2.76,
                eval_metric="logloss",
                random_state=42,
                n_jobs=1
            ))
        ])
    }
    return candidates


def evaluate_cross_validation(candidates: Dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Evaluates candidates using 5-fold Stratified Cross-Validation."""
    print("=" * 65, flush=True)
    print("Running 5-Fold Stratified Cross-Validation...", flush=True)
    print("=" * 65, flush=True)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    results = []
    for name, pipeline in candidates.items():
        print(f"Evaluating {name}...", flush=True)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        res = {
            "Model": name,
            "Accuracy": np.mean(scores["test_accuracy"]),
            "Precision": np.mean(scores["test_precision"]),
            "Recall": np.mean(scores["test_recall"]),
            "F1-Score": np.mean(scores["test_f1"]),
            "ROC-AUC": np.mean(scores["test_roc_auc"])
        }
        results.append(res)
        print(f"  -> ROC-AUC: {res['ROC-AUC']:.4f} | F1: {res['F1-Score']:.4f} | Recall: {res['Recall']:.4f} | Precision: {res['Precision']:.4f}", flush=True)
        
    df_results = pd.DataFrame(results)
    return df_results


def tune_best_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Tunes XGBoost hyperparameters using GridSearchCV."""
    print("\n" + "=" * 65, flush=True)
    print("Hyperparameter Tuning for XGBoost...", flush=True)
    print("=" * 65, flush=True)
    
    preprocessor = get_preprocessor_pipeline()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        ))
    ])
    
    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [3, 4],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__scale_pos_weight": [2.5, 2.76],
        "classifier__subsample": [0.8, 1.0]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}", flush=True)
    print(f"Best Parameters: {grid_search.best_params_}", flush=True)
    
    return grid_search.best_estimator_


def extract_feature_importances(pipeline: Pipeline, X_sample: pd.DataFrame) -> Dict[str, float]:
    """Extracts feature names and corresponding feature importances."""
    preprocessor_pipeline = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]
    
    feature_names = get_feature_names_out(preprocessor_pipeline, X_sample)
    
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.zeros(len(feature_names))
        
    feat_dict = dict(zip(feature_names, [float(x) for x in importances]))
    # Sort descending
    sorted_feat = dict(sorted(feat_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_feat


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Finds decision threshold that optimizes the F1 score for churn prediction."""
    best_f1 = 0.0
    best_threshold = 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thresh)
    return best_threshold


def train_and_save():
    """End-to-end training pipeline orchestration."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load & Clean Data
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    X, y = get_feature_target_split(cleaned_df, target_col="Churn")
    
    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Class distribution: Train Churn = {y_train.mean():.2%}, Test Churn = {y_test.mean():.2%}")
    
    # 3. Candidate Model Evaluation
    candidates = build_candidate_pipelines()
    cv_comparison = evaluate_cross_validation(candidates, X_train, y_train)
    print("\n--- Model Comparison Summary ---")
    print(cv_comparison.to_string(index=False))
    
    # 4. Hyperparameter Tuning on Best Architecture (XGBoost)
    best_pipeline = tune_best_model(X_train, y_train)
    
    # 5. Held-out Test Set Evaluation
    y_pred_default = best_pipeline.predict(X_test)
    
    from sklearn.model_selection import cross_val_predict

    # Find optimal threshold using out-of-fold predictions (no test data leakage)
    print("Finding optimal threshold via cross-validation on training set...", flush=True)
    oof_probs = cross_val_predict(best_pipeline, X_train, y_train, cv=5, method="predict_proba")[:, 1]
    optimal_thresh = find_optimal_threshold(y_train.values, oof_probs)
    print(f"Optimal threshold (from CV): {optimal_thresh:.2f}", flush=True)

    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred_optimal = (y_prob >= optimal_thresh).astype(int)
    
    test_metrics = {
        "cv_results": cv_comparison.to_dict(orient="records"),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "default_threshold": {
            "threshold": 0.5,
            "accuracy": float(accuracy_score(y_test, y_pred_default)),
            "precision": float(precision_score(y_test, y_pred_default)),
            "recall": float(recall_score(y_test, y_pred_default)),
            "f1": float(f1_score(y_test, y_pred_default)),
            "confusion_matrix": confusion_matrix(y_test, y_pred_default).tolist()
        },
        "optimal_threshold": {
            "threshold": optimal_thresh,
            "accuracy": float(accuracy_score(y_test, y_pred_optimal)),
            "precision": float(precision_score(y_test, y_pred_optimal)),
            "recall": float(recall_score(y_test, y_pred_optimal)),
            "f1": float(f1_score(y_test, y_pred_optimal)),
            "confusion_matrix": confusion_matrix(y_test, y_pred_optimal).tolist()
        }
    }
    
    print("\n" + "=" * 65)
    print("FINAL TEST EVALUATION")
    print("=" * 65)
    print(f"ROC-AUC: {test_metrics['test_roc_auc']:.4f}")
    print(f"Default Threshold (0.50): F1={test_metrics['default_threshold']['f1']:.4f} | Recall={test_metrics['default_threshold']['recall']:.4f} | Precision={test_metrics['default_threshold']['precision']:.4f}")
    print(f"Optimal Threshold ({optimal_thresh:.2f}): F1={test_metrics['optimal_threshold']['f1']:.4f} | Recall={test_metrics['optimal_threshold']['recall']:.4f} | Precision={test_metrics['optimal_threshold']['precision']:.4f}")
    print("\nClassification Report (Optimal Threshold):")
    print(classification_report(y_test, y_pred_optimal, target_names=["Retained (0)", "Churned (1)"]))
    
    # 6. Extract Feature Importances
    feat_importances = extract_feature_importances(best_pipeline, X_train)
    print("\nTop 10 Feature Importances:")
    for feat, imp in list(feat_importances.items())[:10]:
        print(f"  {feat}: {imp:.4f}")
        
    # 7. Save Artifacts
    joblib.dump(best_pipeline, MODEL_ARTIFACT_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_ARTIFACT_PATH}")
    
    with open(METRICS_PATH, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved metrics to: {METRICS_PATH}")
    
    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(feat_importances, f, indent=2)
    print(f"Saved feature importances to: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    train_and_save()
