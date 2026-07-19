"""
Modeling pipeline for 30-day hospital readmission prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

from clean import load_and_clean, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET


def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_data(path):
    df = load_and_clean(path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    # Categorical admission/discharge/source ids are actually codes -> treat as strings
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        X[col] = X[col].astype(str)
    y = df[TARGET]
    return X, y, df


def train_test(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def evaluate(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, preds, digits=3))
    return {"name": name, "auc": auc, "proba": proba, "preds": preds}


if __name__ == "__main__":
    X, y, df = get_data("/home/claude/readmission_project/data/diabetic_data.csv")
    X_train, X_test, y_train, y_test = train_test(X, y)
    preprocessor = build_preprocessor()

    results = []

    # 1. Logistic Regression baseline (with SMOTE)
    logreg_pipe = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    logreg_pipe.fit(X_train, y_train)
    results.append(evaluate("Logistic Regression (+SMOTE)", logreg_pipe, X_test, y_test))

    # 2. Random Forest (with SMOTE)
    rf_pipe = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=10,
                                        class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    rf_pipe.fit(X_train, y_train)
    results.append(evaluate("Random Forest (+SMOTE)", rf_pipe, X_test, y_test))

    # 3. XGBoost (class-weighted, no SMOTE needed - handles imbalance via scale_pos_weight)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric="auc",
            random_state=42, n_jobs=-1
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    results.append(evaluate("XGBoost (class-weighted)", xgb_pipe, X_test, y_test))

    print("\n\nSUMMARY")
    for r in results:
        print(f"{r['name']:35s} AUC = {r['auc']:.4f}")
