"""Preprocessing and feature engineering pipeline for Customer Churn Prediction."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for domain-specific churn feature engineering."""

    def __init__(self):
        self.digital_services = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Total active add-on digital services count
        existing_services = [col for col in self.digital_services if col in df.columns]
        if existing_services:
            df["Num_Services"] = (df[existing_services] == "Yes").sum(axis=1)
        else:
            df["Num_Services"] = 0

        # 2. Charge per tenure month ratio
        if "TotalCharges" in df.columns and "tenure" in df.columns and "MonthlyCharges" in df.columns:
            # Expected vs actual total charges ratio (detect rate changes / discounting)
            expected_total = df["MonthlyCharges"] * df["tenure"]
            df["Charge_Discrepancy"] = np.where(
                df["tenure"] > 0,
                df["TotalCharges"] - expected_total,
                0.0
            )
            df["Avg_Monthly_Paid"] = np.where(
                df["tenure"] > 0,
                df["TotalCharges"] / df["tenure"],
                df["MonthlyCharges"]
            )

        # 3. Automatic payment flag
        if "PaymentMethod" in df.columns:
            df["Is_Auto_Pay"] = df["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)

        # 4. Long tenure loyalty flag (> 24 months)
        if "tenure" in df.columns:
            df["Is_Long_Term_Customer"] = (df["tenure"] >= 24).astype(int)

        return df


def get_preprocessor_pipeline() -> Pipeline:
    """Builds a scikit-learn ColumnTransformer pipeline that applies feature engineering,
    standard scaling to numeric features, and one-hot encoding to categorical features.
    """
    numeric_features = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "Num_Services", "Charge_Discrepancy", "Avg_Monthly_Paid",
        "Is_Auto_Pay", "Is_Long_Term_Customer"
    ]

    categorical_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline


def get_feature_names_out(pipeline: Pipeline, input_df: pd.DataFrame) -> list:
    """Extracts transformed feature names after preprocessing."""
    fe_df = pipeline.named_steps["feature_engineering"].transform(input_df)
    preprocessor = pipeline.named_steps["preprocessor"]
    
    # Fit preprocessor on fe_df if not fitted yet
    if not hasattr(preprocessor, "transformers_"):
        preprocessor.fit(fe_df)
        
    feature_names = []
    
    # Numeric feature names
    numeric_features = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "Num_Services", "Charge_Discrepancy", "Avg_Monthly_Paid",
        "Is_Auto_Pay", "Is_Long_Term_Customer"
    ]
    feature_names.extend(numeric_features)
    
    # Categorical one-hot encoded names
    cat_encoder = preprocessor.named_transformers_["cat"]
    categorical_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]
    cat_names = cat_encoder.get_feature_names_out(categorical_features)
    feature_names.extend(cat_names.tolist())
    
    return feature_names
