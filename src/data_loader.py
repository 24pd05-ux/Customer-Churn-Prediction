"""Data loading and initial cleaning for Customer Churn dataset."""

import os
import urllib.request
import pandas as pd
from typing import Tuple, Optional

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
FALLBACK_DATA_URL = "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_telco_churn.csv")


def download_dataset(target_path: Optional[str] = None) -> str:
    """Downloads Telco Customer Churn dataset if not present locally."""
    save_path = target_path or RAW_DATA_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not os.path.exists(save_path):
        print(f"Downloading dataset to {save_path}...")
        try:
            urllib.request.urlretrieve(DEFAULT_DATA_URL, save_path)
        except Exception as e:
            print(f"Primary URL failed ({e}). Trying fallback URL...")
            urllib.request.urlretrieve(FALLBACK_DATA_URL, save_path)
        print("Dataset download complete.")
    else:
        print(f"Dataset already exists at {save_path}.")
    return save_path


def load_raw_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Loads raw dataset from disk, downloading if missing."""
    file_path = data_path or RAW_DATA_PATH
    if not os.path.exists(file_path):
        download_dataset(file_path)
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data: converts TotalCharges to numeric, handles missing values,
    maps target variable to binary (0/1), and drops customerID.
    """
    df_clean = df.copy()
    
    # Drop customerID identifier if present
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])
        
    # TotalCharges contains whitespace strings for tenure=0 customers
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # Impute missing TotalCharges with MonthlyCharges * tenure (0 for tenure 0)
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["MonthlyCharges"] * df_clean["tenure"])
        
    # Map target column Churn to integer (1 / 0)
    if "Churn" in df_clean.columns:
        df_clean["Churn"] = df_clean["Churn"].apply(
            lambda x: 1 if str(x).strip().lower() in ["yes", "1", "true"] else 0
        ).astype(int)
        
    # Standardize 'No internet service' and 'No phone service' to 'No' for categorical clarity
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in service_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace({"No internet service": "No", "No phone service": "No"})
            
    return df_clean


def get_feature_target_split(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """Splits dataframe into feature matrix X and target series y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


if __name__ == "__main__":
    download_dataset()
    df = load_raw_data()
    print(f"Raw shape: {df.shape}")
    cleaned = clean_data(df)
    print(f"Cleaned shape: {cleaned.shape}")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_DATA_PATH}")
