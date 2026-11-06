"""Inference module for Customer Churn Prediction."""

import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, Union, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "churn_pipeline.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "model_metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(PROJECT_ROOT, "models", "feature_importances.json")

_MODEL = None
_METRICS = None
_FEATURE_IMPORTANCES = None


def load_model_assets():
    """Loads the serialized model and metadata."""
    global _MODEL, _METRICS, _FEATURE_IMPORTANCES
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run src/train.py first.")
        _MODEL = joblib.load(MODEL_PATH)
        
    if _METRICS is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            _METRICS = json.load(f)
            
    if _FEATURE_IMPORTANCES is None and os.path.exists(FEATURE_IMPORTANCE_PATH):
        with open(FEATURE_IMPORTANCE_PATH, "r") as f:
            _FEATURE_IMPORTANCES = json.load(f)
            
    return _MODEL, _METRICS, _FEATURE_IMPORTANCES


def get_risk_level(prob: float) -> str:
    """Categorizes churn probability into risk levels."""
    if prob >= 0.60:
        return "High Risk"
    elif prob >= 0.35:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_risk_color(risk_level: str) -> str:
    """Returns visual hex color code for risk level."""
    if risk_level == "High Risk":
        return "#EF4444"  # Red
    elif risk_level == "Medium Risk":
        return "#F59E0B"  # Amber
    else:
        return "#10B981"  # Emerald


def analyze_customer_risk_factors(customer_dict: Dict[str, Any]) -> List[str]:
    """Provides human-interpretable reasons why a customer has elevated or low churn risk."""
    reasons = []
    
    # Contract Analysis
    contract = customer_dict.get("Contract", "")
    if contract == "Month-to-month":
        reasons.append("⚠️ Month-to-month contract (highest correlation with churn)")
    elif contract in ["One year", "Two year"]:
        reasons.append("✅ Long-term contract (strong retention factor)")
        
    # Tenure Analysis
    tenure = float(customer_dict.get("tenure", 0))
    if tenure <= 6:
        reasons.append("⚠️ Early lifecycle customer (tenure ≤ 6 months)")
    elif tenure >= 24:
        reasons.append(f"✅ High tenure loyalty ({int(tenure)} months with company)")
        
    # Internet and Services
    internet = customer_dict.get("InternetService", "")
    tech_support = customer_dict.get("TechSupport", "")
    online_sec = customer_dict.get("OnlineSecurity", "")
    
    if internet == "Fiber optic":
        reasons.append("⚠️ Fiber optic plan (historically higher churn rate if unbundled)")
        
    if tech_support == "No" and internet != "No":
        reasons.append("⚠️ No Tech Support add-on service subscribed")
    elif tech_support == "Yes":
        reasons.append("✅ Active Tech Support protection")
        
    if online_sec == "No" and internet != "No":
        reasons.append("⚠️ No Online Security protection")
        
    # Payment Method
    payment = customer_dict.get("PaymentMethod", "")
    if "Electronic check" in payment:
        reasons.append("⚠️ Electronic check payment (associated with payment friction)")
    elif "automatic" in payment.lower():
        reasons.append("✅ Automated billing configured")
        
    # Charges
    monthly = float(customer_dict.get("MonthlyCharges", 0))
    if monthly > 80:
        reasons.append(f"⚠️ High monthly bill (${monthly:.2f}/mo)")
        
    return reasons


def predict_churn(customer_data: Union[Dict[str, Any], pd.DataFrame], threshold: float = None) -> Dict[str, Any]:
    """Generates churn prediction and probability for input customer data."""
    model, metrics, _ = load_model_assets()
    
    # Use optimal threshold from training if not specified
    if threshold is None:
        opt_thresh = metrics.get("optimal_threshold", 0.5) if metrics else 0.5
        if isinstance(opt_thresh, dict):
            threshold = float(opt_thresh.get("threshold", 0.5))
        else:
            threshold = float(opt_thresh)
    
    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    else:
        df = customer_data.copy()
        
    # Predict probabilities
    prob = float(model.predict_proba(df)[0, 1])
    is_churn = int(prob >= threshold)
    risk_level = get_risk_level(prob)
    
    # Extract risk factors if single customer
    risk_factors = []
    if isinstance(customer_data, dict):
        risk_factors = analyze_customer_risk_factors(customer_data)
        
    return {
        "churn_prediction": is_churn,
        "churn_probability": prob,
        "churn_probability_pct": f"{prob * 100:.1f}%",
        "risk_level": risk_level,
        "risk_color": get_risk_color(risk_level),
        "threshold_used": threshold,
        "key_factors": risk_factors
    }


if __name__ == "__main__":
    # Test sample profile
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 239.55
    }
    
    try:
        res = predict_churn(sample_customer)
        print("Sample Prediction Result:")
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Inference check note: {e}")
