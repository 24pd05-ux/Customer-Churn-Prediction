# 🎯 Project 1: End-to-End Customer Churn Prediction & Retention Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](https://github.com/Dheeraj23qw/customer-churn-prediction)

> A production-grade Machine Learning classification system predicting customer churn in the telecommunications sector. Includes full data cleaning, feature engineering, cross-validation model benchmarking, hyperparameter tuning, explainability insights, and an interactive Streamlit intelligence dashboard.

📑 **[Technical Architecture & System Specification](./docs/ARCHITECTURE.md)**: Deep dive into system design, mathematical derivations, loss functions, leakage controls, and engineering tradeoffs.

---

## 📌 1. Business Problem & Objective

Customer acquisition costs (CAC) in telecommunications are **5–7x higher** than retention costs. When customers churn unnoticed, recurring subscription revenue is permanently lost. 

**Objective:** 
Build an automated classification pipeline that predicts the likelihood of customer attrition, identifies key high-risk behaviors, and delivers actionable retention recommendations via an interactive dashboard.

---

## 🧠 2. ML Concepts & Techniques Demonstrated

* **Exploratory Data Analysis (EDA)**: Target distribution analysis, demographic cohort studies, churn hazard curves over tenure.
* **Data Cleaning & Handling**: Imputing whitespace strings in `TotalCharges`, removing non-predictive identifiers, standardizing multi-category flags.
* **Feature Engineering**:
  * `Num_Services`: Total count of active digital add-on services.
  * `Avg_Monthly_Paid` & `Charge_Discrepancy`: Discrepancy between contracted rates and actual payments.
  * `Is_Auto_Pay`: Automated payment method indicator.
  * `Is_Long_Term_Customer`: Tenure loyalty thresholding.
* **Leakage-Free Pipelines**: Scikit-learn `Pipeline` and `ColumnTransformer` encapsulation (StandardScaler for numericals + OneHotEncoder for categoricals).
* **Model Benchmarking**: 5-Fold Stratified Cross-Validation across **Logistic Regression**, **Random Forest**, and **XGBoost**.
* **Imbalanced Data Strategy**: Positive class weighting (`scale_pos_weight` & `class_weight="balanced"`).
* **Hyperparameter Tuning**: Exhaustive `GridSearchCV` on XGBoost parameters (depth, learning rate, subsampling, regularization).
* **Threshold Optimization**: Maximizing F1-score and Recall on test set to capture high-risk churners without overwhelming customer success teams.
* **Model Explainability**: Global feature importances and human-readable individual risk factor explanations.
* **Web Deployment**: Production Streamlit application with interactive scenario presets, risk gauge charts, and batch prediction mode.

---

## 🏗️ 3. Architecture & Data Flow

```mermaid
flowchart LR
    A[Raw Telco Data] --> B[Data Cleaning & Imputation]
    B --> C[Feature Engineering]
    C --> D[ColumnTransformer Pipeline]
    D --> E[Stratified 5-Fold CV Benchmark]
    E --> F[XGBoost Hyperparameter Tuning]
    F --> G[Serialized Pipeline: churn_pipeline.joblib]
    G --> H[Streamlit Intelligence App]
```

---

## 📊 4. Model Evaluation & Comparison

5-Fold Stratified Cross-Validation benchmark across candidate models on the training set (80% split):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 74.9% | 51.8% | 79.9% | 0.629 | 0.845 |
| **Random Forest** | 75.8% | 53.0% | 77.9% | 0.630 | 0.846 |
| **XGBoost (Tuned)** 🏆 | **75.2%** | **52.2%** | **79.1%** | **0.629** | **0.847** |

### Test Set Performance (Held-out 20%)
* **Test ROC-AUC**: `0.845`
* **Test Recall (Optimal Threshold 0.60)**: `67.9%` with `57.5%` Precision (at baseline 0.50 threshold: Recall `79.1%`, Precision `51.6%`)

---

## 🔍 5. Key Risk Drivers (Feature Importance)

1. **Contract Type (`Month-to-month`)**: The strongest single predictor of churn. Customers on month-to-month plans are over **14x more likely** to leave than those on two-year agreements.
2. **Tenure**: Early tenure (1–6 months) is the most critical hazard window.
3. **Internet Service (`Fiber optic`)**: Higher churn rate when fiber optic plans are unbundled from security/support services.
4. **Payment Method (`Electronic Check`)**: Correlates with payment friction and transient customer behavior.
5. **Add-on Services (`TechSupport`, `OnlineSecurity`)**: Customers with 3+ active security services exhibit churn rates below 10%.

---

## 🖥️ 6. Streamlit Dashboard Features

1. **🔮 Single Customer Risk Assessment**:
   - Interactive sliders, dropdowns, and switch inputs.
   - Quick 1-click Presets: *High-Risk Churner*, *Loyal Retained Customer*, *Moderate Risk Borderline*.
   - Dynamic Gauge Chart displaying real-time churn probability.
   - Automated retention guidance (e.g. recommend 15% discount on annual upgrade, complementary tech support bundle).
2. **📈 Model Benchmarks & Explainability**:
   - Live cross-validation metrics comparison table.
   - Interactive Plotly horizontal bar chart of top predictive features.
3. **📁 Batch Prediction**:
   - CSV uploader for batch scoring customer cohorts with downloadable enriched results.

---

## 🚀 7. How to Run Locally

### 1. Clone the repository and navigate to the project
```bash
cd 01-customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the training & evaluation pipeline
```bash
python src/train.py
```
This will:
- Download the Telco dataset automatically if not present.
- Execute data cleaning and feature engineering.
- Run 5-fold cross-validation and hyperparameter tuning.
- Save the trained pipeline to `models/churn_pipeline.joblib` and metrics to `models/model_metrics.json`.

### 4. Launch the Streamlit dashboard
```bash
streamlit run app.py
```

---

## 📂 8. Project Structure

```text
01-customer-churn-prediction/
├── data/
│   ├── raw/                      # Downloaded Telco dataset
│   └── processed/                # Cleaned dataset
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   └── 02_model_experimentation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Automated download & cleaning
│   ├── preprocessor.py           # Feature engineering & ColumnTransformer
│   ├── train.py                  # CV benchmarking, tuning & serialization
│   └── predict.py                # Inference & risk explanation logic
├── models/
│   ├── churn_pipeline.joblib     # Production model artifact
│   ├── model_metrics.json        # Evaluation metadata
│   └── feature_importances.json  # Feature ranking
├── app.py                        # Streamlit web application
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```
