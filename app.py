"""Streamlit Web Application: Customer Churn Prediction & Retention Intelligence."""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure src is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from predict import predict_churn, load_model_assets, analyze_customer_risk_factors, get_risk_level
from data_loader import clean_data

try:
    st.set_page_config(
        page_title="Customer Churn Prediction & Retention",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass

@st.cache_resource
def get_model_assets():
    return load_model_assets()

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #EF4444;
        font-weight: 700;
    }
    .risk-medium {
        color: #F59E0B;
        font-weight: 700;
    }
    .risk-low {
        color: #10B981;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


def create_gauge_chart(probability: float):
    """Generates a gauge chart for churn probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 38, 'color': "#1E293B"}},
        title={'text': "Churn Probability", 'font': {'size': 18, 'color': "#475569"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
            'bar': {'color': "#3B82F6", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 35], 'color': "#D1FAE5"},
                {'range': [35, 60], 'color': "#FEF3C7"},
                {'range': [60, 100], 'color': "#FEE2E2"}
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 3},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def main():
    st.markdown('<div class="main-header">🎯 Customer Churn Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production ML Pipeline with XGBoost & Explainable Retention Insights</div>', unsafe_allow_html=True)

    # Check if model exists
    try:
        model, metrics, feat_importances = get_model_assets()
    except Exception as e:
        st.error(f"⚠️ Model artifacts not found. Please run the training script first: `python src/train.py`\n\nError: {e}")
        st.stop()

    # Sidebar: Preset Profiles and Inputs
    st.sidebar.header("👤 Customer Profile Builder")
    
    preset = st.sidebar.selectbox(
        "Load Preset Profile for Quick Testing",
        ["Custom Input", "🚨 High Risk (New Fiber Month-to-Month)", "🛡️ Low Risk (Long-term Loyal)", "⚖️ Medium Risk (Borderline)"]
    )

    # Default values based on preset
    defaults = {
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

    if preset == "🚨 High Risk (New Fiber Month-to-Month)":
        defaults.update({
            "tenure": 2,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "TechSupport": "No",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.50,
            "TotalCharges": 179.00
        })
    elif preset == "🛡️ Low Risk (Long-term Loyal)":
        defaults.update({
            "tenure": 60,
            "Contract": "Two year",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "TechSupport": "Yes",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 65.00,
            "TotalCharges": 3900.00,
            "Partner": "Yes",
            "Dependents": "Yes"
        })
    elif preset == "⚖️ Medium Risk (Borderline)":
        defaults.update({
            "tenure": 14,
            "Contract": "One year",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "TechSupport": "Yes",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 70.00,
            "TotalCharges": 980.00
        })

    with st.sidebar.expander("1. Demographics", expanded=True):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if defaults["gender"] == "Female" else 1)
            senior = st.selectbox("Senior Citizen", [0, 1], index=defaults["SeniorCitizen"], format_func=lambda x: "Yes" if x == 1 else "No")
        with col_d2:
            partner = st.selectbox("Partner", ["No", "Yes"], index=0 if defaults["Partner"] == "No" else 1)
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=0 if defaults["Dependents"] == "No" else 1)

    with st.sidebar.expander("2. Account & Contract", expanded=True):
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=int(defaults["tenure"]))
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(defaults["Contract"])
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            index=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ].index(defaults["PaymentMethod"])
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if defaults["PaperlessBilling"] == "Yes" else 1)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=float(defaults["MonthlyCharges"]), step=0.5)
        
        # Calculate suggested total charges
        suggested_total = max(monthly_charges * max(tenure, 1), 18.0) if tenure > 0 else monthly_charges
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(suggested_total), step=10.0)

    with st.sidebar.expander("3. Subscribed Services", expanded=False):
        phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if defaults["PhoneService"] == "Yes" else 1)
        multiple = st.selectbox("Multiple Lines", ["No", "Yes"], index=0 if defaults["MultipleLines"] == "No" else 1)
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(defaults["InternetService"]))
        
        security = st.selectbox("Online Security", ["No", "Yes"], index=0 if defaults["OnlineSecurity"] == "No" else 1)
        backup = st.selectbox("Online Backup", ["No", "Yes"], index=0 if defaults["OnlineBackup"] == "No" else 1)
        protection = st.selectbox("Device Protection", ["No", "Yes"], index=0 if defaults["DeviceProtection"] == "No" else 1)
        tech_support = st.selectbox("Tech Support", ["No", "Yes"], index=0 if defaults["TechSupport"] == "No" else 1)
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], index=0 if defaults["StreamingTV"] == "No" else 1)
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], index=0 if defaults["StreamingMovies"] == "No" else 1)

    # Compile input data
    input_payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Tabs for navigation
    tab1, tab2, tab3 = st.tabs(["🔮 Single Customer Risk Assessment", "📈 Model Benchmarks & Explainability", "📁 Batch Prediction"])

    with tab1:
        # Perform inference
        prediction_res = predict_churn(input_payload)
        prob = prediction_res["churn_probability"]
        risk_level = prediction_res["risk_level"]
        risk_color = prediction_res["risk_color"]
        factors = prediction_res["key_factors"]

        col1, col2 = st.columns([1.2, 1.8])

        with col1:
            st.plotly_chart(create_gauge_chart(prob), use_container_width=True)
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: -10px; margin-bottom: 20px;">
                <span style="font-size: 1.3rem; font-weight: 700; color: {risk_color}; padding: 6px 16px; border-radius: 20px; background-color: {risk_color}1A; border: 1px solid {risk_color};">
                    Status: {risk_level} ({prob * 100:.1f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("💡 Strategic Retention Guidance")
            if risk_level == "High Risk":
                st.error("🚨 **High Risk of Churn within next 30-60 days.**")
                st.markdown("""
                **Recommended Action Plan:**
                - 🎁 **Incentivize Annual Contract**: Offer a 15% discount on an annual commitment.
                - 🛠️ **Complimentary Tech Support**: Provide 3 months free Tech Support and security suite.
                - 💳 **Encourage Auto-Pay**: Offer a $10 one-time billing credit for switching to automated payment.
                """)
            elif risk_level == "Medium Risk":
                st.warning("⚠️ **Moderate Churn Risk — Active Monitoring Recommended.**")
                st.markdown("""
                **Recommended Action Plan:**
                - 📞 **Proactive Check-in**: Send customer satisfaction survey regarding connection quality.
                - 📦 **Bundle Offer**: Suggest bundling cloud backup at discounted rate.
                """)
            else:
                st.success("✅ **Low Churn Risk — Highly Retained Customer Profile.**")
                st.markdown("""
                **Recommended Action Plan:**
                - 🌟 **Loyalty Appreciation**: Eligible for loyalty perks and early access promotions.
                - 🚀 **Upsell Opportunity**: Good candidate for premium bandwidth speed upgrades.
                """)

        st.divider()
        st.subheader("🔍 Contributing Customer Factors")
        if factors:
            factor_cols = st.columns(2)
            for idx, factor in enumerate(factors):
                with factor_cols[idx % 2]:
                    if "⚠️" in factor:
                        st.warning(factor)
                    else:
                        st.success(factor)
        else:
            st.info("No prominent outlier risk drivers identified for this profile.")

    with tab2:
        st.subheader("📊 Model Comparison & Cross-Validation Results")
        
        if metrics and "cv_results" in metrics:
            cv_df = pd.DataFrame(metrics["cv_results"])
            st.dataframe(
                cv_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#D1FAE5", axis=0)
                .format({
                    "Accuracy": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1-Score": "{:.3f}",
                    "ROC-AUC": "{:.3f}"
                }),
                use_container_width=True
            )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if metrics:
                opt = metrics.get("optimal_threshold", {})
                st.metric("Test ROC-AUC", f"{metrics.get('test_roc_auc', 0.84):.4f}")
                st.metric("Test F1-Score (Optimal)", f"{opt.get('f1', 0.65):.4f}")
                st.metric("Test Recall (Optimal)", f"{opt.get('recall', 0.78):.4f}")

        with col_m2:
            st.subheader("Global Feature Importance (XGBoost)")
            if feat_importances:
                top_feats = list(feat_importances.items())[:12]
                feat_df = pd.DataFrame(top_feats, columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
                fig_imp = px.bar(
                    feat_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 12 Predictive Features for Churn",
                    color="Importance",
                    color_continuous_scale="Blues"
                )
                fig_imp.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)

    with tab3:
        st.subheader("📁 Batch Customer Churn Prediction")
        st.write("Upload a CSV file containing customer rows to generate bulk predictions and risk tiers.")
        
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(batch_df)} customer records.")
            if st.button("Run Batch Inference"):
                with st.spinner("Processing customer profiles..."):
                    model, metrics, _ = get_model_assets()
                    raw_thresh = metrics.get("optimal_threshold", 0.5) if metrics else 0.5
                    optimal_threshold = float(raw_thresh.get("threshold", 0.5)) if isinstance(raw_thresh, dict) else float(raw_thresh)
                    # Preserve IDs
                    id_col = None
                    if "customerID" in batch_df.columns:
                        id_col = batch_df["customerID"].copy()
                    batch_clean = clean_data(batch_df)
                    probs = model.predict_proba(batch_clean)[:, 1]
                    result_df = pd.DataFrame({
                        "Churn_Probability": np.round(probs, 4),
                        "Churn_Prediction": (probs >= optimal_threshold).astype(int),
                        "Risk_Level": [get_risk_level(p) for p in probs]
                    })
                    if id_col is not None:
                        result_df.insert(0, "customerID", id_col.values)
                    st.session_state["batch_results"] = result_df

            if "batch_results" in st.session_state:
                result_df = st.session_state["batch_results"]
                st.success(f"Processed {len(result_df)} customers")
                st.dataframe(result_df, use_container_width=True)
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions CSV", csv, "churn_predictions.csv", "text/csv")


if __name__ == "__main__":
    main()
