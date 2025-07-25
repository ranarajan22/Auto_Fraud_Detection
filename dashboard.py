# dashboard.py

import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from preprocess import preprocess_data

st.set_page_config(page_title="AI-Powered Fraud Detection", layout="wide")
st.title("\U0001F697 AI-Powered Fraud Detection Dashboard")

# Load trained model and feature columns
model = joblib.load("fraud_model.pkl")
feature_columns = joblib.load("model_features.pkl")

# --- Section 1: Manual Entry Form ---
st.sidebar.header("\U0001F4DD Predict Single Claim")

with st.sidebar.form("manual_input"):
    age = st.number_input("Driver Age", min_value=18, max_value=100, value=35)
    claim_amount = st.number_input("Total Claim Amount", min_value=0.0, value=5000.0)
    state = st.selectbox("Policy State", ["NY", "CA", "TX", "FL", "OH", "IL"])
    months_as_customer = st.number_input("Months as Customer", min_value=0, value=12)
    num_vehicles = st.number_input("Number of Vehicles", min_value=1, max_value=5, value=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([{
            "Driver_Age": age,
            "Total_Claim": claim_amount,
            "Policy_State": state,
            "Months_As_Customer": months_as_customer,
            "Num_Vehicles": num_vehicles,
            "Gender": gender,
            "Married": married
        }])

        processed_input = preprocess_data(input_df)

        for col in feature_columns:
            if col not in processed_input.columns:
                processed_input[col] = 0
        processed_input = processed_input[feature_columns]

        prediction = model.predict(processed_input)[0]
        result = "\U0001F6A8 Fraudulent Claim" if prediction == 1 else "\u2705 Genuine Claim"
        st.sidebar.success(f"Prediction: {result}")

# --- Section 2: CSV Upload and Analysis ---
uploaded_file = st.file_uploader("\U0001F4E4 Upload Insurance Claim CSV", type="csv")

if uploaded_file:
    raw = pd.read_csv(uploaded_file)
    original = raw.copy()

    df = preprocess_data(raw)
    filtered_original = original.loc[df.index].copy()

    y_true = df["Fraud_Ind"] if "Fraud_Ind" in df.columns else None
    if "Fraud_Ind" in df.columns:
        df.drop("Fraud_Ind", axis=1, inplace=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    y_pred = model.predict(df)
    filtered_original["Fraud_Predicted"] = ["Fraud" if p == 1 else "Genuine" for p in y_pred]
    filtered_original["Fraud_Label"] = y_pred

    st.subheader("\U0001F4CA Key Fraud Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("\U0001F6A8 Fraud Cases", int((filtered_original["Fraud_Label"] == 1).sum()))
    col2.metric("\u2705 Genuine Cases", int((filtered_original["Fraud_Label"] == 0).sum()))
    col3.metric("\U0001F4E6 Total Predictions", len(filtered_original))
    if y_true is not None:
        acc = accuracy_score(y_true, y_pred)
        col4.metric("\U0001F3AF Accuracy", f"{acc * 100:.2f}%")

    st.subheader("\U0001F967 Prediction Distribution")
    pred_counts = filtered_original["Fraud_Predicted"].value_counts().reset_index()
    pred_counts.columns = ["Label", "Count"]
    pie_fig = px.pie(pred_counts, names="Label", values="Count", title="Fraud vs Genuine Claims")
    st.plotly_chart(pie_fig, use_container_width=True)

    if y_true is not None:
        st.subheader("\U0001F4C9 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig_cm)

        st.subheader("\U0001F4CB Classification Report")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(2))

    if "Total_Claim" in filtered_original.columns:
        st.subheader("\U0001F4B8 Average Claim Amount by Class")
        avg_claim = filtered_original.groupby("Fraud_Predicted")["Total_Claim"].mean().reset_index()
        avg_claim_fig = px.bar(avg_claim, x="Fraud_Predicted", y="Total_Claim", color="Fraud_Predicted",
                                title="Average Total Claim Amount by Class")
        st.plotly_chart(avg_claim_fig, use_container_width=True)

    if "Policy_State" in filtered_original.columns:
        st.subheader("\U0001F5FA️ Fraud Distribution by State")
        fraud_states = (
            filtered_original[filtered_original["Fraud_Label"] == 1]["Policy_State"]
            .value_counts()
            .reset_index()
        )
        fraud_states.columns = ["State", "Fraud Count"]
        fraud_state_fig = px.bar(fraud_states, x="State", y="Fraud Count", color="State",
                                  title="Top States with Fraudulent Claims")
        st.plotly_chart(fraud_state_fig, use_container_width=True)

    st.subheader("\U0001F4C8 Feature Importance")
    try:
        feat_imp = pd.Series(model.feature_importances_, index=df.columns).sort_values(ascending=False).head(10)
        fig_feat, ax = plt.subplots()
        sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax, palette="coolwarm")
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Important Features")
        st.pyplot(fig_feat)
    except Exception:
        st.warning("Feature importance is not available for this model type.")

    with st.expander("\U0001F4CB Show Prediction Table"):
        st.dataframe(filtered_original)
