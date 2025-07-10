import streamlit as st
import pandas as pd
import joblib
import os

@st.cache_resource
def load_model_components():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, scaler, label_encoders

# ✅ Define feature order
FEATURE_NAMES = [
    "Age", "Gender", "Smoking", "Hx Smoking", "Hx Radiotherapy",
    "Thyroid Function", "Physical Examination", "Adenopathy",
    "Pathology", "Focality", "Risk", "T", "N", "M", "Stage", "Response"
]

# ✅ Function to build input form
def get_user_input(label_encoders):
    with st.form("patient_form"):
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", ["F", "M"])
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        hx_smoking = st.selectbox("Hx Smoking", ["No", "Yes"])
        hx_radiotherapy = st.selectbox("Hx Radiotherapy", sorted(label_encoders["Hx Radiotherapy"].classes_))
        thyroid_function = st.selectbox("Thyroid Function", sorted(label_encoders["Thyroid Function"].classes_))
        physical_exam = st.selectbox("Physical Examination", sorted(label_encoders["Physical Examination"].classes_))
        adenopathy = st.selectbox("Adenopathy", sorted(label_encoders["Adenopathy"].classes_))
        pathology = st.selectbox("Pathology", sorted(label_encoders["Pathology"].classes_))
        focality = st.selectbox("Focality", sorted(label_encoders["Focality"].classes_))
        risk = st.selectbox("Risk", sorted(label_encoders["Risk"].classes_))
        T = st.selectbox("T Stage", sorted(label_encoders["T"].classes_))
        N = st.selectbox("N Stage", sorted(label_encoders["N"].classes_))
        M = st.selectbox("M Stage", sorted(label_encoders["M"].classes_))
        stage = st.selectbox("Stage", sorted(label_encoders["Stage"].classes_))
        response = st.selectbox("Response", sorted(label_encoders["Response"].classes_))
        submitted = st.form_submit_button("Predict")

    input_data = {
        "Age": age,
        "Gender": gender,
        "Smoking": smoking,
        "Hx Smoking": hx_smoking,
        "Hx Radiotherapy": hx_radiotherapy,
        "Thyroid Function": thyroid_function,
        "Physical Examination": physical_exam,
        "Adenopathy": adenopathy,
        "Pathology": pathology,
        "Focality": focality,
        "Risk": risk,
        "T": T,
        "N": N,
        "M": M,
        "Stage": stage,
        "Response": response
    }
    return pd.DataFrame([input_data]), submitted

# ✅ Prediction function
def predict(model, scaler, label_encoders, input_df):
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])
    input_df = input_df[FEATURE_NAMES]
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    return pred, prob

# ✅ Main App
def main():
    st.set_page_config(page_title="Thyroid Recurrence Predictor", layout="centered")
    st.title("🧠 Thyroid Cancer Recurrence Predictor")
    st.markdown("Enter patient details to predict the likelihood of thyroid cancer recurrence.")

    model, scaler, label_encoders = load_model_components()
    input_df, submitted = get_user_input(label_encoders)

    if submitted:
        pred, prob = predict(model, scaler, label_encoders, input_df)
        st.subheader("Prediction Result")
        st.write(f"🔍 **Prediction**: {'Recurred' if pred == 1 else 'Did Not Recur'}")
        st.write(f"📊 **Probability of Recurrence**: {prob:.2%}")

    st.markdown("""
    ---
    <div style='text-align: center; font-size: 14px; color: gray;'>
        © 2025 | Built by <strong>@benjaminchukwuemekaokere</strong>
    </div>
    """, unsafe_allow_html=True)

# ✅ Run app
if __name__ == "__main__":
    main()
