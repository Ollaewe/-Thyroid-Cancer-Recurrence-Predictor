import streamlit as st
import pandas as pd
import pickle

# ✅ Load model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Feature column order (same as during training)
feature_names = [
    "Age", "Gender", "Smoking", "Hx Smoking", "Hx Radiotherapy",
    "Thyroid Function", "Physical Examination", "Adenopathy",
    "Pathology", "Focality", "Risk", "T", "N", "M", "Stage", "Response"
]

# ✅ Load scaler and encoders
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# ✅ Streamlit UI
st.title("🧠 Thyroid Cancer Recurrence Predictor")
st.markdown("Enter patient details to predict the likelihood of thyroid cancer recurrence.")



form = st.form("patient_form")

# Input form
age = form.number_input("Age", min_value=1, max_value=120, value=30)
gender = form.selectbox("Gender", ["F", "M"])
smoking = form.selectbox("Smoking", ["No", "Yes"])
hx_smoking = form.selectbox("Hx Smoking", ["No", "Yes"])
hx_radiotherapy = form.selectbox("Hx Radiotherapy", sorted(label_encoders["Hx Radiotherapy"].classes_))
thyroid_function = form.selectbox("Thyroid Function", sorted(label_encoders["Thyroid Function"].classes_))
physical_exam = form.selectbox("Physical Examination", sorted(label_encoders["Physical Examination"].classes_))
adenopathy = form.selectbox("Adenopathy", sorted(label_encoders["Adenopathy"].classes_))
pathology = form.selectbox("Pathology", sorted(label_encoders["Pathology"].classes_))
focality = form.selectbox("Focality", sorted(label_encoders["Focality"].classes_))
risk = form.selectbox("Risk", sorted(label_encoders["Risk"].classes_))
T = form.selectbox("T Stage", sorted(label_encoders["T"].classes_))
N = form.selectbox("N Stage", sorted(label_encoders["N"].classes_))
M = form.selectbox("M Stage", sorted(label_encoders["M"].classes_))
stage = form.selectbox("Stage", sorted(label_encoders["Stage"].classes_))
response = form.selectbox("Response", sorted(label_encoders["Response"].classes_))

submitted = form.form_submit_button("Predict")

if submitted:
    # ✅ Build and preprocess input DataFrame
    input_dict = {
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

    input_df = pd.DataFrame([input_dict])

    # ✅ Apply label encoders
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # ✅ Ensure column order
    input_df = input_df[feature_names]

    # ✅ Scale input
    input_scaled = scaler.transform(input_df)

    # ✅ Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # ✅ Display results
    st.subheader("Prediction Result")
    st.write(f"🔍 **Prediction**: {'Recurred' if pred == 1 else 'Did Not Recur'}")
    st.write(f"📊 **Probability of Recurrence**: {prob:.2%}")
    # ✅ Footer
    st.markdown("""
    ---
    <div style='text-align: center; font-size: 14px; color: gray;'>
        © 2025 | Built by <strong>@benjaminchukwuemekaokere</strong>
    </div>
    """, unsafe_allow_html=True)