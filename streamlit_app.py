import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
with open('ckd_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset for visualization
df = pd.read_csv('kidney_disease.csv')
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Clean numeric columns
numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

# Title
st.title("🩺 Chronic Kidney Disease (CKD) Prediction System")
st.markdown("Please fill in the patient's health details below to predict the likelihood of CKD.")

# Input fields
age = st.slider("Age", 1, 100, 45)
bp = st.slider("Blood Pressure (mm Hg)", 50, 180, 80)
sg = st.selectbox("Specific Gravity (SG)", options=[1.005, 1.010, 1.015, 1.020, 1.025])
al = st.selectbox("Albumin", options=[0, 1, 2, 3, 4, 5])
su = st.selectbox("Sugar", options=[0, 1, 2, 3, 4, 5])
bgr = st.slider("Blood Glucose Random (mg/dl)", 70, 500, 150)
bu = st.slider("Blood Urea (mg/dl)", 1, 200, 30)
sc = st.slider("Serum Creatinine (mg/dl)", 0, 15, 1)
sod = st.slider("Sodium (mEq/L)", 100, 160, 140)
pot = st.slider("Potassium (mEq/L)", 2, 10, 4)
hemo = st.slider("Hemoglobin (g/dl)", 3, 17, 12)
pcv = st.slider("Packed Cell Volume", 10, 60, 35)
wc = st.slider("White Blood Cell Count (cells/cumm)", 2000, 25000, 8500)
rc = st.slider("Red Blood Cell Count (millions/cmm)", 2.5, 6.5, 4.5)

user_input = [age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc]

# Predict
if st.button("Predict CKD Status"):
    prediction = model.predict([user_input])[0]
    label = "CKD" if prediction == 1 else "Not CKD"

    st.subheader(f"🧾 Prediction: **{label}**")

    # Explanation note
    if label == "CKD":
        st.error("⚠️ The patient shows signs of Chronic Kidney Disease. It is advised to consult a nephrologist immediately for further tests and medical care.")
    else:
        st.success("✅ The patient's indicators do not suggest CKD at this time. However, regular health checkups are still recommended.")

    # Visualizations
    st.subheader("📊 Dataset Visualizations (for context)")

    with st.expander("Histogram of Features"):
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
