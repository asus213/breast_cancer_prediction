
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('breast_cancer_model.pkl', 'rb'))

st.title("Breast Cancer Prediction App")

input_data = st.text_input("Enter 30 features separated by comma")

if st.button("Predict"):
    data = np.array([float(i) for i in input_data.split(",")]).reshape(1, -1)
    prediction = model.predict(data)

    if prediction[0] == 0:
        st.error("Malignant (Cancer)")
    else:
        st.success("Benign (No Cancer)")
