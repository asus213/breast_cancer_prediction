import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

model = pickle.load(open('breast_cancer_model.pkl', 'rb'))
df = pd.read_csv("breast-cancer.csv")

X = df.drop(["id", "diagnosis"], axis=1)
y = df["diagnosis"].map({"M": 0, "B": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

st.title("Breast Cancer Prediction App")

input_data = st.text_input("Enter 30 features separated by comma")

if st.button("Predict"):
    data = np.array([float(i) for i in input_data.split(",")]).reshape(1, -1)
    prediction = model.predict(data)

    if prediction[0] == 0:
        st.error("Malignant (Cancer)")
    else:
        st.success("Benign (No Cancer)")
