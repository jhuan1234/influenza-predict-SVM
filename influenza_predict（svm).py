import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from openpyxl import load_workbook
# Load the model
model = joblib.load('svm_model.pkl')
scale = joblib.load('scaler.pkl')
# Define feature names
feature_names = ['NEUT%','PCT','AST','Glucose','BUN','C3','B-cell %','total T-cell Count','CD4+ T-cell Count','NK-cell Count']
# Streamlit user interface
st.title("Critical influenza predictor for hospitalized children")
# NEUT%: numerical input
neut = st.number_input("NEUT%:", min_value=0.0, max_value=100.0, value=50.0)
# PCT: numerical input
pct = st.number_input("PCT:", min_value=0.0, max_value=200.0, value=0.5)
# AST: numerical input
ast = st.number_input("AST:", min_value=0.0, max_value=20000.0, value=100.0)
# Glucose: numerical input
glucose = st.number_input("Glucose:", min_value=0.0, max_value=100.0, value=5.0)
# BUN: numerical input
bun = st.number_input("BUN:", min_value=0.0, max_value=100.0, value=20.0)
# C3: numerical input
c3 = st.number_input("C3:", min_value=0.0, max_value=5.0, value=1.0)
# total T-cell Count: numerical input
tcount= st.number_input("total T-cell Count:", min_value=0.0, max_value=10000.0, value=10.0)
# B-cell%: numerical input
bcell = st.number_input("B-cell%:", min_value=0.0, max_value=100.0, value=20.0)
# CD4+Tcell count: numerical input
cd4 = st.number_input("CD4+T cell count:", min_value=0.0, max_value=50000.0, value=1000.0)
# NK-cell Count: numerical input
nk= st.number_input("NK cell:", min_value=0.0, max_value=10000.0, value=10.0)
# Process inputs and make predictions
feature_values = [neut,pct, ast, glucose, bun, c3, tcount, bcell, cd4,nk]
features = np.array([feature_values])
features_scale=pd.DataFrame(scale.transform(features),columns=feature_names)
if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features_scale)[0]
    predicted_proba = model.predict_proba(features_scale)[0]
    predicted_proba_s=model.predict_proba(features_scale)[0,1]
    # Display prediction results
    #st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Predicted Probability of critical influenza :** {predicted_proba_s*100:.2f}%")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, the child has a high risk of severe influenza. "
            f"The model predicts that the probability of having severe influenza is {predicted_proba_s:.2f}%. "
            "While this is just an estimate, it suggests that the child may be at significant risk. "
            "I recommend that you consult a pediatrician as soon as possible for further evaluation and "
            "to ensure the child receives an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, the child has a low risk of severe influenza. "
            f"The model predicts that the probability of having severe influenza is {predicted_proba_s:.2f}%. "
            "However, maintaining a healthy lifestyle and monitoring the child's health is still very important. "
            "I recommend regular check-ups to monitor the child's health, "
            "and to seek medical advice promptly if any symptoms develop."
        )
    #st.write(advice)

   # Calculate SHAP values and display force plot
    # 打开 Excel 文件
    file_path = "https://raw.githubusercontent.com/jhuan1234/influenza-predict-SVM/main/新建Microsoft%20Excel%20工作表.xlsx"      
    dataset_1 = load_workbook(filename=file_path)
    X_test=dataset_1.iloc[:,1:]
    X_test_scaled=scale.transform(X_test)
    background_data = shap.sample(X_test_scaled, 50)
    explainer = shap.KernelExplainer(model.predict_proba, background_data)
    shap_values = explainer.shap_values(features_scale)
    print(shap_values,features.shape)
    shap.force_plot(explainer.expected_value[1], shap_values[1],features,feature_names=feature_names,show=False,matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
