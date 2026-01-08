import streamlit as st
import joblib
import numpy as np

# Load saved models
tfidf = joblib.load("models/tfidf.pkl")
clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")
le = joblib.load("models/label_encoder.pkl")

st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge")
st.write("Predict programming problem difficulty using text description")

# Input boxes
title = st.text_area("Problem Title")
description = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")

if st.button("Predict Difficulty"):
    full_text = title + " " + description + " " + input_desc + " " + output_desc
    
    if full_text.strip() == "":
        st.warning("Please enter problem details")
    else:
        X_text = tfidf.transform([full_text])
        
        # Add dummy numeric features (same as training)
        text_length = len(full_text)
        keyword_count = sum(full_text.lower().count(k) for k in ["graph", "dp", "tree", "recursion"])
        
        X = np.hstack([X_text.toarray(), [[text_length, keyword_count]]])
        
        # Predictions
        class_pred = clf.predict(X)[0]
        score_pred = reg.predict(X)[0]
        
        difficulty = le.inverse_transform([class_pred])[0]
        
        st.success(f"Predicted Difficulty Class: **{difficulty}**")
        st.success(f"Predicted Difficulty Score: **{score_pred:.2f}**")
