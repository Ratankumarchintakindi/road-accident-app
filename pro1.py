# accident_analysis_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Page title
st.title("🚗 Road Accident Severity Prediction App")

# File uploader
uploaded_file = st.file_uploader("Accident_information.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(Accident_information.csv)
    st.subheader("📄 Raw Data")
    st.write(df.head())

    # Keep only important features
    features = ['Weather_Conditions', 'Road_Surface_Conditions', 'Light_Conditions',
                'Speed_limit', 'Day_of_Week', 'Accident_Severity']
    df = df[features]
    df = df.dropna()

    # Label Encoding
    le = LabelEncoder()
    for col in ['Weather_Conditions', 'Road_Surface_Conditions', 'Light_Conditions', 'Day_of_Week']:
        df[col] = le.fit_transform(df[col])

    # Split and scale
    X = df.drop('Accident_Severity', axis=1)
    y = df['Accident_Severity']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Show evaluation
    st.subheader("📊 Model Evaluation")
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Extra: Let user input conditions to predict
    st.subheader("🔮 Predict Accident Severity")
    with st.form("predict_form"):
        speed = st.number_input("Speed Limit", min_value=0, max_value=150, value=50)
        weather = st.selectbox("Weather Conditions", df['Weather_Conditions'].unique())
        road = st.selectbox("Road Surface Conditions", df['Road_Surface_Conditions'].unique())
        light = st.selectbox("Light Conditions", df['Light_Conditions'].unique())
        day = st.selectbox("Day of Week", df['Day_of_Week'].unique())
        submitted = st.form_submit_button("Predict")

        if submitted:
            # Encode and predict
            input_data = pd.DataFrame([[weather, road, light, speed, day]],
                                      columns=['Weather_Conditions', 'Road_Surface_Conditions',
                                               'Light_Conditions', 'Speed_limit', 'Day_of_Week'])
            for col in input_data.columns:
                input_data[col] = le.fit(df[col]).transform(input_data[col])

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            st.success(f"Predicted Severity Level: {prediction[0]}")
