import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open('bike_model.pkl', 'rb'))

st.title("🚲 Bike Rental Demand Prediction")

# Input fields
season = st.selectbox("Season", [1, 2, 3, 4])
yr = st.selectbox("Year (0 = 2011, 1 = 2012)", [0, 1])
mnth = st.slider("Month", 1, 12)
holiday = st.selectbox("Holiday (0 = No, 1 = Yes)", [0, 1])
weekday = st.slider("Weekday (0=Sunday, 6=Saturday)", 0, 6)
workingday = st.selectbox("Working Day (0 = No, 1 = Yes)", [0, 1])
temp = st.slider("Normalized Temperature (0 to 1)", 0.0, 1.0, 0.5)
hum = st.slider("Normalized Humidity (0 to 1)", 0.0, 1.0, 0.5)
windspeed = st.slider("Normalized Windspeed (0 to 1)", 0.0, 1.0, 0.2)

# Prediction
input_data = np.array([[season, yr, mnth, holiday, weekday,
                        workingday, temp, hum, windspeed]])
prediction = model.predict(input_data)[0]

st.markdown(f"### 🔮 Predicted Bike Count: **{int(prediction)}**")