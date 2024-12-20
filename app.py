import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pickle

# Load the model
model = tf.keras.models.load_model('neural_networg_r.keras')

# Streamlit app
st.title("Fire Detect")

st.write("This app predicts area surface affected by forest fire.")

# Load LabelEncoders
with open('month_encoder.pkl', 'rb') as f:
    le_month = pickle.load(f)

with open('day_encoder.pkl', 'rb') as f:
    le_day = pickle.load(f)

# Input values from the user
X = st.number_input("X coordinate")
Y = st.number_input("Y coordinate")
FFMC = st.number_input("FFMC")
DMC = st.number_input("DMC")
DC = st.number_input("DC")
ISI = st.number_input("ISI")
temp = st.number_input("Temperature")
RH = st.number_input("Relative Humidity")
wind = st.number_input("Wind Speed")
rain = st.number_input("Rainfall")
month = st.selectbox("Month", le_month.classes_)
day = st.selectbox("Day", le_day.classes_)

# Encode 'month' and 'day'
month_encoded = le_month.transform([month])[0]
day_encoded = le_day.transform([day])[0]

# Prepare the input data
input_data = np.array([[X, Y, FFMC, DMC, DC, ISI, temp, RH, wind, rain, month_encoded, day_encoded]])

# Ensure input shape matches the model's expected shape
if input_data.shape[1] != 12:
    st.error(f"Expected 12 features, but got {input_data.shape[1]}. Please check your inputs.")
else:
    # Make a prediction
    prediction = model.predict(input_data)
    st.write(f"Predicted Area: {prediction[0][0]:.2f} ha")