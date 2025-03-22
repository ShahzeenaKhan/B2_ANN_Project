import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# =============================
# Load the trained regression model (predicting salary)
# =============================
model = tf.keras.models.load_model('model_salary.h5')

# =============================
# Load encoders and scaler
# =============================
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('Salary_Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# =============================
# Streamlit App UI
# =============================
st.set_page_config(page_title="Estimated Salary Prediction", layout="centered")

st.title("💰 Customer Estimated Salary Prediction App")

st.markdown("""
Welcome to the **Customer Salary Predictor**!  
Fill in customer details to estimate their **expected salary**.
""")

# =============================
# User Inputs in Sidebar
# =============================
st.sidebar.header("Input Customer Details")

geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 92, 35)
balance = st.sidebar.number_input('Balance', min_value=0.0, value=50000.0, step=1000.0)
credit_score = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=650, step=10)
tenure = st.sidebar.slider('Tenure', 0, 10, 5)
num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1], index=1)
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1], index=1)

# =============================
# Encode categorical inputs
# =============================
gender_encoded = label_encoder_gender.transform([gender])[0]
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# =============================
# Prepare input data
# =============================
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

# Combine input data with geography encoding
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# =============================
# Ensure columns match scaler expectation
# =============================
expected_columns = scaler.feature_names_in_.tolist()
input_data = input_data[expected_columns]  # Reorder columns to match training

# =============================
# Scale the input features
# =============================
input_data_scaled = scaler.transform(input_data)

# =============================
# Predict salary
# =============================
prediction = model.predict(input_data_scaled)

# If you applied log1p transformation during training, reverse it:
# predicted_salary = np.expm1(prediction[0][0])

predicted_salary = prediction[0][0]

# =============================
# Display the prediction result
# =============================
st.subheader("Prediction Result")
st.metric(label="Predicted Estimated Salary", value=f"${predicted_salary:,.2f}")

st.success("Prediction complete!")
