import streamlit as st
import pickle
import numpy as np

# Load the regression model from the uploaded pickle file
def load_model(uploaded_file):
    model = pickle.load(uploaded_file)  # Use the file-like object directly
    return model

# Make predictions based on user input
def predict(model, features):
    prediction = model.predict([features])
    return prediction[0]

# Streamlit App Title and Description (Using HTML and CSS)
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
        }
        .description {
            font-size: 18px;
            text-align: center;
            color: #333;
        }
        .predict-button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            border: none;
            cursor: pointer;
            border-radius: 8px;
        }
        .predict-button:hover {
            background-color: #45a049;
        }
        .input-label {
            font-weight: bold;
            margin-top: 10px;
            color: #4CAF50;
        }
    </style>
    <div class="title">Regression Model Prediction</div>
    <p class="description">Upload your trained model and input feature values to get predictions.</p>
""", unsafe_allow_html=True)

# Upload the pickle file
uploaded_file = st.file_uploader("Upload your trained regression model (Pickle file)", type=["pkl"])

if uploaded_file is not None:
    try:
        # Load the model
        model = load_model(uploaded_file)
        st.success("Model loaded successfully!")
        
        # Input fields for user to provide features (adjust based on your model's feature count)
        principalComponents_1 = st.number_input("Feature 1", value=0.0, key="feature_1")
        principalComponents_2 = st.number_input("Feature 2", value=0.0, key="feature_2")
        principalComponents_3 = st.number_input("Feature 3", value=0.0, key="feature_3")
        principalComponents_4 = st.number_input("Feature 4", value=0.0, key="feature_4")
        principalComponents_5 = st.number_input("Feature 5", value=0.0, key="feature_5")
        principalComponents_6 = st.number_input("Feature 6", value=0.0, key="feature_6")
        
        # Create a list of features
        features = [principalComponents_1, principalComponents_2, principalComponents_3, principalComponents_4, principalComponents_5,principalComponents_6]
        
        # Predict button with custom CSS
        if st.button("Predict", key="predict"):
            prediction = predict(model, features)
            st.markdown(f"<h3>Predicted Value: {prediction}</h3>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.warning("Please upload a pickle file to proceed.")
