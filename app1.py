# streamlit app to predict and display digit images
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
from PIL import Image

# Load the trained model and scaler
model = joblib.load('Model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9) to get the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Load image with PIL
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption='Uploaded Image.', use_container_width=True)

    st.write("Classifying...")

    # Convert PIL → numpy array → OpenCV BGR format
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian blur (optional)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Apply threshold (MNIST style)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Resize to 28×28
    result = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

    # Show preprocessed image
    st.image(result, caption='Preprocessed Image.', channels="GRAY",
             use_container_width=True)

    # Convert to numpy array for model
    image_array = result.flatten().reshape(1, -1)

    # Scale image using your scaler
    image_scaled = scaler.transform(image_array)

    # Prediction
    prediction = model.predict(image_scaled)

    st.success(f"Predicted Digit: {prediction[0]}")

else:
    st.warning("Please upload an image file.")
