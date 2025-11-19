#streamlit app to predict and display digit images
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
from PIL import Image

# Load the trained model and scaler
model=joblib.load('Model.joblib')
scaler=joblib.load('scaler.joblib')

st.title("Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9) to get the prediction.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display the uploaded image
    image= Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")
# Preprocess the image
   
# Read image
    img = cv2.imread(image)

# 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Gaussian blur (optional, improves thresholding)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

# 3. Apply threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 4. Resize to MNIST size (28×28)
    result = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

# 5. Save output
    cv2.imwrite("output.png", result)

# Load and preprocess the saved image
    st.image("output.png", caption='Preprocessed Image.', use_container_width=True)

    image=cv2.imread("output.png", cv2.IMREAD_GRAYSCALE)
    image=np.array(image)
    image=image.flatten().reshape(1, -1)  # Flatten the image
    image_scaled=scaler.transform(image)  # Scale the image
# Make prediction
    prediction=model.predict(image_scaled)
    st.write(f"Predicted Digit: {prediction[0]}")
else:
    st.warning("Please upload an image file.")