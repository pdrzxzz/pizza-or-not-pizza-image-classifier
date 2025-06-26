import streamlit as st
import joblib
from imageio.v3 import imread
from io import BytesIO

from feature_extractor import extract_features_from_image

pca = joblib.load('model_files/pca.pkl')
scaler = joblib.load('model_files/scaler.pkl')
model = joblib.load('model_files/model.pkl')

st.title("Pizza or not pizza?")
st.text("Without deep learning. (Spoiler: this model might cry when it sees a lasagna.)")   

uploaded_file = st.file_uploader('Upload a food image.', type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    # Read file
    bytes = uploaded_file.read()
    image = imread(BytesIO(bytes))
    
    # Display image
    st.image(BytesIO(bytes), caption="Uploaded Image", use_container_width=True)

    # Extract Features
    features = extract_features_from_image(BytesIO(bytes))
    st.write("Extracted features shape:", features.shape)
    x = scaler.transform(features)

    # Apply PCA
    x = pca.transform(x)
    st.write("PCA features shape:", x.shape)

    # Predict
    pred = model.predict(x)[0]
    pred = "pizza" if pred == 1 else "not pizza" # Label encoding mapping: not_pizza -> 0, pizza -> 1
    st.write(f"Pred: {pred}")

