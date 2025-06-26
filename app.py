import streamlit as st
import joblib

from features import extract_features_from_image

pca = joblib.load('model_files/pca.pkl')
scaler = joblib.load('model_files/scaler.pkl')
model = joblib.load('model_files/model.pkl')

st.title("Pizza or not pizza?")
st.text("Without deep learning.")

uploaded_file = st.file_uploader('Upload a food image.', type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # print(uploaded_file)
    features = extract_features_from_image(uploaded_file)
    st.write("Extracted features shape:", features.shape)
    x = scaler.transform(features)
    x = pca.transform(x)
    st.write("PCA features shape:", x.shape)
