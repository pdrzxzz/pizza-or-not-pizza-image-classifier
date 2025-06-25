import streamlit as st
st.title("Pizza or not pizza?")
st.text("Without deep learning.")

uploaded_file = st.file_uploader('Upload a food image.')
if uploaded_file is not None:
    print(uploaded_file)