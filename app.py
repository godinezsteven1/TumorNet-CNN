import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("Convolutional Neural Network Scan: Brain Tumor Scanner - ")
st.subheader("Model Analytics Dashboard")

data = np.random.randn(1, 2)
df = pd.DataFrame(data, columns=["Accuracy over time", "Training loss over epochs"])


col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Accuracy & Loss")
    st.line_chart(df, use_container_width=True)

x-col2 = 'Type'
with col2:
    st.subheader("Tumor Prediction Distribution")
    data = pd.DataFrame({
        x-col2: ['Glioma', 'Meningioma', 'Tumor', 'Healthy'],
        'Count': [1, 1, 1, 1]
    })
    st.bar_chart(data.set_index(x-col2), use_container_width=True)  
