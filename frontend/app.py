import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import os
# from server.main import predict


FASTAPI_URL = "http://127.0.0.1:8000"

st.title("Traffic Sign Detection")
st.write("Upload an image to detect appearing traffic signs!")

uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    c1, c2, c3 = st.columns([0.1,10,0.1])
    with c2:
        st.image(image, caption="Uploaded Image", use_container_width=True)
   

    # Send image to backend
    with st.spinner("Classifying..."):
        #response = requests.post("http://127.0.0.1:8000/predict/", files={"file": uploaded_file.getvalue()})
        response = requests.post("http://week7imagedetection-server-1:8000/predict/", files={"file": uploaded_file.getvalue()})

        if response.status_code == 200:
            result = response.json()
            st.write(f"Predicted Classes: {result['predicted_classes']}")
            c1, c2, c3 = st.columns([0.1,10,0.1])
            with c2:
                st.image("/app/shared/annotated-images/annotated_file", caption="Annotated Image with Bounding Boxes", use_container_width=True)
            
        else:
            st.error(f"Error: {response.json()['error']}")







        












