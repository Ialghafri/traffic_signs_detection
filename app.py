import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt


FASTAPI_URL = "http://127.0.0.1:8000"

st.title("Traffic Sign Detection")
st.write("Upload an image to detect appearing traffic signs!")

uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Send image to backend
    with st.spinner("Classifying..."):
        response = requests.post("http://127.0.0.1:8000/predict/", files={"file": uploaded_file.getvalue()})
        if response.status_code == 200:
            result = response.json()
            st.write(f"Predicted Classes: {result['predicted_classes']}")
        else:
            st.error(f"Error: {response.json()['error']}")




# #upload
# if uploaded_file is not None:

#     image = Image.open(uploaded_file)

#     st.image(image, caption="uploaded image or video", use_column_width= True)
#     #st.write("Classifying...")

#     #st.write(f"Predicted Classes: {predicted_class_names}")

#     with st.spinner("Classifying..."):
#         files = {"file": uploaded_file.getvalue()}
#         response = requests.post(FASTAPI_URL, files=files)

#     prediction_response = requests.get("http://127.0.0.1:8000/predict/")

#     # annotated image is generated with bounding boxes
#     annotated_image = result.plot()  # use result.plot() to get the annotated image
    
#     # show the annotated image
#     st.image(annotated_image, caption="Annotated Image with Bounding Boxes", use_column_width=True)

#     # Handle response
#     if response.status_code == 200:
#         predictions = response.json().get("predicted_classes", [])
#         annotated_image = response.json().get("annotated_image", [])

#         st.write(f"Predicted Classes: {predictions}")
#     else:
#         st.error(f"Error: {response.json().get('error', 'Unknown error')}")

    # # Send POST request to FastAPI
    # response = requests.post(FASTAPI_URL, files={"file": (result, annotated_image, predicted_class_names})

        












