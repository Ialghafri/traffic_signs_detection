import streamlit as st
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torchvision import models
import matplotlib.pyplot as plt
from ultralytics import YOLO

device = ("cpu")

@st.cache_data()  # Cache the model to avoid reloading every time
def load_model():
    model = YOLO('runs2/detect/train2/weights/best.pt')  # , weights_only=False
    model.eval()
    return model 

model = load_model()

st.title("Traffic Sign Detection")
st.write("Upload an image or a video to detect appearing traffic signs!")

uploaded_file = st.file_uploader(("Upload an image or a video"), type = ["jpeg", "jpg", "png", "mp4"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="uploaded image or video", use_column_width= True)
    st.write("Classifying...")


    prediction = model.predict(source=image, save=True, save_dir='inference_results')

    # predicted_labels = prediction.boxes.cls
    # predicted_classes = prediction.names
    # predicted_class_names = [predicted_classes[int(label)] for label in predicted_labels]

    # Extract the first prediction (it will be a list of results)
    result = prediction[0]  # First image's prediction

    # Get predicted class names and class IDs
    predicted_classes = result.names  # This contains the class names
    predicted_labels = result.boxes.cls  # The indices of predicted classes (labels)

    # Map predicted labels to class names
    predicted_class_names = [predicted_classes[int(label)] for label in predicted_labels]

    st.write(f"Predicted Classes: {predicted_class_names}")
    
    # Annotated image is generated with bounding boxes
    annotated_image = result.plot()  # Use result.plot() to get the annotated image
    
    # Show the annotated image
    st.image(annotated_image, caption="Annotated Image with Bounding Boxes", use_column_width=True)
