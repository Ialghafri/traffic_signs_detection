from fastapi import FastAPI, UploadFile, File
from PIL import Image
from ultralytics import YOLO
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from io import BytesIO
import numpy as np
import uvicorn
import os
import cv2
import subprocess


app = FastAPI()

def load_model():
    model = YOLO('runs2/detect/train2/weights/best.pt')  
    model.eval()
    return model 

model = load_model()


@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI"}

def read_imagefile(file) -> Image.Image:
    # Read and convert the uploaded image file into a PIL Image.
    # image = Image.open(uploaded_file)
    image = Image.open(BytesIO(file))
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file as an image
        image = read_imagefile(await file.read())

        # Perform prediction
        prediction = model.predict(source=image, save=True, save_dir='inference_results')
        result = prediction[0]  # First prediction
        
        # Extract predicted class names
        predicted_classes = result.names
        predicted_labels = result.boxes.cls  # Indices of predicted classes
        predicted_class_names = [predicted_classes[int(label)] for label in predicted_labels]

        # Generate annotated image
        annotated_image = result.plot()
        
        # Save the annotated image to the specified directory
        annotated_pil_image = Image.fromarray(annotated_image)
        save_path = os.path.join("annotated-images", f"annotated_{file.filename}")
        annotated_pil_image.save(save_path, format="JPEG")
        
        # Prepare the response
        response = {"predicted_classes": predicted_class_names}

        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

