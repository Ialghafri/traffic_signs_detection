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
    model = YOLO('runs2/detect/train2/weights/best.pt')  # , weights_only=False
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
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Imporvements 
# -- import subprocess - include reload= True | "--reload"
# -- c1, c2, c3 = st.columns[(1,3,1)]

# ---- Excess code -----

# def read_imagefile(file: bytes) -> Image.Image:
#     try:
#         image = Image.open(uploaded_file)
#         return Image.open(BytesIO(file)).convert("RGB")
#     except Exception as e:
#         raise ValueError(f"Failed to read image file: {str(e)}")


#       return JSONResponse(content=response)

# @app.get("/predict/annotated-image")
# async def get_annotated_image():
#     try:
#         buffer = app.state.annotated_image_buffer
#         return FileResponse(buffer, media_type="image/jpeg", filename="annotated_image.jpg")
#     except AttributeError:
#         return JSONResponse(content={"error": "Annotated image not found. Make sure to call /predict/ first."}, status_code=404)


# convert to base

# response2 = {"annotated_image": imagee}
# JSONResponse(content=response), FileResponse(response2)

# @app.post("/predict/")
# async def predict(file: UploadFile):

#     contents = await file.read()

#     #image = Image.open(io.BytesIO(contents))

#     image = Image.open(uploaded_file)

#     #predict(uploaded_file, model)

#     prediction = model.predict(source=image, save=True, save_dir='inference_results')

#     result = prediction[0]  # First image's prediction

#     # Get predicted class names and class IDs
#     predicted_classes = result.names  # This contains the class names
#     predicted_labels = result.boxes.cls  # The indices of predicted classes (labels)
#     # Map predicted labels to class names
#     predicted_class_names = [predicted_classes[int(label)] for label in predicted_labels]

#     # annotated image is generated with bounding boxes
#     annotated_image = result.plot()  # use result.plot() to get the annotated image
        

#     return {"predicted_class": predicted_class_names, "Result": result, "Annotated image": annotated_image}