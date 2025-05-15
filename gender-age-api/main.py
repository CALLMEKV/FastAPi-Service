import logging
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from deepface import DeepFace
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Set up logger
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains to make requests. Change this to a specific URL if needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load YOLO model and configuration
cfg_path = os.path.join('yolov3', 'yolov3.cfg')
weights_path = os.path.join('yolov3', 'yolov3.weights')
coco_names_path = os.path.join('yolov3', 'coco.names')

net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load classes for YOLO (COCO dataset classes)
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

class ImageData(BaseModel):
    image_name: str

@app.post("/detect-gender-age/")
async def detect_gender_age(image: UploadFile = File(...)):
    try:
        # Save the image temporarily in memory to avoid using too much memory
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(await image.read())  # Write image to temporary file
            tmp_file_path = tmp_file.name
        
        # Load the image from the temporary file
        try:
            image = Image.open(tmp_file_path).convert('RGB')
            logger.info("Image loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Convert PIL image to numpy array
        image_np = np.array(image)
        logger.info("Image converted to numpy array.")

        # Analyze the image for age and gender using DeepFace
        try:
            result = DeepFace.analyze(img_path=image_np, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
            logger.info("DeepFace analysis successful.")
        except Exception as e:
            logger.error(f"Error in DeepFace analysis: {e}")
            raise HTTPException(status_code=500, detail="Error in DeepFace analysis")

        # Extract results from DeepFace
        age = result[0]['age']
        dominant_gender = result[0]['dominant_gender']
        dominant_emotion = result[0]['dominant_emotion']
        dominant_race = result[0]['dominant_race']
        logger.info(f"Age: {age}, Dominant Gender: {dominant_gender}, Dominant Emotion: {dominant_emotion}, Race: {dominant_race}")

        # Perform object detection using YOLO
        blob = cv2.dnn.blobFromImage(image_np, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        height, width, channels = image_np.shape

        # Process the detections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detected_objects = []

        # Draw the boxes and labels for objects
        for i in range(len(boxes)):
            if i in indexes.flatten():  # Check against flattened indexes
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                detected_objects.append(label)

        logger.info(f"Detected objects: {detected_objects}")

        # Clean up the temporary file after processing
        os.remove(tmp_file_path)

        return {
            'age': age,
            'gender': dominant_gender,
            'emotion': dominant_emotion,
            'race': dominant_race,
            'objects': detected_objects
        }

    except Exception as e:
        logger.error(f"Error in the detect_gender_age endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/gender")
def read_root():
    return {"message": "FastAPI Gender and Age Detection API"}
