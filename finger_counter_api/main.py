from fastapi import FastAPI, Form, HTTPException,Request
from fastapi.responses import JSONResponse ,HTMLResponse
import cv2
import numpy as np
import base64
import os
import requests
import threading
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
from fastapi.middleware.cors import CORSMiddleware

# Import your custom HandDetector and analyze_age_gender functions
from HandTrackingModule import HandDetector


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")
logger = logging.getLogger(__name__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains to make requests. Change this to a specific URL if needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)
# Set template directory
templates = Jinja2Templates(directory="templates")
# At the top of your file
last_gender_age_result = {}

@app.get("/finger", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def decode_base64_image(data: str):
    try:
        format, imgstr = data.split(',', 1)
        image_bytes = base64.b64decode(imgstr)
        np_array = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")

def detect_fingers(img, detector, tip_ids):
    img = detector.findHands(img)
    lm_list = detector.findPosition(img, draw=False)
    fingers = []

    if len(lm_list) != 0:
        # Thumb
        fingers.append(1 if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1] else 0)
        # Other fingers
        for i in range(1, 5):
            fingers.append(1 if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2] else 0)

    return fingers.count(1)

@app.post("/analyze-finger")
async def analyze_fingers(image: str = Form(...)):
    try:
        img = decode_base64_image(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    detector = HandDetector(detectionCon=0.75)
    tip_ids = [4, 8, 12, 16, 20]

    try:
        num_fingers = detect_fingers(img, detector, tip_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Finger detection failed: {str(e)}")

    save_dir = "images_clicked"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "detected_hand.jpg")

    try:
        cv2.imwrite(save_path, img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Saving or analysis failed: {str(e)}")

    # Send image to the gender/age analysis service (running on port 8002)
    threading.Thread(target=send_to_gender_age_service, args=(save_path,)).start()

    # ✅ Return only finger result immediately
    return JSONResponse(content={"success": True, "num_fingers": num_fingers})


def send_to_gender_age_service(image_path: str) -> dict:
    global last_gender_age_result  # use global variable

    try:
        url = "https://pythontest.internetresearchbureau.com/gender-age-detection/detect-gender-age/"
        
        with open(image_path, 'rb') as f:
            response = requests.post(url, files={'image': f})
            

        if response.status_code == 200:
            result = response.json()
            last_gender_age_result = result  # ✅ Store in memory
            return result
        else:
            last_gender_age_result = {"error": response.text}
            return last_gender_age_result
    except Exception as e:
        last_gender_age_result = {"error": str(e)}
        return last_gender_age_result

from fastapi import Query

@app.get("/get-last-gender-age-result")
async def get_last_gender_age_result(surveyTrackId: str = Query(...)):
    global last_gender_age_result

    if not last_gender_age_result:
        return JSONResponse(content={"message": "No result available yet"}, status_code=404)

    gender = last_gender_age_result.get("gender")
    age = last_gender_age_result.get("age")

    if gender and age:
        try:
            # Path of last saved image
            image_path = "images_clicked/detected_hand.jpg"
            filename = f"{surveyTrackId}.jpg"  # ← Use custom filename from frontend

            with open(image_path, 'rb') as img_file:
                upload_url = "https://sales.pasmt.com/aws-bucket/upload?path=python-test"
                upload_response = requests.post(
                    upload_url,
                    files={'file': (filename, img_file, 'image/jpeg')}
                )

            if upload_response.status_code == 200:
                last_gender_age_result["image_upload"] = "success"
                last_gender_age_result["uploaded_filename"] = filename
            else:
                last_gender_age_result["image_upload"] = f"failed: {upload_response.text}"

        except Exception as e:
            last_gender_age_result["image_upload"] = f"error: {str(e)}"

    return JSONResponse(content=last_gender_age_result)
