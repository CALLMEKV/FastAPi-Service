from fastapi import FastAPI, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import Request
import json
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import asyncio
import websockets

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize Vosk model
model = Model(model_path="vosk-model-small-en-us-0.15/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Audio stream parameters
    CHANNELS = 1
    RATE = 16000
    CHUNK = int(RATE * 0.5)  # 0.5 second chunks
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        if len(indata.shape) == 2:
            indata = indata.flatten()
        data = indata.tobytes()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            if result["text"]:
                asyncio.run(websocket.send_text(result["text"]))

    try:
        with sd.InputStream(channels=CHANNELS,
                          samplerate=RATE,
                          blocksize=CHUNK,
                          callback=audio_callback,
                          dtype=np.int16):
            while True:
                data = await websocket.receive_text()
                if data == "stop":
                    break
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"Error: {str(e)}") 