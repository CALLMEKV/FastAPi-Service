# FastAPi-Service
# FastAPI Service 🚀

This repository contains a collection of microservices built using **FastAPI**. Each module serves a different purpose such as gender-age prediction, sentence checking, finger detection for captcha, and audio to text conversion.

---

## 🔧 Project Structure


fastapi-service/
├── gender-age-api/
├── sentence-checker-api/
├── finger-captcha-api/
├── audio-to-text-api/
└── README.md


---

## 🧠 Modules

### 1. `gender-age-api`
- Detects **gender** and **age** from an uploaded image.
- Uses models like `DeepFace` or custom-trained CNNs.
- **API Endpoint Example**: `/predict-gender-age`

---

### 2. `sentence-checker-api`
- Analyzes sentences for:
  - Grammar
  - POS tagging
  - Named Entity Recognition (NER)
  - Sentiment
- **API Endpoint Example**: `/analyze-sentence`

---

### 3. `finger-captcha-api`
- Detects number of fingers in an image.
- Used for image-based human verification.
- **API Endpoint Example**: `/detect-fingers`

---

### 4. `audio-to-text-api`
- Converts speech (audio/video) into text.
- Supports multiple languages using `Vosk`.
- **API Endpoint Example**: `/transcribe-audio`

---

## 🛠 How to Run a Module

Each module is a standalone FastAPI app.

Example for `gender-age-api`:
```bash
cd gender-age-api
uvicorn app.main:app --reload


You can configure port and host using:

uvicorn app.main:app --host 0.0.0.0 --port 8000

📦 Installation (per module)

cd gender-age-api  # or any module folder
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

📡 API Documentation
Each module exposes Swagger UI by default at:

http://localhost:8000/docs
