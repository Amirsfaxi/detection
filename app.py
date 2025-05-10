# save as app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import io
import os
import requests

app = FastAPI()

# Class labels
class_names = ['Flea_Allergy', 'Health', 'Ringworm', 'Scabies']

# Paths and Google Drive config
MODEL_PATH = "best_model.pth"
GDRIVE_FILE_ID = "1WitEsENhyAu4bQCvhhkJWxmgULBhM2_4"
GDRIVE_URL = f"https://drive.google.com/uc?export=download"

# Utility: download large file from Google Drive (handles virus warning page)
def download_from_google_drive(file_id, destination):
    session = requests.Session()
    response = session.get(GDRIVE_URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(GDRIVE_URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Download model if needed
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    download_from_google_drive(GDRIVE_FILE_ID, MODEL_PATH)
    print("Download complete.")

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output, 1)

        result = class_names[pred.item()]
        return JSONResponse({"prediction": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
