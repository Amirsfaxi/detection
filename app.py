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
import uvicorn

app = FastAPI()

# Class labels
class_names = ['Flea_Allergy', 'Health', 'Ringworm', 'Scabies']

# Paths and Google Drive config
MODEL_PATH = "best_model.pth"
GDRIVE_FILE_ID = "1WitEsENhyAu4bQCvhhkJWxmgULBhM2_4"
GDRIVE_URL = "https://drive.google.com/uc?export=download"

# Utility: download large file from Google Drive (handles virus warning pages)
def download_from_google_drive(file_id, destination):
    session = requests.Session()
    response = session.get(GDRIVE_URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(GDRIVE_URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    # Look for the 'download_warning' cookie
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
    # Check if the downloaded file appears to be HTML (i.e. starts with '<')
    with open(destination, "rb") as f:
        beginning = f.read(100).strip()
        if beginning.startswith(b"<"):
            raise ValueError("Error: Downloaded file appears to be HTML. "
                             "This may indicate a problem with the Google Drive link or download confirmation.")

# Download model if needed
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    download_from_google_drive(GDRIVE_FILE_ID, MODEL_PATH)
    print("Download complete.")

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
# Use weights_only=False explicitly for PyTorch 2.6+
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=False))
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

# For local testing, bind to the port provided by the PORT environment variable (Render will use its own)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
