# Importing the necessary libraries
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

import torch
import torchvision.transforms as transforms
from model import FasterRCNNResNet50
from PIL import Image

import io
import requests

# Download the checkpoint
MODEL_PATH = "checkpoint.pth"
HF_URL = (
    "https://huggingface.co/brandonRafael/outfit-ai/resolve/main/checkpoint_epoch_4.pth"
)


def download_pth():
    response = requests.get(HF_URL)

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)


# Initialize the model and load the checkpoint
download_pth()
checkpoint = torch.load(MODEL_PATH)
model = FasterRCNNResNet50()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Define the request and response models
class ImageData(BaseModel):
    image_bytes: bytes


def preprocess_image(image_bytes):
    transform = transforms.Compose(
        [
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image)


# Initialize the FastAPI app
app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    image_input = list(image_tensor.unsqueeze(0))

    with torch.no_grad():
        pred = model(image_input)[0]

    score_threshold = 0.5
    scores = pred["scores"]
    mask = scores >= score_threshold

    filtered_boxes = pred["boxes"][mask]
    filtered_labels = pred["labels"][mask]
    filtered_scores = pred["scores"][mask]

    return {
        "boxes": filtered_boxes.tolist(),
        "labels": filtered_labels.tolist(),
        "scores": filtered_scores.tolist(),
    }


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

