from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
from torch import nn
import io

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


app = FastAPI(title="Traffic Sign Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="ui"), name="ui")


@app.get("/")
def root():
    return FileResponse("ui/index.html")


CLASS_NAMES = {
    0: "Speed limit 20km/h", 1: "Speed limit 30km/h", 2: "Speed limit 50km/h",
    3: "Speed limit 60km/h", 4: "Speed limit 70km/h", 5: "Speed limit 80km/h",
    6: "End speed limit 80km/h", 7: "Speed limit 100km/h", 8: "Speed limit 120km/h",
    9: "No passing", 10: "No passing for vehicles over 3.5t",
    11: "Right of way at intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "No vehicles over 3.5t",
    17: "No entry", 18: "General caution", 19: "Dangerous curve left",
    20: "Dangerous curve right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End all restrictions", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Ahead or right", 37: "Ahead or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End no passing", 42: "End no passing for vehicles over 3.5t"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3337, 0.3064, 0.3171],
                         std=[0.2672, 0.2564, 0.2629])
])

class CNNBaseline(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(hidden_units*2, hidden_units*4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(hidden_units*4*4*4, 256), nn.ReLU(), nn.Linear(256, output_shape))

    def forward(self, x):
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))

def load_model():
    model = CNNBaseline(input_shape=3, hidden_units=32, output_shape=43)
    model.load_state_dict(torch.load("/home/uno21/models/baseline_cnn.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

@app.get("/")
def root():
    return {"status": "ok", "model": "CNNBaseline GTSRB", "classes": 43}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()

    return {
        "class_id": pred_class,
        "class_name": CLASS_NAMES[pred_class],
        "confidence": round(confidence * 100, 2)
    }