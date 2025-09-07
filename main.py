from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd

app = FastAPI(title="Plant Disease Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow requests from any website
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, etc.
    allow_headers=["*"]    # allow headers like Content-Type
)

# Load Treatment Dataset
treatment_df = pd.read_csv("Treatment_dataset.csv")

# Load Label Classes
classes = treatment_df["disease_Class_Name"].tolist()
num_classes = len(classes)

# Load Model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("plant_disease_model.pth", map_location="cpu"))
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # must match training input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Helper: Prettify Label
def prettify_label(label: str) -> str:
    if "___" in label:
        plant, disease = label.split("___", 1)
        disease = disease.replace("_", " ")
        return f"{plant} - {disease}"
    return label.replace("_", " ")

# Helper: Clean treatments into bullet points
def clean_treatments(treatments: str):
    # Split by "OR"
    parts = [t.strip() for t in treatments.split("OR")]
    # Also split further by newlines if present
    final_parts = []
    for p in parts:
        sub_parts = [sp.strip() for sp in p.split("\n") if sp.strip()]
        final_parts.extend(sub_parts)
    return final_parts

# Helper: Get Disease Info  
def get_disease_info(class_index: int):
    row = treatment_df.iloc[class_index]
    return {
        "disease": prettify_label(row["disease_Class_Name"]),
        "description": row["Disease_Description"],
        "treatments": clean_treatments(row["Treatment_Recommendations"])
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load image
    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_index = predicted.item()

    # Get disease info
    info = get_disease_info(class_index)

    return {"prediction": info}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
