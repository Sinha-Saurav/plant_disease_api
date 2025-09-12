from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import torch.nn.functional as F

app = FastAPI(title="Plant Disease Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"]    
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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def prettify_label(label: str) -> str:
    if "___" in label:
        plant, disease = label.split("___", 1)
        disease = disease.replace("_", " ")
        return f"{plant} - {disease}"
    return label.replace("_", " ")

def clean_treatments(treatments: str):
    parts = [t.strip() for t in treatments.split("OR")]
    final_parts = []
    for p in parts:
        sub_parts = [sp.strip() for sp in p.split(".") if sp.strip()]
        final_parts.extend(sub_parts)
    return final_parts

def bio_organic_treatments(treatments: str):
    parts = [t.strip() for t in treatments.split(".")]
    final_parts = []
    for p in parts:
        final_parts.append(p)
    return final_parts


  
def get_disease_info(class_index: int):
    row = treatment_df.iloc[class_index]
    return {
        "disease": prettify_label(row["disease_Class_Name"]),
        "description": row["Disease_Description"],
        "treatments": clean_treatments(row["Treatment_Recommendations"]),
        "bio_treatments": bio_organic_treatments(row["Bio_Pesticides_Fertilizers"])
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        confidence = confidence.item()
        class_index = predicted.item()

    threshold = 0.75  # you can tune this
    if confidence < threshold:
        return {"prediction": "Invalid Photo", "confidence": confidence}

    info = get_disease_info(class_index)
    return {"prediction": info, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
