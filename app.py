
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import torch.nn.functional as F

treatment_df = pd.read_csv("Treatment_dataset.csv")
classes = treatment_df["disease_Class_Name"].tolist()
num_classes = len(classes)

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("plant_disease_model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

#Streamlit UI
st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image to detect plant disease and see treatment recommendations.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            confidence = confidence.item()
            class_index = predicted.item()

        threshold = 0.9
        if confidence < threshold:
            st.error("⚠️ Invalid photo or unrecognized plant leaf. Please upload a clear leaf image.")
            st.write(confidence)
        else:
            info = get_disease_info(class_index)
            st.subheader("Prediction")
            st.write(f"**Disease:** {info['disease']} (Confidence: {confidence:.2f})")
            st.write(f"**Description:** {info['description']}")
            st.subheader("Recommended Treatments")
            for t in info["treatments"]:
                if t.strip():
                    st.markdown(f"- {t}")
            st.subheader("Bio and Organic Treatments")
            for t in info["bio_treatments"]:
                if t.strip():
                    st.markdown(f"- {t}")
