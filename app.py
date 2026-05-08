import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os

# --- 1. Model Architecture ---
def get_model(num_classes=5):
    # Matches the DenseNet121 architecture provided in infer.py
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

@st.cache_resource
def load_ensemble():
    ensemble = []
    # Loading 5-fold models from the root directory
    for i in range(5):
        path = f"best_model_fold{i}.pth"
        if os.path.exists(path):
            m = get_model(num_classes=5)
            m.load_state_dict(torch.load(path, map_location='cpu'))
            m.eval()
            ensemble.append(m)
    return ensemble

# --- 2. Advanced Preprocessing (Ben Graham + CLAHE) ---
def preprocess_image(image):
    # Convert PIL to CV2
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Crop black borders (matches infer.py logic)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 5
    if np.any(mask):
        img = img[np.ix_(mask.any(1), mask.any(0))]
    
    # Resize and Ben Graham's processing
    img = cv2.resize(img, (512, 512))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
    
    # CLAHE Enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Normalize (Standard ImageNet values)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(Image.fromarray(img)).unsqueeze(0)

# --- 3. Streamlit UI (English Interface) ---
st.set_page_config(page_title="DR AI Diagnosis System", layout="centered")

st.title("👁️ Diabetic Retinopathy Detection")
st.markdown("### Intelligent Ensemble Diagnosis System (DenseNet121)")

uploaded_file = st.file_uploader("Upload Retinal Fundus Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Fundus Image", use_column_width=True)
    
    if st.button("Run AI Diagnosis"):
        with st.spinner("Processing with 5 Ensemble Models..."):
            # 1. Load Models
            models_list = load_ensemble()
            if len(models_list) < 5:
                st.error(f"Error: Found only {len(models_list)}/5 models. Please check your .pth files.")
            else:
                # 2. Preprocess
                input_tensor = preprocess_image(image)
                
                # 3. Inference
                all_probs = []
                with torch.no_grad():
                    for m in models_list:
                        output = m(input_tensor)
                        all_probs.append(torch.softmax(output, dim=1))
                    
                    # Average probabilities across all 5 folds
                    avg_probs = torch.mean(torch.stack(all_probs), dim=0)
                    prediction = torch.argmax(avg_probs, dim=1).item()
                    confidence = torch.max(avg_probs).item()
                
                # 4. Display Results
                classes = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
                
                st.subheader("Diagnosis Result:")
                if prediction == 0:
                    st.success(f"Result: {classes[prediction]}")
                else:
                    st.warning(f"Result: {classes[prediction]}")
                
                st.write(f"**Confidence Score:** {confidence*100:.2f}%")
                st.info("Note: This is an AI-generated report for reference only. Please consult a doctor.")
