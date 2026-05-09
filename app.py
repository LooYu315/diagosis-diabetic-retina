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
            models_list = load_ensemble()
            if len(models_list) < 5:
                st.error(f"Error: Found only {len(models_list)}/5 models.")
            else:
                input_tensor = preprocess_image(image)
                
                # --- 1. Inference Logic ---
                all_probs = []
                with torch.no_grad():
                    for m in models_list:
                        output = m(input_tensor)
                        prob = torch.softmax(output, dim=1)
                        all_probs.append(prob)
                
                # Calculate average probabilities
                avg_probs = torch.mean(torch.stack(all_probs), dim=0)
                final_prediction = torch.argmax(avg_probs, dim=1).item()
                final_confidence = torch.max(avg_probs).item()
                
                # --- 2. Final Result UI ---
                classes = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
                
                st.divider()
                st.subheader("🏁 Final Diagnosis Result")
                if final_prediction == 0:
                    st.success(f"**{classes[final_prediction]}** (Confidence: {final_confidence*100:.2f}%)")
                else:
                    st.warning(f"**{classes[final_prediction]}** (Confidence: {final_confidence*100:.2f}%)")

                # --- 3. Probability Distribution Chart ---
                st.divider()
                st.subheader("📈 Probability Distribution (All Classes)")
                
                conf_data = {
                    "Stage": classes,
                    "Confidence (%)": [float(p) * 100 for p in avg_probs[0]]
                }
                # Display horizontal bar chart
                st.bar_chart(data=conf_data, x="Stage", y="Confidence (%)", horizontal=True)

                # --- 4. Individual Model Metrics ---
                st.divider()
                st.subheader("📊 Individual Model Analysis")
                cols = st.columns(5)
                
                for i, prob in enumerate(all_probs):
                    conf = torch.max(prob).item()
                    pred = torch.argmax(prob).item()
                    with cols[i]:
                        st.metric(label=f"Model {i}", value=f"{conf*100:.1f}%")
                        st.caption(f"Class: {pred}")
                
                st.info("The final result is the ensemble average of these 5 specialized models.")
