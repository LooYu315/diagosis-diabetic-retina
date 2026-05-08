import os
import argparse
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Model Architecture ---
def get_model(num_classes=5):
    # Matches DenseNet121 architecture used in training
    model = models.densenet121()
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

# --- 2. Preprocessing (Ben Graham + CLAHE) ---
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # 1. Convert BGR to RGB (matches Kaggle line 56)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Crop black borders (matches Kaggle lines 59-63)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > 5
    if np.any(mask):
        img = img[np.ix_(mask.any(1), mask.any(0))]
    
    img = cv2.resize(img, (512, 512))
    
    # 4. Ben Graham (matches Kaggle line 65)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
    
    # 5. CLAHE (matches Kaggle lines 66-69)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    temp_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', temp_bgr) # Simulate the save-to-disk step
    img_final = cv2.imdecode(buffer, cv2.IMREAD_COLOR) # Reload it
    img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB) # Back to RGB for PIL
    
    return Image.fromarray(img_final)

class TestDataset(Dataset):
    def __init__(self, df, images_path, transform=None):
        self.df = df
        self.images_path = images_path
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image']
        label = row['Label']
        
        image = preprocess_image(os.path.join(self.images_path, img_name))
        
        if image is None:
            image = Image.new('RGB', (512, 512), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

def run_inference(csv_path, images_path, output_filename):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load Labels
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Image'])
    
    val_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    loader = DataLoader(TestDataset(df, images_path, val_trans), batch_size=16, shuffle=False, num_workers=2)

    # Load 5-Fold Ensemble Models
    ensemble = []
    for i in range(5):
        model_path = f"models/best_model_fold{i}.pth"
        if os.path.exists(model_path):
            m = get_model(5).to(DEVICE)
            m.load_state_dict(torch.load(model_path, map_location=DEVICE))
            m.eval()
            ensemble.append(m)
            print(f"Loaded {model_path}")
    
    if not ensemble:
        print("Error: No model weights found!")
        return

    results = []
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels, names in tqdm(loader, desc="Running Inference"):
            images = images.to(DEVICE)
            
            # Ensemble Prediction: Average softmax probabilities across 5 models
            outputs = torch.stack([torch.softmax(m(images), dim=1) for m in ensemble])
            avg_probs = torch.mean(outputs, dim=0)
            preds = torch.argmax(avg_probs, dim=1)
            
            for i in range(len(names)):
                results.append({
                    "Image": names[i],
                    "Actual": labels[i].item(),
                    "Predicted": preds[i].item()
                })
                y_true.append(labels[i].item())
                y_pred.append(preds[i].item())

    # Save CSV Output
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_filename, index=False)
    
    # Calculate Final Metrics
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nFinal Test Results:")
    print(f"Quadratic Kappa: {kappa:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Generate and Save Confusion Matrix
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Ensemble Confusion Matrix')
    plt.savefig('test_confusion_matrix.png')
    
    print(f"\nCompleted! CSV saved as: {output_filename}")
    print("Confusion matrix saved as: test_confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Inference Script")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to test_labels.csv")
    parser.add_argument("--images_path", type=str, required=True, help="Path to folder containing test images")
    parser.add_argument("--output_filename", type=str, default="submission.csv", help="Name of output CSV file")
    
    args = parser.parse_args()
    run_inference(args.csv_path, args.images_path, args.output_filename)