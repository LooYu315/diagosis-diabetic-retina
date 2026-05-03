import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from sklearn.metrics import (
    cohen_kappa_score, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

# --- CONFIGURATION ---
RAW_IMAGE_PATH = "/kaggle/input/datasets/ejunefong/training-data/Training+Testing_data"
PROCESSED_PATH = "/kaggle/working/processed_images" # ### MODIFIED: Faster I/O ###
DF_PATH = "/kaggle/input/datasets/ejunefong/training-data/Training+Testing_data_label.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 20

# --- 1. OFFLINE PREPROCESSING SCRIPT ---
# ### MODIFIED: This function runs once to save time during training ###
def run_offline_preprocessing(df, source_path, dest_path):
    os.makedirs(dest_path, exist_ok=True)
    print("Preprocessing images offline...")
    for img_name in tqdm(df["Image"]):
        save_file = os.path.join(dest_path, img_name)
        if os.path.exists(save_file): continue
        
        img = cv2.imread(os.path.join(source_path, img_name))
        if img is None: continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop, Resize, Ben Graham, CLAHE
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > 5
        if np.any(mask):
            img = img[np.ix_(mask.any(1), mask.any(0))]
        img = cv2.resize(img, (512, 512))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        cv2.imwrite(save_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# --- 2. DATASET & MODELS ---
# ### MODIFIED: Simplified Dataset (No heavy CV2 logic inside) ###
class RetinalDataset(Dataset):
    def __init__(self, df, image_path, transform=None):
        self.df = df
        self.image_path = image_path
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_path, row["Image"])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(row["Label"], dtype=torch.long) # ### MODIFIED: Explicit Long type ###
        
        if self.transform:
            image = self.transform(image)
        return image, label

train_trans = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(359),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_model(num_classes=5):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), # ### MODIFIED: Reduced slightly for better learning ###
        nn.Linear(in_features, num_classes)
    )
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate Quadratic Weighted Kappa
    epoch_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    # Keep F1 as a secondary metric
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Returning Kappa as the primary metric for the training loop
    return running_loss / total, correct / total, epoch_kappa, epoch_f1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    val_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Return everything: Loss, Accuracy, Kappa (Primary), and F1 (Secondary)
    return running_loss / total, correct / total, val_kappa, val_f1

def evaluate_model_performance(y_true, y_pred, train_losses=None, val_losses=None, title="Model Performance"):
    """
    Generic function to display metrics, confusion matrix, and optional loss plots.
    Works for individual folds or the final ensemble.
    """
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    # --- 1. Layout Setup ---
    # If we have losses, show 2 columns (Loss + Matrix). If not (Ensemble), show 1 column.
    has_history = train_losses is not None and val_losses is not None
    fig_cols = 2 if has_history else 1
    plt.figure(figsize=(8 * fig_cols, 6))
    
    # --- 2. Plot Loss Curves (Optional) ---
    if has_history:
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train', marker='o', color='#1f77b4')
        plt.plot(val_losses, label='Val', marker='o', color='#d62728')
        plt.title(f'{title} - Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # --- 3. Plot Confusion Matrix ---
    plt.subplot(1, fig_cols, fig_cols)
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (actual class) to see recall percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title(f'{title} - Confusion Matrix')
    plt.ylabel('Actual Stage')
    plt.xlabel('Predicted Stage')
    
    plt.tight_layout()
    plt.show()

    # --- 4. Calculate & Print Metrics ---
    metrics = {
        "Kappa (Quadratic)": cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average='weighted'),
        "Precision": precision_score(y_true, y_pred, average='weighted'),
        "Recall": recall_score(y_true, y_pred, average='weighted')
    }
    
    print(f"\n" + "="*30)
    print(f" {title.upper()} REPORT")
    print("="*30)
    for name, value in metrics.items():
        print(f"{name:20}: {value:.4f}")
    print("="*30 + "\n")

# Load and Split
df = pd.read_csv(DF_PATH)
run_offline_preprocessing(df, RAW_IMAGE_PATH, PROCESSED_PATH)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=42)
train_df = train_df.reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_df["Fold"] = -1
for fold, (t_idx, v_idx) in enumerate(skf.split(train_df, train_df["Label"])):
    train_df.loc[v_idx, 'Fold'] = fold

# K-Fold Loop
fold_results = []

for fold in range(5):
    print(f"\nSTARTING FOLD {fold+1}")
    f_train = train_df[train_df['Fold'] != fold].reset_index(drop=True)
    f_val = train_df[train_df['Fold'] == fold].reset_index(drop=True)

    train_loader = DataLoader(RetinalDataset(f_train, PROCESSED_PATH, train_trans), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(RetinalDataset(f_val, PROCESSED_PATH, val_trans), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = get_model(NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    
    # Scheduler: mode='max' because higher Kappa is better
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2)

    # --- Replicable Class-Balanced Weights ---
    class_counts = f_train['Label'].value_counts().sort_index().values
    total_samples = len(f_train)
    beta = 0.997  # Updated to be more aggressive for Severe/Proliferative classes
    
    weights = (1.0 - beta) / (1.0 - np.power(beta, class_counts))
    weights = weights / weights.mean()  # Normalize for gradient stability
    weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.00)

    patience = 3             # Number of epochs to wait after loss starts rising
    patience_counter = 0
    best_val_loss = float('inf')
    
    fold_train_losses = [] 
    fold_val_losses = []
    best_kappa = -1.0  # Initialized low for maximization
    
    for epoch in range(NUM_EPOCHS):
        # Unpacking with Kappa as primary return from train_one_epoch
        t_loss, t_acc, t_kappa, t_f1 = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_acc, v_kappa, v_f1 = validate(model, val_loader, criterion, DEVICE)

        fold_train_losses.append(t_loss)
        fold_val_losses.append(v_loss)
        
        # Step scheduler based on Validation Kappa
        scheduler.step(v_kappa)
        
        print(f"Epoch {epoch+1:02d} | Val Kappa: {v_kappa:.4f} | Val F1: {v_f1:.4f} | Val Acc: {v_acc:.4f}")

        # Save the best model based on Kappa score
        if v_kappa > best_kappa:
            best_kappa = v_kappa
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
            print(f"--> Saved New Best Kappa: {best_kappa:.4f}")
    
        # 4. EARLY STOPPING LOGIC (Monitoring Loss)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0  # Reset because we found a more generalized state
        else:
            patience_counter += 1
            print(f"--- Patience: {patience_counter}/{patience} (Loss did not improve) ---")
    
        if patience_counter >= patience:
            print(f"EARLY STOPPING triggered at epoch {epoch+1}. Preventing overfitting.")
            break
    
    # --- Evaluation Block for this Fold ---
    print(f"\nEvaluating Best Model for Fold {fold+1}...")
    
    model.load_state_dict(torch.load(f"best_model_fold{fold}.pth"))
    model.eval()
    
    y_true_fold = []
    y_pred_fold = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            
            y_true_fold.extend(labels.cpu().numpy())
            y_pred_fold.extend(preds.cpu().numpy())
    
    evaluate_model_performance(
        y_true=np.array(y_true_fold), 
        y_pred=np.array(y_pred_fold), 
        train_losses=fold_train_losses, 
        val_losses=fold_val_losses, 
        title=f"Fold {fold+1}"
    )
    
    fold_results.append(best_kappa)

print(f"\nAll Folds Complete. Mean Kappa: {np.mean(fold_results):.4f}")

def evaluate_ensemble(models, test_loader, device):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Ensemble"):
            inputs = inputs.to(device)
            # Accumulate predictions from all 5 models
            ensemble_logits = torch.zeros((inputs.size(0), 5)).to(device)
            
            for model in models:
                model.eval()
                outputs = model(inputs)
                ensemble_logits += torch.softmax(outputs, dim=1)
            
            _, predicted = ensemble_logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    
    print(f"\n{'='*30}")
    print(f"FINAL ENSEMBLE TEST RESULTS")
    print(f"{'='*30}")
    print(f"Test Kappa: {kappa:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    return all_labels, all_preds

# Load models and run
best_models = []
for i in range(5):
    m = get_model(num_classes=5).to(DEVICE)
    m.load_state_dict(torch.load(f"best_model_fold{i}.pth"))
    best_models.append(m)

test_dataset = RetinalDataset(test_df, PROCESSED_PATH, transform=val_trans)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

y_true, y_pred = evaluate_ensemble(best_models, test_loader, DEVICE)

evaluate_model_performance(y_true, y_pred)
