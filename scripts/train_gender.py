import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.diagonnet import DiagonNet
from src.utils.image_processing import preprocess_image
from config.settings import (
    GRID_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    HIDDEN_LAYERS,
    GENDER_DATA_DIR as DATA_DIR,
    GENDER_MODEL_PATH as MODEL_PATH,
    EPOCHS_GENDER as EPOCHS
)

def shift_image(img, dx, dy):
    """Shifts the image by dx, dy, filling the background with black."""
    w, h = img.size
    shifted = Image.new("L", (w, h), "black")
    shifted.paste(img, (dx, dy))
    return shifted

def augment_image(img):
    """Generates 9 versions of the input image (rotations and translations)."""
    augmented = []
    
    # 1. Original (Centered)
    augmented.append(img)
    
    # 2. Rotations (-12, -6, 6, 12 degrees)
    augmented.append(img.rotate(-12))
    augmented.append(img.rotate(-6))
    augmented.append(img.rotate(6))
    augmented.append(img.rotate(12))
    
    # 3. Shifts (up, down, left, right by 4 pixels)
    augmented.append(shift_image(img, -4, 0))
    augmented.append(shift_image(img, 4, 0))
    augmented.append(shift_image(img, 0, -4))
    augmented.append(shift_image(img, 0, 4))
    
    return augmented

def load_dataset():
    # Classes: 0 for man, 1 for woman
    categories = {"man": 0, "woman": 1}
    images = []
    labels = []
    
    for category_name, label_id in categories.items():
        cat_dir = os.path.join(DATA_DIR, category_name)
        if not os.path.exists(cat_dir):
            print(f"Error: Category directory '{cat_dir}' not found.")
            return None, None
            
        search_path = os.path.join(cat_dir, "*")
        img_paths = glob.glob(search_path)
        print(f"Found {len(img_paths)} images for class '{category_name}'.")
        
        for path in img_paths:
            # Skip hidden files like .DS_Store
            if os.path.basename(path).startswith("."):
                continue
            try:
                # Load in grayscale
                img = Image.open(path).convert("L")
                
                # Apply standard crop/center preprocess
                preprocessed = preprocess_image(img, target_size=GRID_SIZE)
                
                images.append(preprocessed)
                labels.append(label_id)
            except Exception as e:
                pass
                
    return images, np.array(labels)

def flatten_and_normalize(image_list):
    X = []
    for img in image_list:
        img_array = np.array(img, dtype=np.float32) / 255.0
        X.append(img_array.flatten())
    return np.array(X)

def train_model(model, train_loader, val_loader, device, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
        val_acc = correct / total
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")
        
    return model

def main():
    print("=" * 70)
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
    # 2. Load and Preprocess Images
    print("Loading and preprocessing dataset...")
    images, labels = load_dataset()
    if not images:
        print("Failed to load dataset.")
        return
        
    print(f"Successfully loaded {len(images)} face images.")
    
    # 3. Train/Val Split
    X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Dataset split: {len(X_train_orig)} training, {len(X_val_orig)} validation.")
    
    # 4. Data Augmentation
    print("Applying Data Augmentation (Rotations & Shifts)...")
    X_train_aug_imgs = []
    y_train_aug = []
    for img, label in zip(X_train_orig, y_train_orig):
        aug_versions = augment_image(img)
        X_train_aug_imgs.extend(aug_versions)
        y_train_aug.extend([label] * len(aug_versions))
        
    y_train_aug = np.array(y_train_aug)
    
    X_train = flatten_and_normalize(X_train_aug_imgs)
    X_val = flatten_and_normalize(X_val_orig)
    y_val = y_val_orig
    
    print(f"Augmented training size: {len(X_train)} samples.")
    
    # 5. Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_aug, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # 6. Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 7. Initialize DiagonNet Model for Binary Classification (man / woman)
    model = DiagonNet(
        input_size=GRID_SIZE,
        hidden_dims=[256, 128, 64],
        num_classes=2 # 2 outputs (man, woman)
    ).to(device)
    
    # 8. Train Model
    print(f"\nTraining DiagonNet on GPU...")
    model = train_model(model, train_loader, val_loader, device, EPOCHS)
    
    # 9. Retrain on Full Dataset to Maximize Performance
    print("\nRetraining on full augmented dataset...")
    X_full_aug_imgs = []
    y_full_aug = []
    for img, label in zip(images, labels):
        aug_versions = augment_image(img)
        X_full_aug_imgs.extend(aug_versions)
        y_full_aug.extend([label] * len(aug_versions))
        
    y_full_aug = np.array(y_full_aug)
    X_full = flatten_and_normalize(X_full_aug_imgs)
    
    X_full_tensor = torch.tensor(X_full, dtype=torch.float32)
    y_full_tensor = torch.tensor(y_full_aug, dtype=torch.long)
    
    full_dataset = TensorDataset(X_full_tensor, y_full_tensor)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    final_model = DiagonNet(
        input_size=GRID_SIZE,
        hidden_dims=[256, 128, 64],
        num_classes=2
    ).to(device)
    
    final_model = train_model(final_model, full_loader, val_loader, device, EPOCHS)
    
    # 10. Save Model Weights
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(final_model.state_dict(), MODEL_PATH)
    print(f"\nSuccess! DiagonNet Gender Model saved to '{MODEL_PATH}'")
    print("=" * 70)

if __name__ == "__main__":
    main()
