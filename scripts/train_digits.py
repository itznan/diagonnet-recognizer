import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.models.diagonnet import DiagonNet
from src.utils.image_processing import preprocess_image
from config.settings import (
    GRID_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    HIDDEN_LAYERS,
    DIGIT_DATA_DIR as DATA_DIR,
    DIGIT_MODEL_PATH as MODEL_PATH,
    EPOCHS_DIGITS as EPOCHS
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
    
    # 3. Shifts (up, down, left, right by 5 pixels)
    augmented.append(shift_image(img, -5, 0))
    augmented.append(shift_image(img, 5, 0))
    augmented.append(shift_image(img, 0, -5))
    augmented.append(shift_image(img, 0, 5))
    
    return augmented

def load_and_preprocess_dataset():
    search_path = os.path.join(DATA_DIR, "*", "*.png")
    image_paths = glob.glob(search_path)
    
    if not image_paths:
        print(f"Error: No training images found in '{DATA_DIR}' folder.")
        print("Please run 'python collect.py' first to collect digit samples!")
        return None, None

    print(f"Found {len(image_paths)} original images. Preprocessing and centering...")
    
    original_images = []
    labels = []
    
    for img_path in image_paths:
        try:
            # Load original image
            img = Image.open(img_path).convert("L")
            
            # Crop, center, and square to 100x100
            preprocessed_img = preprocess_image(img, target_size=GRID_SIZE)
            
            # Extract label (digit folder name)
            parent_dir = os.path.basename(os.path.dirname(img_path))
            digit = int(parent_dir)
            
            original_images.append(preprocessed_img)
            labels.append(digit)
        except Exception as e:
            print(f"Warning: Failed to load/preprocess image {img_path}: {e}")
            
    return original_images, np.array(labels)

def flatten_and_normalize(image_list):
    """Converts a list of PIL images into a normalized flat numpy array."""
    X = []
    for img in image_list:
        img_array = np.array(img, dtype=np.float32) / 255.0
        X.append(img_array.flatten())
    return np.array(X)

def train_model(model, train_loader, val_loader, device, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_acc = 0.0
    
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
        
        # Validation evaluation
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
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:03d}/{epochs:03d} | Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")
            
    return model

def main():
    print("=" * 70)
    print("           PyTorch CUDA-Powered Digit Recognition Model Training")
    print("=" * 70)
    
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device.type.upper()}")
    if device.type == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 2. Load dataset
    images, labels = load_and_preprocess_dataset()
    if images is None or len(images) == 0:
        return
        
    unique_classes = np.unique(labels)
    print(f"Loaded dataset: {len(images)} samples across classes: {unique_classes}")
    
    # Check classes
    if len(unique_classes) < 2:
        print("\nError: You need at least 2 different digits to train a classifier.")
        return

    # 3. Train / Test Split on ORIGINAL images
    # We test on clean validation images, and only augment the training split
    X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nSplit dataset: {len(X_train_orig)} training samples, {len(X_val_orig)} validation/test samples.")
    
    # 4. Apply Data Augmentation only to the training split
    print("Applying Data Augmentation (Rotations & Shifts) to training set...")
    X_train_aug_imgs = []
    y_train_aug = []
    
    for img, label in zip(X_train_orig, y_train_orig):
        augmented_versions = augment_image(img)
        X_train_aug_imgs.extend(augmented_versions)
        y_train_aug.extend([label] * len(augmented_versions))
        
    y_train_aug = np.array(y_train_aug)
    
    # Flatten and normalize images
    X_train = flatten_and_normalize(X_train_aug_imgs)
    X_val = flatten_and_normalize(X_val_orig)
    y_val = y_val_orig
    
    print(f"Augmented training set: {len(X_train)} samples (9x multiplier).")
    
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
    
    # 7. Define PyTorch Neural Network (Custom DiagonNet)
    model = DiagonNet(
        input_size=GRID_SIZE, 
        hidden_dims=HIDDEN_LAYERS, 
        num_classes=10  # 10 outputs so digit x directly maps to output index x
    ).to(device)
    
    print(f"\nTraining Neural Network on {device.type.upper()}...")
    model = train_model(model, train_loader, val_loader, device, EPOCHS)
    
    # 8. Retrain on the ENTIRE dataset (All original + augmented versions)
    print("\nApplying Data Augmentation to full dataset and retraining winner...")
    X_full_aug_imgs = []
    y_full_aug = []
    
    for img, label in zip(images, labels):
        augmented_versions = augment_image(img)
        X_full_aug_imgs.extend(augmented_versions)
        y_full_aug.extend([label] * len(augmented_versions))
        
    X_full = flatten_and_normalize(X_full_aug_imgs)
    y_full = np.array(y_full_aug)
    
    X_full_tensor = torch.tensor(X_full, dtype=torch.float32)
    y_full_tensor = torch.tensor(y_full, dtype=torch.long)
    
    full_dataset = TensorDataset(X_full_tensor, y_full_tensor)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Full augmented training set size: {len(X_full)} samples.")
    
    final_model = DiagonNet(
        input_size=GRID_SIZE, 
        hidden_dims=HIDDEN_LAYERS, 
        num_classes=10
    ).to(device)
    
    final_model = train_model(final_model, full_loader, val_loader, device, EPOCHS)
    
    # 9. Save the PyTorch Model weights
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(final_model.state_dict(), MODEL_PATH)
    print(f"\nSuccess! PyTorch Model trained on GPU and weights saved to '{MODEL_PATH}'")
    print("Next step: Run 'python scripts/run_digit_app.py' to test the model in real time!")
    print("=" * 70)

if __name__ == "__main__":
    main()
