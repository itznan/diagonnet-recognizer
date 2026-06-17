import os
import sys
import time
import shutil
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
    YOLO_GENDER_RUN_DIR as YOLO_PROJECT_DIR,
    COMPARISON_CHART_PATH as CHART_PATH
)
from ultralytics import YOLO

EPOCHS = 20  # Keep epochs at 20 for standard comparative runs

# 1. Define baseline models for comparison
class SimpleMLP(nn.Module):
    """Standard Multi-Layer Perceptron Baseline"""
    def __init__(self, input_dim=10000, hidden_dims=[256, 128, 64], num_classes=2):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class SimpleCNN(nn.Module):
    """Standard Convolutional Neural Network Baseline"""
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Out: 16 x 50 x 50
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Out: 32 x 25 x 25
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # Out: 64 x 12 x 12
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def load_dataset():
    categories = {"man": 0, "woman": 1}
    images = []
    labels = []
    
    for category_name, label_id in categories.items():
        cat_dir = os.path.join(DATA_DIR, category_name)
        if not os.path.exists(cat_dir):
            print(f"Error: Category directory '{cat_dir}' not found.")
            return None, None
            
        img_paths = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if not f.startswith(".")]
        print(f"Loading {len(img_paths)} images for class '{category_name}'...")
        
        for path in img_paths:
            try:
                img = Image.open(path).convert("L")
                preprocessed = preprocess_image(img, target_size=GRID_SIZE)
                images.append(preprocessed)
                labels.append(label_id)
            except Exception as e:
                pass
                
    return images, np.array(labels)

def get_image_tensors(image_list):
    """Returns unflattened tensors of shape (B, 1, 100, 100) normalized to [0,1]."""
    tensors = []
    for img in image_list:
        img_array = np.array(img, dtype=np.float32) / 255.0
        tensors.append(np.expand_dims(img_array, axis=0)) # (1, 100, 100)
    return torch.tensor(np.array(tensors), dtype=torch.float32)

def train_and_evaluate(model_name, model, train_loader, val_loader, device):
    print(f"\n--- Training {model_name} ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    val_acc_history = []
    start_time = time.time()
    
    for epoch in range(EPOCHS):
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
        val_acc_history.append(val_acc * 100)
        
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | Loss: {running_loss/len(train_loader.dataset):.4f} | Val Accuracy: {val_acc*100:.2f}%")
            
    total_time = time.time() - start_time
    print(f"Finished training {model_name} in {total_time:.2f} seconds.")
    return val_acc_history, total_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_yolo_dataset():
    """Restructures dataset into train/val folders for YOLO classification format."""
    yolo_dir = os.path.join(os.path.dirname(__file__), "yolo_data")
    train_man = os.path.join(yolo_dir, "train", "man")
    train_woman = os.path.join(yolo_dir, "train", "woman")
    val_man = os.path.join(yolo_dir, "val", "man")
    val_woman = os.path.join(yolo_dir, "val", "woman")
    
    # Reset directories
    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)
        
    os.makedirs(train_man, exist_ok=True)
    os.makedirs(train_woman, exist_ok=True)
    os.makedirs(val_man, exist_ok=True)
    os.makedirs(val_woman, exist_ok=True)
    
    man_src = os.path.join(DATA_DIR, "man")
    woman_src = os.path.join(DATA_DIR, "woman")
    
    man_files = [os.path.join(man_src, f) for f in os.listdir(man_src) if not f.startswith(".")]
    woman_files = [os.path.join(woman_src, f) for f in os.listdir(woman_src) if not f.startswith(".")]
    
    # Split
    t_man, v_man = train_test_split(man_files, test_size=0.2, random_state=42)
    t_woman, v_woman = train_test_split(woman_files, test_size=0.2, random_state=42)
    
    # Copy files
    for f in t_man: shutil.copy(f, train_man)
    for f in v_man: shutil.copy(f, val_man)
    for f in t_woman: shutil.copy(f, train_woman)
    for f in v_woman: shutil.copy(f, val_woman)
    
    return yolo_dir

def main():
    print("=" * 80)
    print("      DiagonNet vs. Standard Baselines & YOLOv8-Cls Comparison on Faces")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    
    # 1. Load standard dataset
    images, labels = load_dataset()
    if images is None or len(images) == 0:
        return
        
    # Split dataset (unaugmented for clean architectural comparison)
    X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Convert images to tensors (shape: B, 1, 100, 100)
    X_train_tensor = get_image_tensors(X_train_orig)
    y_train_tensor = torch.tensor(y_train_orig, dtype=torch.long)
    X_val_tensor = get_image_tensors(X_val_orig)
    y_val_tensor = torch.tensor(y_val_orig, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Initialize baselines & DiagonNet
    diagon_net = DiagonNet(input_size=GRID_SIZE, hidden_dims=[256, 128, 64], num_classes=2).to(device)
    simple_cnn = SimpleCNN(num_classes=2).to(device)
    simple_mlp = SimpleMLP(input_dim=GRID_SIZE*GRID_SIZE, hidden_dims=[256, 128, 64], num_classes=2).to(device)
    
    # Train standard models
    acc_diagon, time_diagon = train_and_evaluate("DiagonNet", diagon_net, train_loader, val_loader, device)
    acc_cnn, time_cnn = train_and_evaluate("SimpleCNN", simple_cnn, train_loader, val_loader, device)
    acc_mlp, time_mlp = train_and_evaluate("SimpleMLP", simple_mlp, train_loader, val_loader, device)
    
    # 3. Setup and train YOLOv8 Classification
    print("\n--- Preparing YOLOv8 Classification Dataset ---")
    yolo_data_dir = setup_yolo_dataset()
    
    print("\n--- Training YOLOv8-Classification ---")
    yolo_model = YOLO("yolov8n-cls.pt")  # Download pre-trained nano classifier
    
    # Count YOLO parameters
    params_yolo = sum(p.numel() for p in yolo_model.model.parameters() if p.requires_grad)
    
    # Train YOLO
    yolo_project = YOLO_PROJECT_DIR
    if os.path.exists(yolo_project):
        shutil.rmtree(yolo_project)
        
    start_yolo_time = time.time()
    yolo_model.train(
        data=yolo_data_dir,
        epochs=EPOCHS,
        imgsz=GRID_SIZE,
        device=0 if device.type == "cuda" else "cpu",
        project=yolo_project,
        name="gender_run",
        exist_ok=True,
        verbose=False
    )
    time_yolo = time.time() - start_yolo_time
    print(f"Finished training YOLOv8-Cls in {time_yolo:.2f} seconds.")
    
    # Read YOLO validation accuracy history from results.csv
    acc_yolo = []
    results_csv = os.path.join(yolo_project, "gender_run", "results.csv")
    if os.path.exists(results_csv):
        with open(results_csv, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
                # top1 accuracy is a float in [0, 1]
                val_acc = float(cleaned_row.get("metrics/accuracy_top1", 0.0)) * 100
                acc_yolo.append(val_acc)
    
    # Fallback if CSV reading fails
    if len(acc_yolo) < EPOCHS:
        acc_yolo = [0.0] * (EPOCHS - len(acc_yolo)) + acc_yolo
        if len(acc_yolo) > EPOCHS:
            acc_yolo = acc_yolo[:EPOCHS]
            
    # Clean up yolo duplicate dataset folder to save disk space
    if os.path.exists(yolo_data_dir):
        shutil.rmtree(yolo_data_dir)
        
    params_diagon = count_parameters(diagon_net)
    params_cnn = count_parameters(simple_cnn)
    params_mlp = count_parameters(simple_mlp)
    
    # Output comparison results
    print("\n" + "=" * 70)
    print("                        COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model Name':<18} | {'Val Accuracy':<13} | {'Parameters':<12} | {'Train Time':<10}")
    print("-" * 70)
    print(f"{'DiagonNet':<18} | {acc_diagon[-1]:.2f}%{'':<8} | {params_diagon:<12} | {time_diagon:.2f}s")
    print(f"{'SimpleCNN':<18} | {acc_cnn[-1]:.2f}%{'':<8} | {params_cnn:<12} | {time_cnn:.2f}s")
    print(f"{'SimpleMLP':<18} | {acc_mlp[-1]:.2f}%{'':<8} | {params_mlp:<12} | {time_mlp:.2f}s")
    print(f"{'YOLOv8-Classifier':<18} | {acc_yolo[-1]:.2f}%{'':<8} | {params_yolo:<12} | {time_yolo:.2f}s")
    print("=" * 70)
    
    # Plotting comparison graph
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    epochs_range = range(1, EPOCHS + 1)
    plt.plot(epochs_range, acc_diagon, label=f'DiagonNet ({acc_diagon[-1]:.2f}%)', color='#8A2BE2', linewidth=2, marker='o')
    plt.plot(epochs_range, acc_cnn, label=f'SimpleCNN ({acc_cnn[-1]:.2f}%)', color='#00FF7F', linewidth=2, marker='s')
    plt.plot(epochs_range, acc_mlp, label=f'SimpleMLP ({acc_mlp[-1]:.2f}%)', color='#FF4500', linewidth=2, marker='^')
    plt.plot(epochs_range, acc_yolo, label=f'YOLOv8-Classifier ({acc_yolo[-1]:.2f}%)', color='#FFD700', linewidth=2, marker='d')
    
    plt.title('Face Gender Classifier Model Comparison (with YOLOv8)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Validation Accuracy (%)', fontsize=11)
    plt.xticks(epochs_range)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)
    
    chart_path = CHART_PATH
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved comparison chart to '{chart_path}'")
    print("=" * 80)

if __name__ == "__main__":
    main()
