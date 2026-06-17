import os
import sys
import torch
import numpy as np
from PIL import Image

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.diagonnet import DiagonNet
from src.utils.image_processing import preprocess_image
from config.settings import (
    GRID_SIZE,
    HIDDEN_LAYERS,
    GENDER_MODEL_PATH as MODEL_PATH
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_gender.py <path_to_image>")
        return
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: Image '{img_path}' not found.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model (binary classifier: 2 outputs)
    model = DiagonNet(input_size=GRID_SIZE, hidden_dims=HIDDEN_LAYERS, num_classes=2)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    try:
        # Load image in grayscale
        img = Image.open(img_path).convert("L")
        
        # Crop/center preprocess
        preprocessed = preprocess_image(img, target_size=GRID_SIZE)
        
        # Normalize and flatten (100x100 -> 10000)
        img_array = (np.array(preprocessed, dtype=np.float32) / 255.0).flatten()
        
        # Convert to tensor
        tensor_img = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = int(torch.argmax(outputs).item())
            
        classes = ["Male (Man)", "Female (Woman)"]
        print(f"\n=========================================")
        print(f"  DiagonNet Gender Recognition Result")
        print(f"=========================================")
        print(f"Image: {os.path.basename(img_path)}")
        print(f"Prediction: {classes[prediction]}")
        print(f"Confidence:")
        print(f"  - Male:   {probabilities[0]*100:.2f}%")
        print(f"  - Female: {probabilities[1]*100:.2f}%")
        print(f"=========================================\n")
        
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
