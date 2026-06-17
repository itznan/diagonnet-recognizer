import numpy as np
from PIL import Image
import torch
import torch.nn as nn

def preprocess_image(img, target_size=100, padding=12):
    """
    Crops the drawn digit to its bounding box, pads it to a square aspect ratio,
    and resizes it to target_size x target_size.
    This ensures the model is invariant to where the user draws and how large they draw.
    """
    img = img.convert("L")
    bbox = img.getbbox()
    if bbox is None:
        # Return a blank image if canvas is empty
        return Image.new("L", (target_size, target_size), "black")
        
    # Crop to bounding box of the drawing
    cropped = img.crop(bbox)
    
    w, h = cropped.size
    max_dim = max(w, h)
    
    # Create a black square image with padding
    square_size = max_dim + padding * 2
    square_img = Image.new("L", (square_size, square_size), "black")
    
    # Center the cropped image
    paste_x = padding + (max_dim - w) // 2
    paste_y = padding + (max_dim - h) // 2
    square_img.paste(cropped, (paste_x, paste_y))
    
    # Resize to target 100x100
    return square_img.resize((target_size, target_size), Image.Resampling.LANCZOS)

class DiagonNet(nn.Module):
    """
    DiagonNet Custom Architecture with Knight's Move Extension:
    Calculates pixel-level spatial relationships including:
      - 4 Diagonal relationships (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
      - 8 Chess Knight (L-shape) moves (Up 2 Right 1, Up 2 Left 1, etc.)
    Concatenates them into a 13-channel feature representation for classification.
    """
    def __init__(self, input_size=100, hidden_dims=[128, 64], num_classes=10):
        super(DiagonNet, self).__init__()
        self.input_size = input_size
        
        # 13 Channels:
        # [1 Original Image] + [4 Diagonal Difference Maps] + [8 Knight's Move Difference Maps]
        self.channels = 1 + 4 + 8
        self.flattened_dim = self.channels * input_size * input_size
        
        # Dense classification layers
        layers = []
        prev_dim = self.flattened_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        # Ensure input is reshaped to (batch_size, 1, input_size, input_size)
        if len(x.shape) == 2:
            x = x.view(-1, 1, self.input_size, self.input_size)
            
        # -------------------------------------------------------------
        # Part 1: Standard Diagonals (Shift by 1 pixel diagonally)
        # -------------------------------------------------------------
        tl = torch.zeros_like(x)
        tl[:, :, 1:, 1:] = x[:, :, :-1, :-1]  # Top-Left
        
        tr = torch.zeros_like(x)
        tr[:, :, 1:, :-1] = x[:, :, :-1, 1:]  # Top-Right
        
        bl = torch.zeros_like(x)
        bl[:, :, :-1, 1:] = x[:, :, 1:, :-1]  # Bottom-Left
        
        br = torch.zeros_like(x)
        br[:, :, :-1, :-1] = x[:, :, 1:, 1:]  # Bottom-Right
        
        diff_tl = x - tl
        diff_tr = x - tr
        diff_bl = x - bl
        diff_br = x - br
        
        # -------------------------------------------------------------
        # Part 2: Chess Knight's Moves (L-shape shifts: 2 steps in one dir, 1 in the other)
        # -------------------------------------------------------------
        # k1: Up 2, Right 1
        k1 = torch.zeros_like(x)
        k1[:, :, :-2, 1:] = x[:, :, 2:, :-1]
        
        # k2: Up 2, Left 1
        k2 = torch.zeros_like(x)
        k2[:, :, :-2, :-1] = x[:, :, 2:, 1:]
        
        # k3: Down 2, Right 1
        k3 = torch.zeros_like(x)
        k3[:, :, 2:, 1:] = x[:, :, :-2, :-1]
        
        # k4: Down 2, Left 1
        k4 = torch.zeros_like(x)
        k4[:, :, 2:, :-1] = x[:, :, :-2, 1:]
        
        # k5: Right 2, Up 1
        k5 = torch.zeros_like(x)
        k5[:, :, :-1, 2:] = x[:, :, 1:, :-2]
        
        # k6: Right 2, Down 1
        k6 = torch.zeros_like(x)
        k6[:, :, 1:, 2:] = x[:, :, :-1, :-2]
        
        # k7: Left 2, Up 1
        k7 = torch.zeros_like(x)
        k7[:, :, :-1, :-2] = x[:, :, 1:, 2:]
        
        # k8: Left 2, Down 1
        k8 = torch.zeros_like(x)
        k8[:, :, 1:, :-2] = x[:, :, :-1, 2:]
        
        # Compute Knight's Move difference maps
        diff_k1 = x - k1
        diff_k2 = x - k2
        diff_k3 = x - k3
        diff_k4 = x - k4
        diff_k5 = x - k5
        diff_k6 = x - k6
        diff_k7 = x - k7
        diff_k8 = x - k8
        
        # -------------------------------------------------------------
        # Part 3: Concatenation and Classification
        # -------------------------------------------------------------
        # Concatenate along the channel dimension (dim=1)
        # Out shape: (batch_size, 13, input_size, input_size)
        features = torch.cat([
            x, 
            diff_tl, diff_tr, diff_bl, diff_br,
            diff_k1, diff_k2, diff_k3, diff_k4, diff_k5, diff_k6, diff_k7, diff_k8
        ], dim=1)
        
        # Flatten features for classification
        flat_features = features.view(-1, self.flattened_dim)
        
        return self.classifier(flat_features)
