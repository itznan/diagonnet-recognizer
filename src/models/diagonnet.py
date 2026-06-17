import torch
import torch.nn as nn

class DiagonNet(nn.Module):
    """
    DiagonNet Custom Architecture with Knight's Move Extension & CNN Features:
    Calculates pixel-level spatial relationships including:
      - 4 Diagonal relationships (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
      - 8 Chess Knight (L-shape) moves (Up 2 Right 1, Up 2 Left 1, etc.)
    Concatenates them into a 13-channel feature representation.
    Then, feeds this 13-channel representation through 2D Convolutions, Batch Normalization,
    Max Pooling, and Dropout to learn spatial hierarchy and transition invariance,
    slashing the parameters down to under 2 Million.
    """
    def __init__(self, input_size=100, hidden_dims=[128, 64], num_classes=10):
        super(DiagonNet, self).__init__()
        self.input_size = input_size
        
        # 13 Channels:
        # [1 Original Image] + [4 Diagonal Difference Maps] + [8 Knight's Move Difference Maps]
        self.channels = 1 + 4 + 8
        
        # Convolutional layers to process custom DiagonNet maps
        self.conv_layers = nn.Sequential(
            # Layer 1: Input has 13 channels
            nn.Conv2d(self.channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 100x100 -> 50x50
            nn.Dropout2d(0.1),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 50x50 -> 25x25
            nn.Dropout2d(0.15),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 25x25 -> 12x12
            nn.Dropout2d(0.2),
            
            # Layer 4: Additional depth & Adaptive Pooling for input-size invariance
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5))             # 12x12 -> 5x5
        )
        
        # Compute flattened dimension dynamically to support flexible input sizes
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size, input_size)
            dummy_features = self._custom_features(dummy)
            dummy_conv = self.conv_layers(dummy_features)
            self.flattened_dim = dummy_conv.numel()
        
        # Dense classification layers
        layers = []
        prev_dim = self.flattened_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
    def _custom_features(self, x):
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
        # Part 2: Chess Knight's Moves (L-shape shifts)
        # -------------------------------------------------------------
        k1 = torch.zeros_like(x)
        k1[:, :, :-2, 1:] = x[:, :, 2:, :-1]
        
        k2 = torch.zeros_like(x)
        k2[:, :, :-2, :-1] = x[:, :, 2:, 1:]
        
        k3 = torch.zeros_like(x)
        k3[:, :, 2:, 1:] = x[:, :, :-2, :-1]
        
        k4 = torch.zeros_like(x)
        k4[:, :, 2:, :-1] = x[:, :, :-2, 1:]
        
        k5 = torch.zeros_like(x)
        k5[:, :, :-1, 2:] = x[:, :, 1:, :-2]
        
        k6 = torch.zeros_like(x)
        k6[:, :, 1:, 2:] = x[:, :, :-1, :-2]
        
        k7 = torch.zeros_like(x)
        k7[:, :, :-1, :-2] = x[:, :, 1:, 2:]
        
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
        
        # Concatenate along the channel dimension (dim=1)
        features = torch.cat([
            x, 
            diff_tl, diff_tr, diff_bl, diff_br,
            diff_k1, diff_k2, diff_k3, diff_k4, diff_k5, diff_k6, diff_k7, diff_k8
        ], dim=1)
        
        return features

    def forward(self, x):
        features = self._custom_features(x)
        conv_features = self.conv_layers(features)
        flat_features = conv_features.view(-1, self.flattened_dim)
        return self.classifier(flat_features)
