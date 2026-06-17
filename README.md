# 🌌 DiagonNet Recognizer Project (Hybrid CNN-DiagonNet)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-orange.svg)](https://pytorch.org/)
[![CUDA Enabled](https://img.shields.io/badge/CUDA-RTX%203060%20Ti-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License-MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

An end-to-end framework implementing the custom **DiagonNet** neural network architecture. The project supports:
1. **Interactive Digit Recognition**: Draw digits on a live Tkinter canvas and run real-time inference on your GPU.
2. **Face Gender Classification**: Predict male vs. female faces using a custom face dataset, comparing performance directly against standard baselines (SimpleCNN, SimpleMLP) and YOLOv8-Classifier.

---

## 📁 Repository Structure

```
diagonnet-recognizer/
├── config/
│   └── settings.py          # Central configurations & hyperparameters
├── src/                     # Core source package
│   ├── __init__.py
│   ├── models/              # Neural network architectures
│   │   ├── __init__.py
│   │   └── diagonnet.py     # Custom Hybrid CNN-DiagonNet
│   ├── utils/               # Code utilities
│   │   ├── __init__.py
│   │   └── image_processing.py # Crop, padding & image preprocessing
│   └── pipelines/           # Placeholder for advanced data loaders
├── scripts/                 # Standalone execution scripts
│   ├── collect_digits.py    # Drawing canvas for digit dataset collection
│   ├── train_digits.py      # Digit classifier training script (DiagonNet)
│   ├── run_digit_app.py     # Live GUI canvas for real-time digit prediction
│   ├── train_gender.py      # Face gender classifier training script (DiagonNet)
│   ├── predict_gender.py    # Command-line prediction tool for face images
│   └── run_comparison.py    # Baseline (CNN, MLP, YOLOv8) comparison test
├── assets/                  # Media & visualization charts
│   └── comparison_chart.png # Benchmark performance comparison curve
├── weights/                 # Ignored local folder for model weights
│   ├── diagonnet_digits.pth
│   └── diagonnet_gender.pth
├── requirements.txt         # Core dependencies
└── README.md
```

---

## 📊 Performance Benchmarks (Face Gender Classifier)

Models trained on **2,307 face images** using an NVIDIA GeForce RTX 3060 Ti GPU over 20 epochs.

| Model Name | Final Validation Accuracy | Trainable Parameters | Training Time (20 Epochs) |
| :--- | :---: | :---: | :---: |
| **YOLOv8-Classifier** (Pre-trained) | **98.27%** | 1,437,442 | 167.28 seconds |
| **DiagonNet** (Hybrid Custom) | **97.62%** | **1,106,050** | **11.64 seconds** |
| **SimpleCNN** (Baseline CNN) | **88.31%** | 1,203,330 | 3.86 seconds |
| **SimpleMLP** (Baseline MLP) | **82.25%** | 2,601,538 | 1.66 seconds |

> [!IMPORTANT]
> **🏆 The DiagonNet Advantage:**
> * **Near-SOTA Accuracy from Scratch**: Achieves a massive **97.62% validation accuracy**—on par with YOLOv8 (**98.27%**)—without utilizing any external pre-trained weights (unlike YOLOv8 which is pre-trained on ImageNet).
> * **14.3x Speedup in Training Time**: Trains in just **11.64 seconds** on GPU, compared to YOLOv8's **167.28 seconds** (for the full 20-epoch run).
> * **Outperforms Baselines by up to 15%**: Beats standard convolutional baselines (SimpleCNN at **88.31%**) by **9.31%** and MLPs (SimpleMLP at **82.25%**) by **15.37%** validation accuracy, leveraging the custom shift-difference maps as a strong inductive spatial bias.
> * **Ultra-Lean Resource Footprint**: Slashing parameter size to **1.1 Million** (a **96.7% reduction** from the previous MLP implementation) ensures the model fits on resource-constrained edge devices and runs predictions in milliseconds.

### 📈 Accuracy Trajectory

![Validation Accuracy Comparison](assets/comparison_chart.png)


---

## 🚀 Getting Started

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### 1. Digit Recognition Workflow

#### Step A: Collect Your Drawings
Run the dataset collection tool to draw 30 attempts for each digit (1-9):
```bash
python scripts/collect_digits.py
```
* **Draw**: Drag to write the digit prompted at the top.
* **Save & Next**: Press **Spacebar** or **Enter** to save the drawing.
* **Clear**: Press **C** or **Escape** to wipe the canvas.
* **Undo / Back**: Step back one index and delete the previous save.

#### Step B: Train the Digit Classifier
Train the model on your custom drawings:
```bash
python scripts/train_digits.py
```
* Preprocesses drawings by cropping to their bounding box and centering them.
* Appends rotations and translations to augment the data 9x (generating **2,430 samples**).
* Trains on your GPU and dumps the weights into `weights/diagonnet_digits.pth`.

#### Step C: Run Real-Time GUI Predictions
Test the trained model in real time:
```bash
python scripts/run_digit_app.py
```
* Draw on the canvas. The model crops, pads, and feeds the image to your GPU for live inference as you draw!
* Highlights class probabilities via color-coded progress bars in the side panel.

---

### 2. Face Gender Classifier Workflow

#### Step A: Train the Gender Model
Train the custom DiagonNet binary gender classifier on face images:
```bash
python scripts/train_gender.py
```
* Auto-splits data, applies rotations and shifts, trains for 25 epochs on CUDA, and dumps weights to `weights/diagonnet_gender.pth`.

#### Step B: Run Predictions on Custom Files
Test the classifier on any custom image file:
```bash
python scripts/predict_gender.py <path_to_image>
```
*Example:*
```bash
python scripts/predict_gender.py "E:\Download\download.jpg"
```

#### Step C: Compare Baselines
Compare DiagonNet against SimpleCNN, SimpleMLP, and YOLOv8-Cls:
```bash
python scripts/run_comparison.py
```
* Restructures dataset, trains all four networks from scratch, prints a comparison table, and saves the comparison chart to `assets/comparison_chart.png`.

---

## 🧠 Technical Overview: The DiagonNet Pipeline

1. **Alignment & Centering (`src/utils/image_processing.py`)**:
   Achieves scale and position invariance by cropping digits to their bounding box (`Image.getbbox()`), padding symmetrically with a black boundary to restore a square aspect ratio, and resizing to $100 \times 100$.
2. **Double-Resizing Consistency**:
   To ensure the testing drawings match the exact image degradation (anti-aliasing) of the training dataset, live drawings are first scaled to $100 \times 100$ and then centered.
3. **Custom DiagonNet Spatial Shifts (`src/models/diagonnet.py`)**:
   The custom shift layer computes 13 feature maps:
   * **1 original channel**
   * **4 diagonal maps** (shifted by 1 pixel diagonally: Top-Left, Top-Right, Bottom-Left, Bottom-Right)
   * **8 chess knight-move maps** (L-shape shifts, e.g., Up 2, Right 1)
   The differences between the original image and shifted images are concatenated to form a 13-channel feature map.
4. **CNN Pyramidal Processing**:
   The 13-channel map is processed through 4 convolutional layers (filter channels: $13 \to 32 \to 64 \to 128 \to 128$) with `BatchNorm2d`, `Dropout2d`, and `MaxPool2d` to capture spatial hierarchies and enforce translation invariance. An `AdaptiveAvgPool2d((5, 5))` aggregates the outputs into a compact 3,200-dimensional vector before classification.
