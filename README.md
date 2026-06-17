# Interactive Digit Recognition Project (DiagonNet Grid)

An end-to-end Python workflow to collect hand-drawn digits on a 100x100 grid, train the custom **DiagonNet** neural network classifier using PyTorch with CUDA GPU acceleration, and run real-time inference on a canvas.

---

## 📁 File Structure

* **`collect.py`**: Tkinter GUI to draw digits (1 to 9, 30 times each). Saves processed 100x100 images to the `data/` directory.
* **`utils.py`**: Shared helper utility containing the `preprocess_image` function (bounding-box cropping, square aspect-ratio padding, centering) and the custom PyTorch `DiagonNet` neural network class definition.
* **`train.py`**: Model training script. Loads dataset, applies centering and 9x data augmentation, trains the custom PyTorch `DiagonNet` neural network on your CUDA GPU, and saves the trained weights to `model.pth`.
* **`use.py`**: Real-time deployment GUI. Includes the drawing canvas, live PyTorch-based GPU predictions on mouse movements via `DiagonNet`, and color-coded confidence score bar rows for each class.
* **`data/`**: Subfolders `1/` to `9/` containing 100x100 PNG grayscale images.
* **`model.pth`**: Saved `DiagonNet` model weights state dictionary.

---

## ⚙️ Prerequisites

The project requires the following Python libraries (pre-configured with CUDA compatibility in this environment):
```bash
pip install numpy pillow torch scikit-learn
```

---

## 🚀 How to Run the Project

Follow these steps in your terminal inside the workspace directory (`E:\NAN\Ai\1`):

### Step 1: Collect Your Drawing Data
Launch the dataset collection tool:
```bash
python collect.py
```
* **Draw**: Drag your mouse to write the prompted digit.
* **Save & Next**: Press **Spacebar** or **Enter** to save the drawing and advance.
* **Clear**: Press **C** or **Escape** to wipe the canvas and redraw.
* **Undo / Back**: Click the **Undo / Back** button to delete the previous save and redraw it.
* You will collect 30 examples for each digit (1-9), making a total of 270 samples.

### Step 2: Train the DiagonNet Classifier
Train the PyTorch model on your custom dataset:
```bash
python train.py
```
* Automatically detects and utilizes your **NVIDIA CUDA GPU** (e.g., GeForce RTX 3060 Ti) for accelerated training.
* Preprocesses all drawings by cropping them to their bounding boxes and centering them.
* Augments the dataset by creating 9 variations of each image (1 original + 4 rotations + 4 directional shifts) to expand the training set to **2,430 samples**.
* Runs 100 epochs of training on the GPU using CrossEntropyLoss and the Adam optimizer.
* Dumps the trained model weight states into `model.pth`.

### Step 3: Run Interactive Predictions
Run the real-time inference tool:
```bash
python use.py
```
* **Real-time Recognition**: Draw on the canvas. The model processes your drawings and runs GPU-accelerated inference using `DiagonNet` to predict the digit dynamically as you drag your mouse!
* **Confidence Scores**: Shows probability percentages dynamically in a side panel.
* **Reset**: Press **C** or **Escape** to clear the canvas.

---

## 🧠 Under the Hood (Custom DiagonNet Pipeline)

1. **Alignment & Centering (`utils.py`)**: 
   To achieve scale and position invariance, every drawn digit is cropped down to its tightest bounding box (`Image.getbbox()`), padded symmetrically with a black border to restore a square aspect ratio, and then resized back to the target size.
2. **Double-Resizing Consistency**:
   To ensure the live testing environment in `use.py` matches the image degradation of the training files, drawings are first resized to 100x100 and then run through the centering pipeline. This matches the exact stroke widths and anti-aliasing artifacts.
3. **Data Augmentation**:
   Each image is cloned and modified with small rotations ($\pm 6^{\circ}$, $\pm 12^{\circ}$) and small translations ($\pm 5$ pixels) to build a robust classifier that handles variations in how you tilt and place drawings.
4. **Custom DiagonNet Architecture (with Chess Knight Extension)**:
   The custom `DiagonNet` model dynamically computes pixel-level spatial relationships. In its forward pass, it shifts the image in 4 diagonal directions (Top-Left, Top-Right, Bottom-Left, Bottom-Right) and 8 chess knight-move directions (L-shape shifts, e.g., Up 2 Right 1). It computes the difference maps between the original image and all 12 shifted versions. The original image, 4 diagonal maps, and 8 knight-move maps are concatenated (yielding a 13-channel feature tensor) before being flattened and passed to dense classification layers (`128` $\rightarrow$ `64` $\rightarrow$ `10` outputs).

---

## 👩‍🦰 Face Gender Classifier Subproject (`gender_classifier/`)

We have adapted the **DiagonNet** architecture to perform real-world face gender classification (recognizing Male vs. Female faces) using a dataset of ~2,300 cropped face images.

### 📁 Gender Project Files
* **[gender_classifier/train_gender.py](file:///E:/NAN/Ai/1/gender_classifier/train_gender.py)**: Loads, splits, augments, and trains DiagonNet on the face images using your CUDA GPU.
* **[gender_classifier/predict_gender.py](file:///E:/NAN/Ai/1/gender_classifier/predict_gender.py)**: Command-line prediction tool to test custom face images:
  ```bash
  python gender_classifier/predict_gender.py <path_to_image>
  ```
* **`gender_classifier/gender_model.pth`**: Trained binary classification weights state dict.

### 🚀 Training & Results
* **Dataset**: ~2,300 scraped and cropped face images from Google Images (split: 1,173 men, 1,134 women).
* **Augmented Dataset**: 16,605 samples (9x augmentation).
* **Accuracy**: Achieved **84.20% validation accuracy** on the split and **99.35%** on the full training dataset after 25 epochs on the RTX 3060 Ti GPU.


