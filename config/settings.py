import os

# Project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# General Configuration
GRID_SIZE = 100
BATCH_SIZE = 64
EPOCHS_DIGITS = 100
EPOCHS_GENDER = 25
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Hidden dims for MLP classifier layers
HIDDEN_LAYERS = [256, 128, 64]

# File Paths
DIGIT_DATA_DIR = os.path.join(BASE_DIR, "data")
GENDER_DATA_DIR = os.path.join(BASE_DIR, "gender_classifier", "gender_dataset_face")

DIGIT_MODEL_PATH = os.path.join(BASE_DIR, "weights", "diagonnet_digits.pth")
GENDER_MODEL_PATH = os.path.join(BASE_DIR, "weights", "diagonnet_gender.pth")
YOLO_GENDER_RUN_DIR = os.path.join(BASE_DIR, "gender_classifier", "yolo_runs")

COMPARISON_CHART_PATH = os.path.join(BASE_DIR, "assets", "comparison_chart.png")
