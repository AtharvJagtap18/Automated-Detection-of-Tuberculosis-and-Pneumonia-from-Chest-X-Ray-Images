"""
Configuration file for the TB X-ray Analysis Backend
All paths and hyperparameters are defined here for easy modification
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model paths
MODELS_DIR = BASE_DIR / "models"
DISEASE_MODEL_PATH = MODELS_DIR / "disease_model.pth"
STAGE_MODEL_PATH = MODELS_DIR / "stage_model.pth"

# Image preprocessing settings
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class labels
DISEASE_CLASSES = ["Normal", "Pneumonia", "Tuberculosis"]
TB_STAGE_CLASSES = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]

# Model architecture settings
DISEASE_MODEL_ARCH = "resnet50"  # Options: resnet50, resnet101, densenet121, densenet169
STAGE_MODEL_ARCH = "resnet50"

# Training hyperparameters (for reference)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 10

# Inference settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to consider prediction valid
TB_THRESHOLD = 0.6  # Minimum confidence to trigger stage classification

# Device settings
DEVICE = "cpu"  # Force CPU for offline laptop deployment

# Random seed for reproducibility
RANDOM_SEED = 42

# Evaluation settings
EVAL_BATCH_SIZE = 16
CONFUSION_MATRIX_FIGSIZE = (10, 8)
CONFUSION_MATRIX_DPI = 100

# Output directories
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Data augmentation settings (for training)
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'rotation_range': 10,
    'brightness_range': (0.9, 1.1),
    'zoom_range': 0.1,
}

print(f"✅ Configuration loaded")
print(f"📁 Base directory: {BASE_DIR}")
print(f"🤖 Disease model: {DISEASE_MODEL_ARCH}")
print(f"🎯 Stage model: {STAGE_MODEL_ARCH}")
print(f"💻 Device: {DEVICE}")
