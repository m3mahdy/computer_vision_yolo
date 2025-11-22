# YOLO Computer Vision Project - BDD100K Dataset

End-to-end pipeline for training and evaluating YOLO models on the BDD100K autonomous driving dataset with hyperparameter optimization, training, and comprehensive testing.

## 🚀 Quick Start

```bash
# 1. Setup environment
git clone https://github.com/m3mahdy/computer_vision_yolo
cd computer_vision_yolo
python -m venv yolo_project
source yolo_project/bin/activate  # Windows: yolo_project\Scripts\activate
pip install -r requirements.txt

# 2. Prepare dataset (automatic download and processing)
python3 process_bdd100k_to_yolo_dataset.py

# OR use limited dataset for quick testing
python3 process_limited_dataset.py

# 3. Run hyperparameter tuning
cd hyperparameter_tuning
jupyter notebook yolo_v_hyperparameter_tuning.ipynb

# 4. Train model with optimized parameters
cd ../training
jupyter notebook yolo_training.ipynb

# 5. Test model
cd ../yolo_test
jupyter notebook yolo_test.ipynb
```

## 📋 Project Overview

**Features:**
- **Hyperparameter Optimization**: Automated tuning using Optuna (30 trials)
- **Model Training**: Fine-tuning YOLO models with optimized parameters
- **Model Testing**: Comprehensive evaluation with detailed metrics and PDF reports
- **GPU Support**: Automatic CUDA detection for faster training
- **Multiple YOLO Versions**: Support for YOLOv8, v9, v10, v11, v12

**BDD100K Dataset - 10 Classes:**
pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign

## 📂 Project Structure

```
computer_vision_yolo/
├── bdd100k_yolo/                 # Full dataset (~100k images)
├── bdd100k_yolo_limited/         # Limited dataset (~2.3k samples)
├── models/{model_name}/          # Model weights
├── hyperparameter_tuning/        # Optuna optimization notebooks
├── training/                     # Training notebooks and configs
├── yolo_test/                    # Testing notebooks
├── process_bdd100k_to_yolo_dataset.py  # Dataset preparation
└── requirements.txt
```

## 📊 Dataset Setup

### Automatic Setup

```bash
# Complete dataset preparation with automatic download
python3 process_bdd100k_to_yolo_dataset.py
```

**This command:**
- Downloads dataset from Google Drive (if needed)
- Extracts images and labels
- Performs data integrity check
- Converts to YOLO format
- Creates full dataset (~100k images) and limited dataset (~2.3k samples)
- Generates data.yaml configuration

**Options:**
```bash
# Skip download check (when files already exist)
python3 process_bdd100k_to_yolo_dataset.py --skip-download

# Remove temporary files after processing
python3 process_bdd100k_to_yolo_dataset.py --cleanup
```

### Quick Start with Limited Dataset

```bash
# Download and extract only the limited dataset (~2.3k samples)
python3 process_limited_dataset.py
```

Perfect for quick experiments without processing 100k images.

**Dataset Comparison:**

| Dataset | Images | Training Time | Use Case |
|---------|--------|---------------|----------|
| Full | ~100k | ~25 hrs (50 epochs) | Final production models |
| Limited | ~2.3k | ~2 hrs (50 epochs) | Quick testing, tuning, experiments |

## 🔧 Configuration

Switch datasets by updating notebook configuration:

```python
BASE_DIR = Path.cwd().parent

# For limited dataset (quick experiments)
YOLO_DATASET_ROOT = BASE_DIR / 'bdd100k_yolo_limited'
DATA_YAML_PATH = YOLO_DATASET_ROOT / 'data.yaml'

# For full dataset (production training)
YOLO_DATASET_ROOT = BASE_DIR / 'bdd100k_yolo'
DATA_YAML_PATH = YOLO_DATASET_ROOT / 'data.yaml'
```

## 🎯 Workflow

### 1. Hyperparameter Tuning (Recommended)

```bash
cd hyperparameter_tuning
jupyter notebook yolo_v_hyperparameter_tuning.ipynb
```

**Configuration:**
```python
MODEL_NAME = "yolov8n"  # Choose model version
N_TRIALS = 30           # Number of optimization trials
EPOCHS_PER_TRIAL = 50   # Training epochs per trial
```

**Output:**
- Best hyperparameters JSON
- Optimization history visualization
- PDF report with results

### 2. Model Training

```bash
cd training
jupyter notebook yolo_training.ipynb
```

**Configuration:**
```python
MODEL_NAME = "yolov8n"  # Must match tuning notebook
# Hyperparameters loaded automatically from tuning phase
```

**Output:**
- Fine-tuned model: `models/{model_name}/{model_name}_finetuned-{date}.pt`
- Training curves visualization
- PDF training report

### 3. Model Testing

```bash
cd yolo_test
jupyter notebook yolo_test.ipynb
```

**Configuration:**
```python
MODEL_NAME = "yolov8n"  # Must match training
FINETUNED_MODEL = "{model_name}_finetuned-{date}.pt"
```

**Output:**
- Validation metrics (mAP@0.5, mAP@0.5:0.95, precision, recall)
- Confusion matrix visualization
- GT vs Predictions comparison images
- PDF testing report

## 📈 Supported Models

**YOLOv8:** yolov8n, yolov8s, yolov8m, yolov8l, yolov8x  
**YOLOv9:** yolov9s, yolov9m, yolov9l, yolov9x  
**YOLOv10:** yolov10n, yolov10s, yolov10m, yolov10l, yolov10x  
**YOLO11:** yolo11n, yolo11s, yolo11m, yolo11l, yolo11x  
**YOLO12:** yolo12n, yolo12s, yolo12m, yolo12l, yolo12x

(n=nano, s=small, m=medium, l=large, x=extra large)

## 🛠️ Utility Scripts

### Git Automation
```bash
python git_commit_push.py
```
Automatically adds, commits, and pushes changes to Git.

## 📚 Dependencies

Key libraries:
- `ultralytics`: YOLO implementation
- `optuna`: Hyperparameter optimization
- `torch`: Deep learning framework
- `opencv-python`: Image processing
- `matplotlib`, `seaborn`: Visualization
- `reportlab`: PDF generation

See `requirements.txt` for complete list.

## 📄 License

This project uses the BDD100K dataset. Please refer to the [BDD100K license](https://bdd-data.berkeley.edu/) for dataset usage terms.

## 🙏 Acknowledgments

- BDD100K dataset: Berkeley DeepDrive
- Ultralytics YOLO: State-of-the-art object detection
- Optuna: Automated hyperparameter optimization

---

**Happy Training! 🚀**
