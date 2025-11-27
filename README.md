# YOLO Computer Vision Project - BDD100K Dataset

End-to-end pipeline for training and evaluating YOLO models on the BDD100K autonomous driving dataset with dataset preparation, hyperparameter optimization, training, and comprehensive performance testing with attribute-based failure analysis.


## Colab Quick start

not needed if repo exist in g-drive, use pull
```bash
git clone https://github.com/m3mahdy/computer_vision_yolo
```

```bash
cd computer_vision_yolo
pip install -r requirements.txt```
```



## 🚀 Quick Start

```bash
# 1. Setup environment
git clone https://github.com/m3mahdy/computer_vision_yolo
cd computer_vision_yolo
python -m venv yolo_project
source yolo_project/bin/activate  # Windows: yolo_project\Scripts\activate
pip install -r requirements.txt

# 2. Prepare dataset (automated pipeline)
cd dataset
python 1_download_extract_main_dataset.py
python 2_convert_labels_to_yolo.py
python 2.5_validate_conversion.py
python 3_create_metadata_for_yolo.py
python 4_create_limited_datasets.py
python 5_validate_dataset.py
python 6_compress_dataset.py  # Optional: create zip archives

# 3. Run hyperparameter tuning
cd ../tune_train
jupyter notebook yolo_v0_tuning_main.ipynb

# 4. Train model with optimized parameters
jupyter notebook yolo_v0_training_main.ipynb

# 5. Test model with comprehensive analysis
cd ../yolo_test
# Option A: Run validation report (standard metrics + confusion matrix)
python run_yolo_validation_report.py --model yolov10n --dataset bdd100k_yolo_tiny --split test

# Option B: Run detailed testing report (includes attribute-based failure analysis)
python run_yolo_detailed_testing_report.py --model yolov10n --dataset bdd100k_yolo_tiny --split test
```

## 📋 Project Overview

**Features:**
- **Dataset Pipeline**: Automated download, conversion, validation, and metadata generation
- **Multiple Dataset Sizes**: Full (~100k), Limited (~2.3k), Tiny (~570 images), Tuning subsets
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Model Training**: Fine-tuning YOLO models with optimized parameters
- **Comprehensive Testing**: Two validation modes:
  - **Standard Validation**: Official YOLO metrics, confusion matrix, per-class analysis
  - **Detailed Analysis**: Attribute-based failure analysis (weather, scene, time of day, object size)
- **Performance Analysis**: Charts and CSVs analyzing accuracy by environmental conditions
- **GPU Support**: Automatic CUDA detection for faster training
- **Multiple YOLO Versions**: Support for YOLOv8, v9, v10, v11, v12

**BDD100K Dataset - 10 Classes:**
pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign

## 📂 Project Structure

```
computer_vision_yolo/
├── dataset/                           # Dataset preparation scripts
│   ├── 1_download_extract_main_dataset.py
│   ├── 2_convert_labels_to_yolo.py
│   ├── 2.5_validate_conversion.py
│   ├── 3_create_metadata_for_yolo.py
│   ├── 4_create_limited_datasets.py
│   ├── 5_validate_dataset.py
│   ├── 6_compress_dataset.py
│   ├── 7_compress_test_only.py
│   └── bdd100k_config.py              # Dataset configuration
│
├── bdd100k_yolo/                    # Full dataset (~100k images)
│   ├── images/ (train/val/test)
│   ├── labels/ (train/val/test)
│   ├── representative_json/           # Metadata with attributes
│   └── data.yaml
│
├── bdd100k_yolo_limited/              # Limited dataset (~2.3k samples)
├── bdd100k_yolo_tiny/                 # Tiny dataset (~570 images)
├── bdd100k_yolo_tuning_x/             # Tuning subset
│
├── models/                            # Trained model weights
│   ├── yolo10n/
│   ├── yolo12n/
│   └── ...
│
├── tune_train/                        # Training and tuning notebooks
│   ├── yolo_v0_tuning_main.ipynb
│   └── yolo_v0_training_main.ipynb
│
├── yolo_test/                         # Testing and validation
│   ├── run_yolo_validation_report.py  # Standard validation
│   ├── run_yolo_detailed_testing_report.py  # Detailed analysis
│   ├── detailed_analysis.py           # Analysis module
│   ├── collect_validation_results.py
│   ├── analyze_finetuned_models.ipynb
│   └── test_multi_model_yolo_report.ipynb
│
├── data_explore/                      # Data exploration notebooks
├── requirements.txt
└── README.md
```

## 📊 Dataset Setup

### Automated Dataset Pipeline

The dataset preparation is organized as a sequential pipeline with dedicated scripts:

```bash
cd dataset

# Step 1: Download and extract BDD100K dataset
python 1_download_extract_main_dataset.py
# Downloads images and labels, extracts to temporary folders

# Step 2: Convert BDD100K labels to YOLO format
python 2_convert_labels_to_yolo.py
# Converts JSON labels to YOLO txt format with proper class mapping

# Step 2.5: Validate conversion integrity
python 2.5_validate_conversion.py
# Ensures all images have corresponding labels and vice versa

# Step 3: Create metadata with image attributes
python 3_create_metadata_for_yolo.py
# Generates representative_json/ with weather, scene, timeofday metadata

# Step 4: Create subset datasets
python 4_create_limited_datasets.py
# Creates bdd100k_yolo_limited, bdd100k_yolo_tiny, bdd100k_yolo_tuning_x

# Step 5: Validate final datasets
python 5_validate_dataset.py
# Comprehensive validation of all dataset variants

# Step 6: (Optional) Compress datasets for distribution
python 6_compress_dataset.py
python 7_compress_test_only.py
```

**Configuration:**
All dataset paths and parameters are centralized in `dataset/bdd100k_config.py`:
- Source paths
- Output directories
- Class mappings
- Subset sizes
- Compression settings
**Dataset Variants:**

| Dataset | Images | Training Time | Use Case | Metadata |
|---------|--------|---------------|----------|----------|
| bdd100k_yolo_x | ~100k | ~25 hrs (50 epochs) | Final production models | ✓ Full attributes |
| bdd100k_yolo_limited | ~2.3k | ~2 hrs (50 epochs) | Quick testing, experiments | ✓ Representative |
| bdd100k_yolo_tiny | ~570 | ~30 min (50 epochs) | Rapid prototyping | ✓ Balanced sample |
| bdd100k_yolo_tuning_x | Subset | Variable | Hyperparameter tuning | ✓ Optimized for tuning |

**Metadata Structure:**
Each dataset includes `representative_json/` with per-image attributes:
```json
{
  "image_id": {
    "weather": "clear|rainy|snowy|overcast|partly cloudy|foggy",
    "scene": "city street|highway|residential|parking lot|...",
    "timeofday": "daytime|night|dawn/dusk",
    "object_count": 15,
    "class_counts": {
      "car": 8,
      "pedestrian": 5,
      "traffic light": 2
    }
  }
}
```

This enables detailed failure analysis by environmental conditions.

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
cd tune_train
jupyter notebook yolo_v0_tuning_main.ipynb
```

**Configuration:**
```python
MODEL_NAME = "yolov10n"  # Choose model version
DATASET_NAME = "bdd100k_yolo_tuning_x"  # Use tuning dataset
N_TRIALS = 30            # Number of optimization trials
EPOCHS_PER_TRIAL = 50    # Training epochs per trial
```

**Output:**
- Best hyperparameters JSON
- Optimization history visualization
- PDF report with results

### 2. Model Training

```bash
cd tune_train
jupyter notebook yolo_v0_training_main.ipynb
```

**Configuration:**
```python
MODEL_NAME = "yolov10n"  # Must match tuning notebook
DATASET_NAME = "bdd100k_yolo_tiny"  # Choose dataset size
# Hyperparameters loaded automatically from tuning phase
```

**Output:**
- Fine-tuned model: `models/{model_name}/{model_name}_finetuned-{date}.pt`
- Training curves visualization
- WandB logging (optional)

### 3. Model Validation & Testing

#### Option A: Standard Validation Report

```bash
cd yolo_test
python run_yolo_validation_report.py \
    --model yolov10n \
    --dataset bdd100k_yolo_tiny \
    --split test \
    --iou 0.5 \
    --batch-size 16
```

**Output Structure:**
```
yolo_test/analysis_runs/{model}_{dataset}_{timestamp}/
├── yolo_validation/              # Official YOLO validation results
│   ├── confusion_matrix.png
│   ├── F1_curve.png
│   ├── PR_curve.png
│   ├── P_curve.png
│   └── R_curve.png
├── metrics_summary.json          # All metrics in JSON
├── confusion_matrix_enhanced.png # Enhanced confusion matrix
├── per_class_metrics.csv         # Detailed per-class breakdown
└── validation_report.pdf         # Comprehensive PDF report
```

**Metrics Provided:**
- Official YOLO metrics: mAP@0.5, mAP@0.5:0.95
- Per-class: Precision, Recall, F1-Score
- Confusion matrix with percentages
- Model info: Parameters, Size, FPS

#### Option B: Detailed Testing Report (with Attribute Analysis)

```bash
cd yolo_test
python run_yolo_detailed_testing_report.py \
    --model yolov10n \
    --dataset bdd100k_yolo_tiny \
    --split test \
    --iou 0.5 \
    --batch-size 16
```

**Output Structure:**
```
yolo_test/analysis_runs/{model}_{dataset}_{timestamp}/
├── yolo_validation/              # Official YOLO validation results
├── performance_analysis/         # Detailed attribute-based analysis
│   ├── per_image_accuracy.csv
│   ├── accuracy_by_weather.csv
│   ├── accuracy_by_weather.png
│   ├── accuracy_by_scene.csv
│   ├── accuracy_by_scene.png
│   ├── accuracy_by_timeofday.csv
│   ├── accuracy_by_timeofday.png
│   ├── accuracy_by_object_count.csv
│   ├── accuracy_by_object_count.png
│   ├── accuracy_by_size.csv
│   ├── accuracy_by_size.png
│   ├── accuracy_by_class_and_*.csv/png
│   ├── train_test_distribution.png
│   ├── accuracy_vs_training_exposure.png
│   └── accuracy_vs_train_test_ratio.png
├── confusion_matrix_enhanced.png
├── per_class_metrics.csv
├── metrics_summary.json
└── detailed_testing_report.pdf   # Comprehensive PDF with all analysis
```

**Additional Analysis:**
- **By Weather**: Accuracy breakdown for clear, rainy, snowy, overcast, foggy conditions
- **By Scene**: Performance on city streets, highways, residential areas, parking lots
- **By Time of Day**: Daytime vs night vs dawn/dusk performance
- **By Object Count**: How density affects accuracy (1-5, 6-10, 11-20, 21-50, 50+ objects)
- **By Object Size**: Small (<1%), medium (1-5%), large (>5%) object detection
- **Per-Class Breakdowns**: Each class performance within each attribute
- **Training Exposure**: Correlation between training data exposure and test accuracy

**Configuration Constants** (`detailed_analysis.py`):
```python
# Easily adjustable thresholds
PREDICTION_CONF_THRESHOLD = 0.001       # Confidence threshold
MAX_DETECTIONS_PER_IMAGE = 300          # Max detections
OBJECT_COUNT_THRESHOLDS = [5,10,20,50]  # Count bucket boundaries
SIZE_BUCKET_SMALL_THRESHOLD = 0.01      # Small object threshold
SIZE_BUCKET_MEDIUM_THRESHOLD = 0.05     # Medium object threshold
```

## 📈 Supported Models

**YOLOv8:** yolov8n, yolov8s, yolov8m, yolov8l, yolov8x  
**YOLOv9:** yolov9s, yolov9m, yolov9l, yolov9x  
**YOLOv10:** yolov10n, yolov10s, yolov10m, yolov10l, yolov10x  
**YOLO11:** yolo11n, yolo11s, yolo11m, yolo11l, yolo11x  
**YOLO12:** yolo12n, yolo12s, yolo12m, yolo12l, yolo12x

(n=nano, s=small, m=medium, l=large, x=extra large)

## 🔧 Key Modules

### Dataset Pipeline (`dataset/`)
- **Automated preprocessing**: Download → Extract → Convert → Validate → Subset
- **Metadata generation**: Automatic attribute extraction (weather, scene, time)
- **Configuration-driven**: Centralized config in `bdd100k_config.py`
- **Integrity checks**: Validation at each step

### Detailed Analysis (`yolo_test/detailed_analysis.py`)
- **Hybrid validation approach**: 
  - Phase 1: Official `model.val()` for accurate metrics
  - Phase 2: Per-image `model.predict()` for detailed analysis
- **Modular design**: Separated prediction collection and analysis functions
- **Configurable thresholds**: All parameters adjustable via constants
- **Performance analysis subfolder**: Organized output structure

### Testing Scripts
- **`run_yolo_validation_report.py`**: Standard YOLO validation with enhanced visualizations
- **`run_yolo_detailed_testing_report.py`**: Extended analysis with attribute breakdowns
- **`collect_validation_results.py`**: Aggregate results from multiple test runs
- **Jupyter notebooks**: Interactive testing and multi-model comparison

## 🛠️ Utility Scripts

### Git Automation
```bash
python git_commit_push.py
```
Automatically adds, commits, and pushes changes to Git.

## 📚 Dependencies

Key libraries:
- `ultralytics`: YOLO implementation (v8/v10/v11/v12)
- `optuna`: Hyperparameter optimization
- `torch`: Deep learning framework with CUDA support
- `opencv-python`: Image processing
- `pillow`: Image manipulation
- `matplotlib`, `seaborn`: Visualization and plotting
- `pandas`, `numpy`: Data analysis and manipulation
- `tqdm`: Progress bars
- `pyyaml`: Configuration management
- `reportlab`: PDF report generation (optional)
- `wandb`: Experiment tracking (optional)

See `requirements.txt` for complete list with versions.

## 🔍 Example Use Cases

### Quick Model Comparison
```bash
# Test multiple models on tiny dataset
python yolo_test/run_yolo_validation_report.py --model yolov10n --dataset bdd100k_yolo_tiny --split test
python yolo_test/run_yolo_validation_report.py --model yolo12n --dataset bdd100k_yolo_tiny --split test
python yolo_test/collect_validation_results.py  # Compare results
```

### Deep Dive Analysis
```bash
# Run detailed testing with attribute analysis
python yolo_test/run_yolo_detailed_testing_report.py \
    --model yolov10n \
    --dataset bdd100k_yolo_limited \
    --split test \
    --iou 0.5

# Check performance_analysis/ folder for:
# - Accuracy by weather conditions
# - Accuracy by scene types
# - Object size impact
# - Training exposure correlation
```

### Production Training
```bash
# Full pipeline on complete dataset
cd dataset && python 1_download_extract_main_dataset.py
# ... complete dataset pipeline ...

cd ../tune_train
# Tune on tuning subset
jupyter notebook yolo_v0_tuning_main.ipynb  # DATASET_NAME = "bdd100k_yolo_tuning_x"

# Train on full dataset with optimized params
jupyter notebook yolo_v0_training_main.ipynb  # DATASET_NAME = "bdd100k_yolo_x"

# Test on full test set
cd ../yolo_test
python run_yolo_detailed_testing_report.py --model yolov10n --dataset bdd100k_yolo_x --split test
```

## 📄 License

This project uses the BDD100K dataset. Please refer to the [BDD100K license](https://bdd-data.berkeley.edu/) for dataset usage terms.

## 🙏 Acknowledgments

- **BDD100K dataset**: Berkeley DeepDrive - Comprehensive autonomous driving dataset
- **Ultralytics YOLO**: State-of-the-art object detection framework
- **Optuna**: Efficient hyperparameter optimization framework

## 📝 Notes

- **Hybrid Validation**: The detailed testing uses a two-phase approach:
  1. Official `model.val()` for accurate mAP metrics
  2. Per-image `model.predict()` for detailed attribute analysis
- **Metadata-driven**: All analysis leverages rich metadata (weather, scene, timeofday) extracted during dataset preparation
- **Modular Architecture**: Separated concerns enable easy extension and customization
- **Configuration Constants**: Key parameters are configurable constants at the top of `detailed_analysis.py`

## 🚧 Troubleshooting

**AttributeError: 'DetMetrics' object has no attribute 'per_image'**
- Fixed: Use `validation_results.per_image_records` (not `.per_image`)

**Empty performance_analysis folder**
- Fixed: Module now saves CSVs and charts to `performance_analysis/` subfolder

**Missing metadata attributes**
- Ensure Step 3 (`3_create_metadata_for_yolo.py`) was run successfully
- Check `representative_json/` folder exists with metadata files

---

**Happy Training! 🚀**
