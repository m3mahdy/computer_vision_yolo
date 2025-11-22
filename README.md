# YOLO Computer Vision Project - BDD100K Dataset

A comprehensive computer vision project using YOLO models for object detection on the BDD100K dataset. This project includes complete workflows for hyperparameter tuning, model training, and testing with detailed reporting.

## 📋 Project Overview

This project implements an end-to-end pipeline for training and evaluating YOLO models on the BDD100K autonomous driving dataset. It includes:

- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Model Training**: Fine-tuning YOLO models with optimized parameters
- **Model Testing**: Comprehensive evaluation with detailed metrics
- **PDF Reporting**: Professional reports for all phases
- **Visualization**: Training curves, confusion matrices, and sample predictions

## 🏗️ Project Structure

```
computer_vision_yolo/
├── bdd_100k_source/              # Raw dataset downloads (zip files)
├── bdd100k_yolo/                 # Full YOLO dataset (~100k images)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── representative_json/      # Diverse samples with metadata
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml                 # YOLO configuration file
├── bdd100k_yolo_limited/         # Limited dataset (representative samples)
│   ├── images/
│   ├── labels/
│   ├── representative_json/
│   └── data.yaml
├── models/                       # YOLO model weights
│   └── {model_name}/
│       ├── {model_name}.pt              # Base model
│       ├── {model_name}_finetuned-{date}.pt  # Fine-tuned model
│       └── {model_name}_finetuned-{date}_metadata.json
├── hyperparameter_tuning/        # Hyperparameter optimization
│   ├── yolo_v_hyperparameter_tuning.ipynb
│   └── runs/
├── training/                     # Model training
│   ├── yolo_training.ipynb
│   ├── {model_name}_best_hyperparameters.json
│   └── runs/
├── testing/                      # Model evaluation
│   ├── yolo_testing.ipynb
│   └── runs/
├── quick_test/                   # Quick validation tests
│   └── yolo_quick_test.ipynb
├── tmp/                          # Temporary files
├── process_bdd100k_to_yolo_dataset.py  # Unified dataset preparation script
├── git_commit_push.py            # Git automation script
└── requirements.txt              # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 50GB free disk space for full dataset
- Git for version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/m3mahdy/computer_vision_yolo
   cd computer_vision_yolo
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv yolo_project
   source yolo_project/bin/activate  # On Windows: yolo_project\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Setup

### Automatic Dataset Preparation (Recommended)

The project includes a unified script that handles everything automatically:

```bash
# Run the complete dataset preparation
python3 process_bdd100k_to_yolo_dataset.py
```

This single command will:
1. **Check for dataset files** in `bdd_100k_source/`
   - If missing, prompts to download automatically using `gdown` (faster Google Drive mirror)
   - Original BDD100K URLs available for manual download: http://bdd-data.berkeley.edu/
2. **Extract and convert** images and labels to YOLO format
3. **Select representative samples** with diverse attributes (weather, scene, time of day)
4. **Save original JSON files** for representative samples with metadata
5. **Create full dataset** (~100k images): `bdd100k_yolo/`
6. **Create limited dataset** (representative samples): `bdd100k_yolo_limited/`
   - Automatically includes all representative samples selected from full dataset
   - Comprehensive coverage with diverse attributes (weather, scene, time)
   - Perfect for quick testing, visualization, and notebook experimentation
   - Same metadata files as full dataset for consistent analysis
7. **Compress limited dataset** to `bdd100k_yolo_limited.zip` (~500MB-1GB)
   - Easy distribution and quick setup
   - No need to process full 100K dataset for experiments
   - Ideal for quick starts and testing
8. **Keep temporary files** by default for reference and debugging

**Key Features:**
- **Comprehensive Representative Sampling** with 4-step coverage:
  1. **Attribute Combinations**: 10 samples per (weather × scene × time) combination
  2. **Class Coverage**: Minimum 10 samples per object class
  3. **Attribute Values**: Minimum 5 samples per individual attribute value
  4. **Class×Attribute Combinations**: Minimum 3 samples per (class × attribute) combo per split
- **Metadata Report Files**: Pre-computed statistics saved in `{split}_metadata.json`
  - Contains all coverage statistics
  - Lists selected sample basenames for each case
  - Enables fast visualization without re-analysis
  - Separated per split for independent coverage
- **JSON Preservation**: Original BDD100K JSON files with attributes preserved
- **Deterministic Selection**: Same representative samples every run
- **Quality Over Quantity**: Guaranteed diverse coverage over random sampling

**Advanced Options:**

```bash
# Skip download check (use when files already exist)
python3 process_bdd100k_to_yolo_dataset.py --skip-download

# Force metadata regeneration (regenerate statistics even if dataset exists)
python3 process_bdd100k_to_yolo_dataset.py --force-reanalysis

# Remove temporary extraction directories (by default, they are kept)
python3 process_bdd100k_to_yolo_dataset.py --cleanup
```

### 📦 Quick Start with Limited Dataset

**If you want to start quickly without processing the full 100K dataset**, use the compressed limited dataset:

```bash
# Extract the limited dataset (~2.3K representative samples)
python unzip_limited_dataset.py
```

This provides:
- ✅ **Representative samples** (~2,300 images) covering all classes and attributes
- ✅ **Quick experiments** - Train, tune, and test in minutes instead of hours
- ✅ **Low disk space** - ~2-3GB uncompressed vs ~6GB+ for full dataset
- ✅ **Same structure** - Drop-in replacement for `bdd100k_yolo` in notebooks
- ✅ **Perfect for**:
  - Learning YOLO training workflows
  - Hyperparameter tuning (faster iterations)
  - Pipeline testing and validation
  - Quick model prototyping

**When to use which dataset:**
- **Limited dataset** (`bdd100k_yolo_limited.zip`): Quick experiments, learning, tuning (IS the representative ~2.3K samples)
- **Full dataset** (`bdd100k_yolo`): Production training, final evaluation, benchmarking (all 100K images)

**Important:** The limited dataset IS the representative sample - it's a physical copy of carefully selected diverse samples from the full dataset.

All notebooks support both datasets - just update the dataset path in configuration cells.

**Dataset Structure After Extraction:**
```
bdd100k_yolo/
├── images/          # Training images
│   ├── train/
│   ├── val/
│   └── test/
├── labels/          # YOLO format labels
│   ├── train/
│   ├── val/
│   └── test/
├── representative_json/  # Diverse samples with metadata
│   ├── train_metadata.json   # Pre-computed statistics for train
│   ├── val_metadata.json     # Pre-computed statistics for val
│   ├── test_metadata.json    # Pre-computed statistics for test
│   ├── train/       # Selected JSON files with attributes
│   ├── val/
│   └── test/
└── data.yaml        # YOLO configuration
```

### Manual Download (Alternative)

If you prefer to download manually:

```bash
# 1. Create directory
mkdir -p bdd_100k_source

# 2. Install gdown
pip install gdown

# 3. Download using gdown (faster - Google Drive mirror)
gdown 1yHEpeEdRDAz5yH4pbo4o1SvzKzGKRaLS -O bdd_100k_source/bdd100k_images_100k.zip
gdown 1Gh_5g-MAx1R5X3eNsTTdz_GPialECz0L -O bdd_100k_source/bdd100k_labels.zip

# Alternative: Download from official website
# Visit: http://bdd-data.berkeley.edu/
# Download and place files in bdd_100k_source/

# 4. Run dataset preparation
python3 process_bdd100k_to_yolo_dataset.py --skip-download
```

**Download Information:**
- **Expected sizes**: Images (~6GB), Labels (~300MB)
- **Google Drive IDs**: Images: `1yHEpeEdRDAz5yH4pbo4o1SvzKzGKRaLS`, Labels: `1Gh_5g-MAx1R5X3eNsTTdz_GPialECz0L`
- **Original URLs** (for reference):
  - Images: `https://dl.cv.ethz.ch/bdd100k/data/100k_images.zip`
  - Labels: `https://dl.cv.ethz.ch/bdd100k/data/det_20_labels.zip`
  - Website: `http://bdd-data.berkeley.edu/`
- **Why gdown?** Faster downloads via Google Drive CDN, built-in progress bars, automatic retry, more reliable

### Dataset Structure

**Full Dataset** (`bdd100k_yolo/`):
```
bdd100k_yolo/
├── images/
│   ├── train/     (~70k images)
│   ├── val/       (~10k images)
│   └── test/      (~20k images)
├── labels/
│   ├── train/     (~70k .txt files in YOLO format)
│   ├── val/       (~10k .txt files)
│   └── test/      (~20k .txt files)
├── representative_json/  # NEW: Diverse samples with metadata
│   ├── train/     (Selected JSON files with attributes)
│   ├── val/       (Selected JSON files with attributes)
│   ├── test/      (Selected JSON files with attributes)
│   ├── train_metadata.json           # Aggregate statistics
│   ├── train_performance_analysis.json  # Per-image details
│   ├── val_metadata.json
│   ├── val_performance_analysis.json
│   ├── test_metadata.json
│   └── test_performance_analysis.json
└── data.yaml      # YOLO configuration
```
- ~100k images total (70k train, 10k val, 20k test)
- Best for final model training
- Requires ~6GB disk space
- **Representative JSON files** preserve original BDD100K metadata for visualization and analysis
- **Performance analysis files** contain per-image attributes for model evaluation

**Representative Samples & Metadata System:**
The extraction script creates a comprehensive metadata system:

**Metadata Files** (`{split}_metadata.json`):
- **Pre-computed statistics**: All coverage stats calculated during extraction
- **Selected samples**: Basenames for each case/combination
- **Configuration tracking**: Records constants used for selection
- **Per-split separation**: Independent metadata for train/val/test

**Coverage Guarantees** (configurable via constants):
- **Attribute Combinations**: `SAMPLES_PER_ATTRIBUTE_COMBO = 10`
  - 10 samples per (weather × scene × time) combination
  - Weather: clear, overcast, rainy, snowy, partly cloudy
  - Scene: city street, highway, residential, parking lot
  - Time: daytime, night, dawn/dusk
- **Class Coverage**: `MIN_SAMPLES_PER_CLASS = 10`
  - Minimum 10 samples per object class per split
- **Attribute Value Coverage**: `MIN_SAMPLES_PER_ATTRIBUTE_VALUE = 5`
  - Minimum 5 samples per individual attribute value
- **Class×Attribute Combos**: `MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO = 3`
  - Minimum 3 samples per (class × attribute) pair per split
  - Examples: "car in rain", "pedestrian at night", "truck on highway"

**Metadata Usage in Notebooks:**
```python
# Load pre-computed metadata (fast - just JSON parsing)
with open('representative_json/train_metadata.json') as f:
    metadata = json.load(f)

# Use statistics directly (no re-analysis needed)
print(f"Total selected: {metadata['statistics']['total_selected']}")
print(f"Car samples: {metadata['statistics']['by_class']['car']}")
print(f"Car in rain: {metadata['statistics']['by_class_weather']['car|rainy']}")

# Get selected sample basenames
car_samples = metadata['selected_samples']['by_class']['car']
```

This ensures exploration notebooks are **fast** (no re-scanning) and **reproducible** (same samples every time).

**Limited Dataset** (`bdd100k_yolo_limited/`):
```
bdd100k_yolo_limited/
├── images/
│   ├── train/     (~300-500 images)
│   ├── val/       (~100-200 images)
│   └── test/      (~200-300 images)
├── labels/
│   ├── train/     (YOLO format labels)
│   ├── val/       (YOLO format labels)
│   └── test/      (YOLO format labels)
├── representative_json/  # Original BDD100K JSON files
│   ├── train/     (Metadata with attributes)
│   ├── val/
│   ├── test/
│   ├── train_metadata.json           # Aggregate statistics
│   ├── train_performance_analysis.json  # Per-image details
│   ├── val_metadata.json
│   ├── val_performance_analysis.json
│   ├── test_metadata.json
│   └── test_performance_analysis.json
└── data.yaml      # YOLO configuration
```
- **Diverse representative samples** (not random)
- Ensures **minimum 10 samples per object class** per split
- Covers all weather conditions, scenes, and times of day
- Perfect for quick testing and experimentation
- Faster training iterations while maintaining dataset diversity
- Includes original JSON files with attribute metadata
- **Performance analysis files** enable attribute-based model evaluation
- Requires minimal disk space (~100-500MB)

Both datasets include proper YOLO structure with `data.yaml` configuration.

### Performance Comparison

| Dataset | Images per Split | Epoch Time | Full Training Time | Use Case |
|---------|-----------------|------------|-------------------|----------|
| Full | 70k train / 10k val | ~30 min | ~25 hours (50 epochs) | Final models |
| Limited | ~300-500 train / ~100-200 val | ~1-2 min | ~1-2 hours (50 epochs) | Quick testing |

**Limited Dataset Advantages:**
- **Comprehensive coverage**: All attribute combinations represented
- **Class balance**: Minimum 10 samples per class guaranteed
- **Attribute coverage**: Each weather/scene/time value covered
- **Scenario diversity**: Class × attribute combinations included
- **Metadata-driven**: Pre-computed stats enable fast exploration
- **No re-analysis**: Notebooks load metadata files directly
- **Reproducible**: Same samples selected deterministically
- **Quick iterations**: Fast training while maintaining quality
- **Configuration aware**: Tracks coverage parameters used

**Recommendation:** Use limited dataset for:
- Hyperparameter tuning trials
- Notebook development and debugging
- Quick validation of changes
- Testing new architectures
- Understanding dataset composition

Then switch to full dataset for final model training.

### Verify Dataset Setup

After running the script, verify your setup:

```bash
# Check full dataset
ls -lh bdd100k_yolo/
cat bdd100k_yolo/data.yaml

# Check limited dataset
ls -lh bdd100k_yolo_limited/
cat bdd100k_yolo_limited/data.yaml
```

Both `data.yaml` files should contain:
```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
```

## 🔧 Configuration

All notebooks use the YOLO dataset directly via `data.yaml`:

```python
# In hyperparameter tuning and training notebooks
BASE_DIR = Path.cwd().parent

# For full dataset
YOLO_DATASET_ROOT = BASE_DIR / 'bdd100k_yolo'
DATA_YAML_PATH = YOLO_DATASET_ROOT / 'data.yaml'

# For limited dataset (recommended for quick testing)
YOLO_DATASET_ROOT = BASE_DIR / 'bdd100k_yolo_limited'
DATA_YAML_PATH = YOLO_DATASET_ROOT / 'data.yaml'
```

**Using Representative Metadata System:**

The extraction script automatically creates metadata files with pre-computed statistics:

```python
# In exploration notebook
REPRESENTATIVE_JSON_DIR = YOLO_DATASET_ROOT / 'representative_json'

# Load pre-computed metadata (fast - no re-analysis)
import json
with open(REPRESENTATIVE_JSON_DIR / 'train_metadata.json') as f:
    train_metadata = json.load(f)

# Use statistics directly
print(f"Total selected: {train_metadata['statistics']['total_selected']}")
print(f"Car samples: {train_metadata['statistics']['by_class']['car']}")
print(f"Coverage: {len(train_metadata['statistics']['by_attribute_combo'])} combos")

# Get selected samples for visualization
car_samples = train_metadata['selected_samples']['by_class']['car']
car_rainy = train_metadata['selected_samples']['by_class_weather']['car|rainy']
```

**Metadata Benefits:**
- ⚡ **Instant loading**: JSON parsing vs. scanning thousands of files
- 🔄 **Reproducible**: Same samples every time
- 📊 **Complete stats**: All combinations pre-computed per split
- ✅ **Coverage guarantee**: Ensures minimum samples for each case

**Switching Between Datasets:**

Simply change the dataset root in notebook configuration:
- Use `bdd100k_yolo/` for full dataset (~100k images)
- Use `bdd100k_yolo_limited/` for quick testing (representative samples)

## 📋 Representative Metadata System

### Overview
The representative JSON system creates **pre-computed metadata files** during dataset extraction:
- **Fast exploration**: Load statistics instantly without re-scanning
- **Reproducible**: Same representative samples every time
- **Complete coverage**: Guarantees minimum samples for all cases
- **Per-split tracking**: Independent metadata for train/val/test

### Metadata Files

Each split has two types of metadata files:

**1. Aggregate Statistics** (`{split}_metadata.json`):
```json
{
  "split": "train",
  "generation_date": "2025-11-22 10:30:45",
  "configuration": {
    "samples_per_attribute_combo": 10,
    "min_samples_per_class": 10,
    "min_samples_per_attribute_value": 5,
    "min_samples_per_class_attribute_combo": 3
  },
  "statistics": {
    "total_selected": 850,
    "by_class": {"car": 120, "pedestrian": 95, ...},
    "by_weather": {"clear": 350, "rainy": 180, ...},
    "by_attribute_combo": {"clear|city street|daytime": 10, ...},
    "by_class_weather": {"car|rainy": 8, ...}
  },
  "selected_samples": {
    "by_class": {"car": ["img001", "img052", ...], ...},
    "by_class_weather": {"car|rainy": ["img123", ...], ...}
  }
}
```

**2. Performance Analysis** (`{split}_performance_analysis.json`):
```json
{
  "split": "test",
  "generation_date": "2024-11-22 10:30:00",
  "total_images": 10000,
  "images": [
    {
      "basename": "cabc30fc-e7726578",
      "weather": "clear",
      "scene": "city street",
      "timeofday": "daytime",
      "classes_present": ["car", "person", "traffic sign"],
      "objects_per_class": {"car": 5, "person": 2, "traffic sign": 3},
      "total_objects": 10
    }
  ]
}
```

### 4-Step Coverage System

1. **Attribute Combinations** (`SAMPLES_PER_ATTRIBUTE_COMBO = 10`)
   - 10 samples per (weather × scene × time) combination
   - Ensures all environmental conditions covered

2. **Class Coverage** (`MIN_SAMPLES_PER_CLASS = 10`)
   - Minimum 10 samples per object class per split
   - Guarantees adequate representation for training

3. **Attribute Values** (`MIN_SAMPLES_PER_ATTRIBUTE_VALUE = 5`)
   - Minimum 5 samples per individual attribute value
   - Covers each weather type, scene type, and time independently

4. **Class×Attribute Combos** (`MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO = 3`)
   - Minimum 3 samples per (class × attribute) pair **per split**
   - Examples: "car in rain", "pedestrian at night"
   - Separated by split for independent coverage

### Usage in Notebooks

**Exploration Notebook** loads both metadata types:

```python
# Load aggregate statistics (fast - just JSON parsing)
with open('representative_json/train_metadata.json') as f:
    metadata = json.load(f)

# No re-analysis needed! Use pre-computed statistics:
print(f"Total: {metadata['statistics']['total_selected']}")
print(f"Cars: {metadata['statistics']['by_class']['car']}")
print(f"Cars in rain: {metadata['statistics']['by_class_weather']['car|rainy']}")

# Get sample basenames for visualization
for basename in metadata['selected_samples']['by_class']['car'][:9]:
    img_path = f"images/train/{basename}.jpg"
    # Display image...

# Load performance data for attribute analysis
with open('representative_json/test_performance_analysis.json') as f:
    perf_data = json.load(f)

# Access per-image attributes
for img in perf_data['images'][:5]:
    print(f"{img['basename']}: {img['weather']}, {img['scene']}, {img['timeofday']}")
```

**Test Notebook** uses performance metadata for evaluation:

```python
# Load performance metadata
with open('representative_json/test_performance_analysis.json') as f:
    performance_data = json.load(f)

# Create quick lookup
image_attributes = {img['basename']: img for img in performance_data['images']}

# Analyze model performance by attributes
for result in validation_results:
    basename = Path(result['image_path']).stem
    attrs = image_attributes.get(basename, {})
    
    # Group performance by weather, scene, time
    if attrs.get('weather') == 'rainy':
        rainy_results.append(result)
    # Calculate metrics per attribute group...
```

**Benefits:**
- ⚡ **10x faster**: Instant loading vs. 5-10 minutes scanning
- 🔄 **100% reproducible**: Same samples every run
- 📊 **Complete**: All combinations tracked per split
- ✅ **Guaranteed coverage**: No missing scenarios
- 🎯 **Attribute-based evaluation**: Performance analysis by conditions

## 🎯 Workflow

### 1. Hyperparameter Tuning (Optional but Recommended)

Optimize model hyperparameters using Optuna:

```bash
cd hyperparameter_tuning
jupyter notebook yolo_v_hyperparameter_tuning.ipynb
```

**What it does:**
- Runs 20 optimization trials with conservative hyperparameter ranges
- Tests learning rates, augmentation parameters, loss weights around YOLO defaults
- Uses 50 epochs per trial for better convergence
- Enables data augmentation for improved generalization
- Uses validation split for evaluation
- Generates optimization history and parameter importance plots
- Saves best hyperparameters to `training/{model_name}_best_hyperparameters.json`
- Creates comprehensive PDF report

**Key Configuration:**
```python
MODEL_NAME = "yolov8m"  # Choose your model

# Dataset Selection
YOLO_DATASET_ROOT = BASE_DIR / 'bdd100k_yolo_limited'  # Limited for quick tuning
# OR
YOLO_DATASET_ROOT = BASE_DIR / 'bdd100k_yolo'  # Full dataset for final optimization
DATA_YAML_PATH = YOLO_DATASET_ROOT / 'data.yaml'
```

**Output:**
- Best hyperparameters JSON file
- Optimization history visualization
- Parameter importance analysis
- PDF report with all results
- W&B integration logs (optional)

### 2. Model Training

Train YOLO model with optimized hyperparameters:

```bash
cd training
jupyter notebook yolo_training.ipynb
```

**What it does:**
- Loads optimized hyperparameters from tuning phase (or uses defaults)
- Trains on train split, validates on val split
- Saves fine-tuned model as `{model_name}_finetuned-{date}.pt`
- Tracks training metrics (loss, mAP, precision, recall)
- Generates training curves and per-class performance analysis
- Creates comprehensive PDF report

**Key Configuration:**
```python
MODEL_NAME = "yolov8n"  # Must match tuning notebook

# Dataset automatically loaded from data.yaml
YOLO_DATASET_ROOT = BASE_DIR / 'bdd100k_yolo'
DATA_YAML_PATH = YOLO_DATASET_ROOT / 'data.yaml'
```

**Output:**
- Fine-tuned model: `models/{model_name}/{model_name}_finetuned-{date}.pt`
- Training metadata JSON
- Training curves visualization
- Per-class performance charts
- PDF training report

### 3. Model Testing

Evaluate trained model on test set:

```bash
cd testing
jupyter notebook yolo_testing.ipynb
```

**What it does:**
- Loads fine-tuned model from training phase
- Evaluates on test split
- Generates confusion matrix
- Shows GT vs Predictions comparisons (12 samples)
- Calculates detailed metrics per class
- **Attribute-based performance analysis**: Evaluates model across environmental conditions
- Creates comprehensive PDF report

**Attribute-Based Analysis Features:**
- **Performance by weather**: Compare model accuracy across clear, rainy, snowy conditions
- **Performance by scene**: Analyze results for city streets, highways, residential areas
- **Performance by time of day**: Evaluate daytime vs. night vs. dawn/dusk performance
- **Per-class attribute breakdown**: Identify which classes struggle in specific conditions
- **Visual charts**: 4-panel visualization showing F1 scores across all attributes
- **Actionable insights**: Pinpoint problematic scenarios (e.g., "model struggles with pedestrians at night")

**Example Output:**
```
PERFORMANCE BY WEATHER CONDITION
CLEAR:
  Images: 5000
  Precision: 0.842
  Recall: 0.798
  F1 Score: 0.819

RAINY:
  Images: 1200
  Precision: 0.731
  Recall: 0.689
  F1 Score: 0.709
  
PERFORMANCE BY SCENE TYPE
CITY STREET:
  Images: 3500
  Precision: 0.815
  Recall: 0.782
  F1 Score: 0.798
```

**Output:**
- Validation metrics (mAP@0.5, mAP@0.5:0.95, precision, recall)
- Confusion matrix visualization
- Per-class performance analysis
- Attribute-based performance breakdown (weather/scene/time)
- GT vs Predictions comparison images
- `attribute_performance.png` - Performance visualization by attributes
- PDF testing report

### 4. Quick Validation (Optional)

For rapid model validation during development:

```bash
cd quick_test
jupyter notebook yolo_quick_test.ipynb
```

## 📈 Model Support

The project supports multiple YOLO versions:

- **YOLOv8**: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- **YOLOv9**: `yolov9s`, `yolov9m`, `yolov9l`, `yolov9x`
- **YOLOv10**: `yolov10n`, `yolov10s`, `yolov10m`, `yolov10l`, `yolov10x`
- **YOLO11**: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`
- **YOLO12**: `yolo12n`, `yolo12s`, `yolo12m`, `yolo12l`, `yolo12x`

Model sizes (n=nano, s=small, m=medium, l=large, x=extra large)

## 🎨 BDD100K Classes

The dataset includes 10 object classes:
- car
- person
- truck
- bus
- motorcycle
- bicycle
- rider
- train
- traffic light
- traffic sign

## 📊 Reports and Outputs

Each phase generates detailed reports:

### Hyperparameter Tuning Report
- Configuration summary
- Best hyperparameters table
- Top 10 trials comparison
- Optimization history chart
- Parameter importance analysis

### Training Report
- Model configuration
- Fine-tuned model information
- Validation metrics
- Per-class performance
- Training curves (loss, mAP, precision, recall, learning rate)
- Class performance analysis

### Testing Report
- Test set metrics
- Confusion matrix
- Per-class performance
- GT vs Predictions visual comparisons (12 samples)
- Detailed error analysis

## 🔧 Configuration

### Key Parameters to Adjust

**In hyperparameter tuning notebook:**
```python
MODEL_NAME = "yolo11n"  # Model version
N_TRIALS = 20           # Number of optimization trials
TIMEOUT_HOURS = 4       # Maximum time for optimization
USED_DATA_SPLIT = "val" # Split used for tuning
```

**In training notebook:**
```python
MODEL_NAME = "yolo11n"     # Must match tuning
USED_DATASET = "bdd100k_tmp_images_limited"  # Dataset to use
# Hyperparameters loaded automatically from tuning phase
```

**In testing notebook:**
```python
MODEL_NAME = "yolo11n"     # Must match training
FINETUNED_MODEL = "{model_name}_finetuned-{date}.pt"  # Model to test
USED_DATA_SPLIT = "test"   # Use test split
```

## 🛠️ Utility Scripts

### Dataset Preparation
```bash
python3 process_bdd100k_to_yolo_dataset.py
```
Unified script that handles download, extraction, conversion, and dataset creation.

### Git Automation
```bash
python git_commit_push.py
```
Automatically adds, commits, and pushes changes to Git.

## 🐛 Troubleshooting

**Representative samples not showing attributes**
- **Issue**: Exploration notebook doesn't display weather/scene/time metadata
- **Solution**: Ensure `representative_json/` directory exists in your dataset
  ```bash
  # Regenerate dataset with representative samples
  python3 process_bdd100k_to_yolo_dataset.py
  ```
- **Check**: Look for `bdd100k_yolo/representative_json/{train,val,test}/` directories

### Dataset Setup Issues

**Script fails with "command not found: python"**
```bash
# Use python3 instead
python3 process_bdd100k_to_yolo_dataset.py
```

**Download is slow or fails**
```bash
# Use manual gdown download for faster speeds
mkdir -p bdd_100k_source
pip install gdown
gdown 1yHEpeEdRDAz5yH4pbo4o1SvzKzGKRaLS -O bdd_100k_source/bdd100k_images_100k.zip
gdown 1Gh_5g-MAx1R5X3eNsTTdz_GPialECz0L -O bdd_100k_source/bdd100k_labels.zip
python3 process_bdd100k_to_yolo_dataset.py --skip-download
```

**"Dataset files not found" but files exist**
- Ensure zip files are in `bdd_100k_source/` directory:
  - `bdd100k_images_100k.zip`
  - `bdd100k_labels.zip`

**gdown installation fails**
```bash
pip install --upgrade pip
pip install gdown
```
Or download from official site: http://bdd-data.berkeley.edu/

**Download quota exceeded (Google Drive)**
- Wait a few hours and retry
- Use the official BDD100K URLs
- Download from a different network/IP

**Want to recreate datasets**
```bash
rm -rf bdd100k_yolo bdd100k_yolo_limited
python3 process_bdd100k_to_yolo_dataset.py
```

### Training Issues

**Out of Memory Errors:**
- Reduce batch size in hyperparameters
- Use smaller model (e.g., yolo11n instead of yolo11l)
- Use limited dataset

**Download Failures:**
- Check internet connection
- Script handles SSL errors automatically
- Re-run download script to resume

**Model Not Found:**
- Models auto-download on first use
- Check `models/{model_name}/` directory
- Verify MODEL_NAME spelling

**Import Errors:**
```bash
pip install -r requirements.txt --upgrade
```

## 📝 Best Practices

1. **Start Small**: Use limited dataset for initial experiments
2. **Tune First**: Run hyperparameter tuning before full training
3. **Monitor Training**: Check training curves for overfitting
4. **Validate Regularly**: Use quick_test for rapid validation
5. **Document Results**: Review PDF reports after each phase
6. **Version Models**: Fine-tuned models include date suffix for tracking
7. **Use W&B**: Enable Weights & Biases for experiment tracking

## 📝 Notes

- **No dataset preparation in notebooks**: All notebooks now directly use `data.yaml`
- **Absolute imports**: All scripts use absolute paths for reliability
- **Clean structure**: Limited datasets are independent copies, not symlinks
- **Git-friendly**: Large datasets are in `.gitignore`

## 📚 Dependencies

Key libraries:
- `ultralytics`: YOLO implementation
- `optuna`: Hyperparameter optimization
- `torch`: Deep learning framework
- `opencv-python`: Image processing
- `matplotlib`, `seaborn`: Visualization
- `pandas`: Data analysis
- `reportlab`: PDF generation
- `wandb`: Experiment tracking (optional)

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project uses the BDD100K dataset. Please refer to the [BDD100K license](https://bdd-data.berkeley.edu/) for dataset usage terms.

## 🙏 Acknowledgments

- BDD100K dataset: Berkeley DeepDrive
- Ultralytics YOLO: State-of-the-art object detection
- Optuna: Automated hyperparameter optimization

## 📧 Support

For issues and questions:
1. Check existing issues in repository
2. Review troubleshooting section
3. Create new issue with detailed description

---

**Happy Training! 🚀**
