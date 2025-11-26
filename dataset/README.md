# BDD100K Dataset Processing Scripts

Modular scripts for processing BDD100K dataset into YOLO format. Scripts are numbered for execution order.

## Scripts Overview

### 1️⃣ Download and Extract Main Dataset
```bash
python dataset/1_download_extract_main_dataset.py
```
Downloads BDD100K source files (~6GB images, ~300MB labels) from Google Drive and extracts to `bdd100k_tmp_images/` and `bdd100k_tmp_labels/`. Automatically skips if already extracted.

### 2️⃣ Convert Labels to YOLO Format
```bash
python dataset/2_convert_labels_to_yolo.py
```
Converts JSON labels to YOLO `.txt` format, creates `data.yaml`, and **automatically copies images** from `bdd100k_tmp_images/100k/` to `bdd100k_yolo/images/`. Output: `bdd100k_yolo/`

### 2.5️⃣ Validate Conversion Output
```bash
python dataset/2.5_validate_conversion.py
```
Comprehensive validation of script 2 outputs: verifies object counts match between JSON and YOLO labels, validates file counts, checks directory structure, verifies data.yaml format, and ensures image-label correspondence.

### 3️⃣ Create Limited Datasets
```bash
python dataset/3_create_limited_datasets.py
```
Interactive menu creates subset datasets with sophisticated representative sampling: Limited (~25K train), Tuning (~14K train), or Tiny (~500 train). Ensures diverse coverage across weather, scene, time, and class combinations.

### 4️⃣ Create Metadata and Performance JSON
```bash
python dataset/4_create_metadata_performance.py
```
Creates `representative_json/` with comprehensive metadata including object counts, file stats, and attribute distribution analysis (weather/scene/timeofday) for selected dataset.

### 5️⃣ Comprehensive Dataset Validation
```bash
python dataset/5_validate_dataset.py
```
Complete validation suite for limited datasets:
- Source comparison - Object/file counts match source dataset
- YOLO format validation - Structure, data.yaml, label format
- JSON metadata validation - Metadata files correctness
- Image-label integrity - All images have labels and vice versa
- Attribute distribution - Weather/scene/time diversity maintained

### 6️⃣ Compress Dataset
```bash
python dataset/6_compress_dataset.py
```
Compresses selected dataset to ZIP for distribution. Output: `bdd100k_zipped/` or `bdd100k_limited_datasets_zipped/`

### 7️⃣ Compress Test Split Only
```bash
python dataset/7_compress_test_only.py
```
Compresses only test split from full dataset as standalone package. Output: `bdd100k_test_split_zipped/`

### 8️⃣ Download Pre-Made Datasets
```bash
python dataset/8_download_extract_other_datasets.py
```
Interactive menu downloads pre-made datasets (Limited, Tuning, Tiny, Test-only) from Google Drive with automatic extraction.

---

## Utility Modules

- **`bdd100k_config.py`** - Constants, paths, configurations, representative attributes
- **`bdd100k_download.py`** - Google Drive download utilities
- **`bdd100k_validate.py`** - Validation functions
- **`bdd100k_compress.py`** - ZIP compression utilities

---

## Quick Start

**Full workflow:**
```bash
python dataset/1_download_extract_main_dataset.py
python dataset/2_convert_labels_to_yolo.py
python dataset/2.5_validate_conversion.py  # Verify conversion
python dataset/3_create_limited_datasets.py  # Create subsets
python dataset/4_create_metadata_performance.py
python dataset/5_validate_dataset.py  # Comprehensive validation
```

**Download pre-made datasets:**
```bash
python dataset/8_download_extract_other_datasets.py
```

## BDD100K Classes

10 classes: `person`, `rider`, `car`, `truck`, `bus`, `train`, `motor`, `bike`, `traffic light`, `traffic sign`

## Representative Attributes

Sampling ensures diversity across:
- **Weather**: clear, foggy, overcast, partly cloudy, rainy, snowy, undefined
- **Scene**: city street, gas stations, highway, parking lot, residential, tunnel, undefined
- **Time**: daytime, night, dawn/dusk, undefined

## Requirements

```bash
pip install gdown tqdm pyyaml
```
