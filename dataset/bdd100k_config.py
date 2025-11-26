"""
BDD100K Dataset Configuration and Constants.

This module contains all configuration constants, class definitions, URLs,
and paths used throughout the BDD100K to YOLO conversion pipeline.
"""

from pathlib import Path


# BDD100K attribute values for representative sampling
REPRESENTATIVE_ATTRIBUTES = {
    'weather': ['clear', 'foggy', 'overcast', 'partly cloudy', 'rainy', 'snowy', 'undefined'],
    'scene': ['city street', 'gas stations', 'highway', 'parking lot', 'residential', 'tunnel', 'undefined'],
    'timeofday': ['daytime', 'night', 'dawn/dusk', 'undefined']
}

# Define base paths - project root (parent of dataset/ folder)
BASE_DIR = Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / "bdd_100k_source"
YOLO_DATASET_ROOT = BASE_DIR / 'bdd100k_yolo'

# Temporary directories for extraction
TMP_IMAGES_DIR = BASE_DIR / 'bdd100k_tmp_images'
TMP_LABELS_DIR = BASE_DIR / 'bdd100k_tmp_labels'

# BDD100K object detection classes (10 classes)
# CRITICAL: These names must match exactly what's in the BDD100K JSON files
# Validated against actual dataset (10K samples analyzed)
# Note: BDD100K also has segmentation classes (area/*, lane/*) which are not included here
BDD100K_CLASSES = [
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motor',
    'bike',
    'traffic light',
    'traffic sign'
]

# BDD100K standard image dimensions
# All images in the dataset are 1280x720 pixels
BDD100K_IMAGE_WIDTH = 1280
BDD100K_IMAGE_HEIGHT = 720

# Create class name to index mapping
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(BDD100K_CLASSES)}

# BDD100K download URLs
BDD100K_URLS = {
    'website': 'http://bdd-data.berkeley.edu/',
    'images': 'https://dl.cv.ethz.ch/bdd100k/data/100k_images.zip',
    'labels': 'https://dl.cv.ethz.ch/bdd100k/data/bdd100k_labels_release.zip'
}

# Google Drive file IDs for faster downloads (mirrors)
BDD100K_GDOWN_IDS = {
    'images': '1yHEpeEdRDAz5yH4pbo4o1SvzKzGKRaLS',
    'labels': '1Gh_5g-MAx1R5X3eNsTTdz_GPialECz0L'
}

# Multiple limited dataset configurations
# SEQUENTIAL PROCESSING: Each dataset is created from the previous one
# Config 1 → from Full dataset
# Config 2 → from Config 1 (completed)
# Config 3 → from Config 1 (NOT from Config 2)
# This ensures: Config 3 ⊆ Config 2 ⊆ Config 1
LIMITED_DATASET_CONFIGS = [
    {
        'id': 1,
        'name': 'bdd100k_yolo_limited',
        'description': 'Balanced limited dataset - 30-40% coverage (target: ~25K train images)',
        'samples_per_attribute_combo': 1000,       # Increased to get more samples per combo
        'min_samples_per_class': 2500,            # Increased from 1500 to ensure class coverage
        'min_samples_per_attribute_value': 1000,  # Increased from 750
        'min_samples_per_class_attribute_combo': 1000,  # Increased from 500
        'splits': ['train', 'val', 'test'],
        'contain_full_val_split': True,  # Val split: full 10K images
        'contain_full_test_split': True,  # Test split: full 20K images
        'source_dataset': 'full'  # Source: Full dataset (bdd100k_yolo)
    },
    {
        'id': 2,
        'name': 'bdd100k_yolo_tuning',
        'description': 'Tuning dataset - 20% coverage (target: ~14K train images)',
        'samples_per_attribute_combo': 400,       # Moderate sampling per combo
        'min_samples_per_class': 1200,            # Adequate class coverage
        'min_samples_per_attribute_value': 550,   # Moderate attribute coverage
        'min_samples_per_class_attribute_combo': 350,  # Balanced combo coverage
        'splits': ['train', 'val'],
        'contain_full_val_split': True,  # Val split: full 10K images for reliable tuning
        'contain_full_test_split': False,  # No test split for tuning
        'source_dataset': 'bdd100k_yolo_limited'  # Source: Config 1 (must exist)
    },
    {
        'id': 3,
        'name': 'bdd100k_yolo_tiny',
        'description': 'Tiny dataset - ~500 train, ~1K total (for fast testing)',
        'samples_per_attribute_combo': 5,        # Very small per combo
        'min_samples_per_class': 15,              # Minimal class coverage
        'min_samples_per_attribute_value': 10,    # Minimal attribute coverage
        'min_samples_per_class_attribute_combo': 5,  # Minimal combo coverage
        'splits': ['train', 'val', 'test'],
        'contain_full_val_split': False,  # Sampled val split
        'contain_full_test_split': False,  # Sampled test split
        'source_dataset': 'bdd100k_yolo_limited'  # Source: Config 1 (NOT Config 2)
    }
]

# BDD100K attribute definitions (for representative sampling)
BDD100K_ATTRIBUTES = {
    'weather': ['clear', 'partly cloudy', 'overcast', 'rainy', 'snowy', 'foggy'],
    'scene': ['city street', 'highway', 'residential', 'parking lot', 'gas stations', 'tunnel'],
    'timeofday': ['daytime', 'night', 'dawn/dusk']
}
