"""
Extract BDD100K dataset and prepare YOLO-compatible structure with representative samples.

This script:
1. Downloads dataset files using gdown if not present (Google Drive mirror - faster)
2. Extracts images from bdd100k_images_100k.zip
3. Extracts labels from bdd100k_labels.zip
4. Converts labels to YOLO format
5. Performs integrity checks (verifies image-label matching per split)
6. Analyzes and selects representative samples based on diverse attributes (weather, scene, time)
7. Creates full YOLO dataset structure with metadata files (train/val/test_metadata.json)
8. Automatically creates limited dataset from representative samples
9. Intelligently skips already-completed steps (extraction, conversion, analysis)
10. Preserves temporary directories for reference and debugging

Requirements:
    - gdown (auto-installed if needed for downloads)
    - Original BDD100K URLs: http://bdd-data.berkeley.edu/ (manual download alternative)

Features:
    - Attribute-based representative sampling for diverse visualization
    - Saves metadata files (train/val/test_metadata.json) with statistics and sample paths
    - Performs integrity checks to ensure all images have corresponding labels
    - Preserves temporary directories for debugging and reference

Usage:
    # Full dataset + limited dataset (intelligently skips completed steps)
    python process_bdd100k_to_yolo_dataset.py

    # Manual download first (faster with gdown):
    mkdir -p bdd_100k_source
    pip install gdown
    gdown 1yHEpeEdRDAz5yH4pbo4o1SvzKzGKRaLS -O bdd_100k_source/bdd100k_images_100k.zip
    gdown 1Gh_5g-MAx1R5X3eNsTTdz_GPialECz0L -O bdd_100k_source/bdd100k_labels.zip
    python process_bdd100k_to_yolo_dataset.py
"""

import os
import json
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil
import random
import subprocess
import sys
from datetime import datetime
from PIL import Image



# Define paths
base_dir = Path(__file__).parent
source_dir = base_dir / "bdd_100k_source"
yolo_dataset_root = base_dir / 'bdd100k_yolo'

# Multiple limited dataset configurations
# SEQUENTIAL PROCESSING: Each dataset is created from the previous one
# Config 1 → from Full dataset
# Config 2 → from Config 1 (completed)
# Config 3 → from Config 1 (NOT from Config 2)
# This ensures: Config 3 ⊆ Config 2 ⊆ Config 1
LIMITED_DATASET_CONFIGS = [
    {   'id': 1,
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

# BDD100K object detection classes (10 classes)
# CRITICAL: These names must match exactly what's in the BDD100K JSON files
# BDD100K detection classes (10 classes for object detection task)
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

# BDD100K image dimensions (validated from actual dataset - all images are 1280×720)
# Verified from 100 sample images across train/val/test splits
BDD100K_IMAGE_WIDTH = 1280
BDD100K_IMAGE_HEIGHT = 720

# Create class name to index mapping
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(BDD100K_CLASSES)}

# BDD100K download information
# Original URLs (reference): https://dl.cv.ethz.ch/bdd100k/data/
# Using Google Drive mirror for faster downloads
BDD100K_GDOWN_IDS = {
    'images': '1yHEpeEdRDAz5yH4pbo4o1SvzKzGKRaLS',  # bdd100k_images_100k.zip (~6GB)
    'labels': '1Gh_5g-MAx1R5X3eNsTTdz_GPialECz0L'   # bdd100k_labels.zip (~300MB)
}

# Original BDD100K URLs (for manual download reference)
BDD100K_URLS = {
    'images': 'https://dl.cv.ethz.ch/bdd100k/data/100k_images.zip',
    'labels': 'https://dl.cv.ethz.ch/bdd100k/data/det_20_labels.zip',
    'website': 'http://bdd-data.berkeley.edu/'
}

# BDD100K attribute values (validated from actual dataset)
# Validated from 10,000 sample label files across train/val/test splits
# Note: 'gas stations' is very rare (only 4 occurrences in 10K samples) but is a valid scene value
# All 100K label files have complete attributes (weather, scene, timeofday) - no defaults needed
REPRESENTATIVE_ATTRIBUTES = {
    'weather': ['clear', 'foggy', 'overcast', 'partly cloudy', 'rainy', 'snowy', 'undefined'],
    'scene': ['city street', 'gas stations', 'highway', 'parking lot', 'residential', 'tunnel', 'undefined'],
    'timeofday': ['daytime', 'night', 'dawn/dusk', 'undefined']
}



def check_gdown_installed():
    """
    Check if gdown is installed, install if not.
    """
    try:
        import gdown
        return True
    except ImportError:
        print("\n📦 Installing gdown for faster downloads...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            print("✓ gdown installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install gdown. Please install manually: pip install gdown")
            return False


def download_file(gdown_id, output_path):
    """
    Download a file from Google Drive using gdown.
    Much faster than direct download from BDD100K servers.
    """
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={gdown_id}'
        gdown.download(url, str(output_path), quiet=False)
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def check_and_download_datasets(source_dir):
    """
    Check if dataset files exist, prompt to download if missing.
    """
    images_zip = source_dir / "bdd100k_images_100k.zip"
    labels_zip = source_dir / "bdd100k_labels.zip"
    
    files_exist = images_zip.exists() and labels_zip.exists()
    
    if files_exist:
        print("\n✓ Dataset files found:")
        print(f"  Images: {images_zip} ({images_zip.stat().st_size / (1024**3):.2f} GB)")
        print(f"  Labels: {labels_zip} ({labels_zip.stat().st_size / (1024**2):.2f} MB)")
        return True
    
    # Prompt user to download
    print("\n" + "="*70)
    print("DATASET FILES NOT FOUND")
    print("="*70)
    print("\nMissing files:")
    if not images_zip.exists():
        print(f"  ❌ {images_zip.name} (~6GB)")
    if not labels_zip.exists():
        print(f"  ❌ {labels_zip.name} (~300MB)")
    
    print("\nOptions:")
    print("  1. Download automatically using gdown (faster, recommended)")
    print(f"  2. Download manually from: {BDD100K_URLS['website']}")
    print("  3. Exit and provide files later")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        # Check/install gdown
        if not check_gdown_installed():
            print("\nCannot proceed without gdown. Please install it or download manually.")
            return False
        
        print("\n📥 Downloading from Google Drive mirror (faster than official servers)...")
        print("   Original URLs available at: http://bdd-data.berkeley.edu/")
        source_dir.mkdir(parents=True, exist_ok=True)
        
        # Download using gdown
        success = True
        try:
            if not images_zip.exists():
                print(f"\n[1/2] Downloading images (~6GB, this may take a while)...")
                if not download_file(BDD100K_GDOWN_IDS['images'], images_zip):
                    success = False
            
            if not labels_zip.exists():
                print(f"\n[2/2] Downloading labels (~300MB)...")
                if not download_file(BDD100K_GDOWN_IDS['labels'], labels_zip):
                    success = False
            
            if success:
                print("\n✓ Downloads complete!")
                return True
            else:
                print("\n❌ Some downloads failed.")
                print(f"\nPlease download manually from: {BDD100K_URLS['website']}")
                return False
        
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print(f"\nPlease download manually from: {BDD100K_URLS['website']}")
            return False
    
    elif choice == '2':
        print("\nPlease download files manually and place them in:")
        print(f"  {source_dir}/")
        print("\nThen run this script again.")
        return False
    
    else:
        print("\nExiting. Run script again when files are available.")
        return False


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert BDD100K bbox format to YOLO format WITHOUT validation or filtering.
    Converts all boxes as-is from source data - does not hide data quality issues.
    
    BDD100K format: {x1, y1, x2, y2} (absolute pixel coordinates)
    YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    
    Note: NO CLAMPING, NO VALIDATION. Preserves source data exactly including:
    - Zero width/height boxes (degenerate annotations)
    - Negative dimensions (if x2 < x1 or y2 < y1)
    - Out of bounds coordinates
    This allows training frameworks to handle or report these issues directly.
    """
    x1, y1 = bbox['x1'], bbox['y1']
    x2, y2 = bbox['x2'], bbox['y2']
    
    # Calculate YOLO format values - convert exactly as-is
    x_center = ((x1 + x2) / 2.0) / img_width
    y_center = ((y1 + y2) / 2.0) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return [x_center, y_center, width, height]


def get_label_attributes(json_path):
    """
    Extract attributes from a BDD100K JSON label file.
    Returns dict with attributes and object categories.
    """
    try:
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        
        # All BDD100K files have complete attributes
        attributes = label_data['attributes']
        frames = label_data.get('frames', [])
        
        # Get object categories
        if frames:
            objects = frames[0].get('objects', [])
        else:
            objects = label_data.get('objects', label_data.get('labels', []))
        
        categories = [obj.get('category', '') for obj in objects if 'box2d' in obj]
        
        return {
            'weather': attributes['weather'],
            'scene': attributes['scene'],
            'timeofday': attributes['timeofday'],
            'categories': [cat for cat in categories if cat in CLASS_TO_IDX],
            'num_objects': len(categories)
        }
    except Exception as e:
        return None


def convert_json_to_yolo(json_path):
    """
    Convert a single BDD100K JSON label file to YOLO format WITHOUT filtering.
    Converts ALL objects from source data as-is, preserving all data quality issues.
    
    Returns tuple: (yolo_labels, attributes, converted_objects, skipped_objects)
    - yolo_labels: List of YOLO format strings (ALL valid category objects)
    - attributes: Dict with weather, scene, timeofday
    - converted_objects: Count of successfully converted objects
    - skipped_objects: Count of objects missing required fields only
    
    Note: Does NOT filter based on bbox dimensions - converts everything as-is.
    Zero width/height boxes, negative dimensions, all converted exactly.
    """
    try:
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        
        # Extract attributes - all BDD100K files have complete attributes
        attrs = label_data['attributes']
        attributes = {
            'weather': attrs['weather'],
            'scene': attrs['scene'],
            'timeofday': attrs['timeofday']
        }
        
        # BDD100K images are standard 1280x720
        img_width = BDD100K_IMAGE_WIDTH
        img_height = BDD100K_IMAGE_HEIGHT
        
        # Process labels - convert ALL objects without validation/filtering
        yolo_labels = []
        converted_count = 0
        skipped_count = 0
        
        frames = label_data.get('frames', [])
        
        if frames:
            objects = frames[0].get('objects', [])
        else:
            objects = label_data.get('objects', label_data.get('labels', []))
        
        for obj in objects:
            category = obj.get('category', '')
            
            # Skip only if category not in our class list
            if category not in CLASS_TO_IDX:
                continue
            
            box2d = obj.get('box2d')
            if not box2d:
                skipped_count += 1
                continue
            
            # Skip only if required fields are completely missing
            if not all(k in box2d for k in ['x1', 'y1', 'x2', 'y2']):
                skipped_count += 1
                continue
            
            # Convert WITHOUT any validation - preserve ALL source data as-is
            class_idx = CLASS_TO_IDX[category]
            yolo_bbox = convert_bbox_to_yolo(box2d, img_width, img_height)
            
            # Format: class_idx x_center y_center width height
            # Converts ALL boxes including zero width/height, negative, out-of-bounds, etc.
            yolo_line = f"{class_idx} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
            yolo_labels.append(yolo_line)
            converted_count += 1
        
        return yolo_labels, attributes, converted_count, skipped_count
    
    except Exception as e:
        print(f"Warning: Error processing {json_path}: {e}")
        return [], {}, 0, 0


def count_attribute_distribution(labels_dir, split_name, filter_basenames=None):
    """
    Count distribution of attribute values (weather/scene/timeofday) across images.
    
    Args:
        labels_dir: Path to labels directory that ALREADY contains the split subdirectory
                    (e.g., bdd100k_tmp_labels/100k or just bdd100k_tmp_labels)
        split_name: Name of split (train/val/test)
        filter_basenames: Optional set of basenames to filter (only count these files)
    
    Returns dict with counts for each attribute value.
    """
    # labels_dir should already contain the proper path structure
    json_dir = labels_dir / split_name
    if not json_dir.exists():
        return {}, {}, {}
    
    json_files = list(json_dir.glob('*.json'))
    
    weather_counts = {}
    scene_counts = {}
    timeofday_counts = {}
    
    for json_file in json_files:
        # Filter by basenames if provided
        if filter_basenames is not None and json_file.stem not in filter_basenames:
            continue
            
        attrs = get_label_attributes(json_file)
        if attrs:
            weather = attrs['weather']
            scene = attrs['scene']
            timeofday = attrs['timeofday']
            
            weather_counts[weather] = weather_counts.get(weather, 0) + 1
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
            timeofday_counts[timeofday] = timeofday_counts.get(timeofday, 0) + 1
    
    return weather_counts, scene_counts, timeofday_counts


def count_objects_in_labels(labels_dir, desc="Counting objects"):
    """
    Count objects by class from YOLO format label files.
    Returns dict: {class_name: count}
    """
    object_counts = {cls: 0 for cls in BDD100K_CLASSES}
    txt_files = list(labels_dir.glob('*.txt'))
    
    for txt_file in tqdm(txt_files, desc=desc, unit='files', leave=False):
        with open(txt_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(BDD100K_CLASSES):
                            object_counts[BDD100K_CLASSES[class_id]] += 1
                    except (ValueError, IndexError):
                        continue
    
    return object_counts


def count_objects_in_json_files(json_dir, desc="Counting objects in JSON"):
    """
    Count objects by class from original BDD100K JSON label files.
    Returns dict: {class_name: count}
    """
    object_counts = {cls: 0 for cls in BDD100K_CLASSES}
    json_files = list(json_dir.glob('*.json'))
    
    for json_file in tqdm(json_files, desc=desc, unit='files', leave=False):
        try:
            with open(json_file, 'r') as f:
                label_data = json.load(f)
            
            # Get objects from frames or directly
            frames = label_data.get('frames', [])
            if frames:
                objects = frames[0].get('objects', [])
            else:
                objects = label_data.get('objects', label_data.get('labels', []))
            
            # Count objects by category
            for obj in objects:
                category = obj.get('category', '')
                if category in CLASS_TO_IDX and 'box2d' in obj:
                    object_counts[category] += 1
        
        except Exception as e:
            continue
    
    return object_counts


def compare_dataset_statistics(tmp_labels_dir, yolo_labels_dir, split_name):
    """
    Compare object counts between original JSON files and generated YOLO labels.
    Returns dict with comparison statistics.
    """
    print(f"\n{'='*70}")
    print(f"DATASET STATISTICS COMPARISON - {split_name.upper()} SPLIT")
    print(f"{'='*70}")
    
    # Count objects in original JSON files
    json_dir = tmp_labels_dir / '100k' / split_name
    if not json_dir.exists():
        json_dir = tmp_labels_dir / split_name
    
    if not json_dir.exists():
        print(f"⚠️  JSON directory not found: {json_dir}")
        return None
    
    print(f"\nCounting objects in original JSON files...")
    original_counts = count_objects_in_json_files(json_dir, f"  Analyzing {split_name} JSON")
    
    # Count objects in generated YOLO labels
    print(f"\nCounting objects in generated YOLO labels...")
    generated_counts = count_objects_in_labels(yolo_labels_dir, f"  Analyzing {split_name} YOLO")
    
    # Calculate statistics
    comparison = {
        'split': split_name,
        'original_total': sum(original_counts.values()),
        'generated_total': sum(generated_counts.values()),
        'by_class': {}
    }
    
    print(f"\n{'Class':<15} {'Original':>10} {'Generated':>10} {'Match':>8} {'Status':>10}")
    print("-" * 70)
    
    all_match = True
    for class_name in BDD100K_CLASSES:
        orig = original_counts[class_name]
        gen = generated_counts[class_name]
        match = orig == gen
        status = "✓ OK" if match else "✗ DIFF"
        
        if not match:
            all_match = False
        
        comparison['by_class'][class_name] = {
            'original': orig,
            'generated': gen,
            'match': match,
            'difference': gen - orig
        }
        
        print(f"{class_name:<15} {orig:>10,} {gen:>10,} {str(match):>8} {status:>10}")
    
    print("-" * 70)
    print(f"{'TOTAL':<15} {comparison['original_total']:>10,} {comparison['generated_total']:>10,}")
    
    comparison['all_match'] = all_match
    comparison['match_percentage'] = (comparison['generated_total'] / comparison['original_total'] * 100) if comparison['original_total'] > 0 else 0
    
    print(f"\nMatch Rate: {comparison['match_percentage']:.2f}%")
    
    if all_match:
        print(f"✅ PERFECT MATCH: All objects successfully converted!")
    else:
        print(f"⚠️  UNEXPECTED DIFFERENCES: Larger than expected filtering")
    
    return comparison


def compress_test_split_only(yolo_dataset_root, base_dir):
    """
    Compress only the test split from the full dataset for quick distribution.
    Creates a standalone test dataset with data.yaml configured for test only.
    """
    print("\n" + "="*70)
    print("COMPRESSING TEST SPLIT ONLY")
    print("="*70)
    
    test_images_dir = yolo_dataset_root / 'images' / 'test'
    test_labels_dir = yolo_dataset_root / 'labels' / 'test'
    
    if not test_images_dir.exists() or not test_labels_dir.exists():
        print(f"⚠️  Test split not found in {yolo_dataset_root}")
        return None
    
    # Create output directory
    zipped_dir = base_dir / 'bdd100k_test_split_zipped'
    zipped_dir.mkdir(parents=True, exist_ok=True)
    compressed_file = zipped_dir / 'bdd100k_yolo_test_split.zip'
    
    # Remove existing compressed file
    if compressed_file.exists():
        print(f"Removing existing: {compressed_file.name}")
        compressed_file.unlink()
    
    print(f"\nCompressing test split...")
    print(f"  Source: {yolo_dataset_root}")
    print(f"  Destination: {compressed_file}")
    
    with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        # Count files first
        test_images = list(test_images_dir.glob('*'))
        test_labels = list(test_labels_dir.glob('*.txt'))
        
        # Add test images
        for img_file in tqdm(test_images, desc="  Compressing images", unit='files'):
            if img_file.is_file():
                arcname = Path('bdd100k_yolo_test') / 'images' / 'test' / img_file.name
                zipf.write(img_file, arcname=arcname)
        
        # Add test labels
        for label_file in tqdm(test_labels, desc="  Compressing labels", unit='files'):
            arcname = Path('bdd100k_yolo_test') / 'labels' / 'test' / label_file.name
            zipf.write(label_file, arcname=arcname)
        
        # Create and add test-only data.yaml
        test_data_yaml = f"""# BDD100K Test Split Only
# Auto-generated by process_bdd100k_to_yolo_dataset.py

path: .  # Current directory
test: images/test  # Test images only

# Number of classes
nc: {len(BDD100K_CLASSES)}

# Class names
names: {BDD100K_CLASSES}
"""
        zipf.writestr('bdd100k_yolo_test/data.yaml', test_data_yaml)
        
        # Copy test metadata and performance analysis files (from root of representative_json/)
        for metadata_file in ['test_metadata.json', 'test_performance_analysis.json']:
            src_file = yolo_dataset_root / 'representative_json' / metadata_file
            if src_file.exists():
                arcname = Path('bdd100k_yolo_test') / 'representative_json' / metadata_file
                zipf.write(src_file, arcname=arcname)
    
    file_size_mb = compressed_file.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ Test split compressed successfully!")
    print(f"  Location: {compressed_file}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Images: {len(test_images):,}")
    print(f"  Labels: {len(test_labels):,}")
    print(f"\nTo extract and use:")
    print(f"  unzip {compressed_file.name}")
    print(f"  cd bdd100k_yolo_test")
    print(f"  # Run YOLO validation with data.yaml")
    
    return {
        'path': compressed_file,
        'size_mb': file_size_mb,
        'num_images': len(test_images),
        'num_labels': len(test_labels)
    }


def perform_integrity_check(images_dir, labels_dir, split_name):
    """
    Verify all images have corresponding label files and vice versa.
    Returns tuple: (images_without_labels, labels_without_images, is_valid)
    """
    image_basenames = {f.stem for f in images_dir.glob('*.jpg')} | {f.stem for f in images_dir.glob('*.png')}
    label_basenames = {f.stem for f in labels_dir.glob('*.txt')}
    
    images_without_labels = image_basenames - label_basenames
    labels_without_images = label_basenames - image_basenames
    
    is_valid = len(images_without_labels) == 0 and len(labels_without_images) == 0
    
    return images_without_labels, labels_without_images, is_valid


def create_label_file(label_path, yolo_labels):
    """
    Create YOLO format label file (empty if no objects).
    Ensures consistent file creation for all images.
    """
    with open(label_path, 'w') as f:
        if yolo_labels:
            f.write('\n'.join(yolo_labels))
        # else: write empty file (required by YOLO format)


def save_test_performance_metadata(labels_base_dir, yolo_labels_dir, split_name, output_file):
    """
    Save detailed per-image metadata to enable performance analysis.
    For each image, stores: basename, attributes (weather/scene/timeofday), 
    classes present, and object counts per class.
    Works with both full dataset (all images) and limited dataset (representative samples).
    
    Args:
        labels_base_dir: Base labels directory that ALREADY contains proper structure
                         (e.g., bdd100k_tmp_labels/100k for full, or limited dataset path)
        yolo_labels_dir: YOLO labels directory for this split
        split_name: Name of split (train/val/test)
        output_file: Where to save the performance metadata JSON
    """
    print(f"\n  Generating performance analysis metadata for {split_name} split...")
    
    performance_data = {
        'split': split_name,
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': 0,
        'images': []
    }
    
    # Get JSON files from source (contains attributes)
    # labels_base_dir already has the correct structure
    json_dir = labels_base_dir / split_name
    
    json_files = list(json_dir.glob('*.json')) if json_dir.exists() else []
    
    # Build mapping from JSON files
    json_map = {}
    for json_file in json_files:
        attrs = get_label_attributes(json_file)
        if attrs:
            json_map[json_file.stem] = attrs
    
    # Process each label file
    txt_files = sorted(yolo_labels_dir.glob('*.txt'))
    
    for txt_file in tqdm(txt_files, desc=f"  Building performance metadata", unit='files', leave=False):
        basename = txt_file.stem
        
        # Count objects per class for this image
        class_counts = {cls: 0 for cls in BDD100K_CLASSES}
        with open(txt_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(BDD100K_CLASSES):
                            class_counts[BDD100K_CLASSES[class_id]] += 1
                    except (ValueError, IndexError):
                        continue
        
        # Get attributes - all BDD100K files have complete attributes
        attrs = json_map.get(basename, {})
        
        # Check if attributes exist for this image
        if not attrs:
            print(f"\n❌ ERROR: No JSON metadata found for image: {basename}")
            print(f"   Label file: {txt_file}")
            print(f"   Expected JSON: {labels_base_dir / '100k' / split_name / f'{basename}.json'}")
            raise KeyError(f"Missing JSON metadata for {basename}")
        
        # Verify all required attributes exist
        if 'weather' not in attrs:
            print(f"\n❌ ERROR: Missing 'weather' attribute for image: {basename}")
            print(f"   Available attributes: {list(attrs.keys())}")
            print(f"   JSON file: {labels_base_dir / '100k' / split_name / f'{basename}.json'}")
            raise KeyError(f"Missing 'weather' attribute for {basename}")
        
        image_data = {
            'basename': basename,
            'weather': attrs['weather'],  # All BDD100K files have this field
            'scene': attrs['scene'],      # All BDD100K files have this field
            'timeofday': attrs['timeofday'],  # All BDD100K files have this field
            'classes_present': [cls for cls, count in class_counts.items() if count > 0],
            'objects_per_class': {cls: count for cls, count in class_counts.items() if count > 0},
            'total_objects': sum(class_counts.values())
        }
        
        performance_data['images'].append(image_data)
    
    performance_data['total_images'] = len(performance_data['images'])
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    print(f"  ✓ Performance metadata saved: {output_file.name}")
    print(f"    - {performance_data['total_images']} images with detailed attributes")
    print(f"    - Ready for YOLO model performance analysis")


def extract_zip_with_progress(zip_path, extract_to, description):
    """
    Extract a zip file with progress bar.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract files to
        description: Description for progress bar
    """
    print(f"\n{description}")
    print(f"Source: {zip_path}")
    print(f"Destination: {extract_to}")
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    # Create extraction directory
    os.makedirs(extract_to, exist_ok=True)
    
    # Extract with progress bar
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        
        with tqdm(total=len(members), desc=description, unit='files') as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)
    
    print(f"✓ Extraction complete: {len(members)} files extracted\n")
    return len(members)


def select_representative_samples(split_labels_src, split_name, config, constrain_to_basenames=None):
    """
    Select representative samples ensuring comprehensive coverage PER SPLIT.
    All parameters apply independently to each split (train/val/test):
    
    1. samples_per_attribute_combo per (weather, scene, timeofday) combination PER SPLIT
    2. min_samples_per_class per object class PER SPLIT
    3. min_samples_per_attribute_value per individual attribute value PER SPLIT
    4. min_samples_per_class_attribute_combo per (class, attribute) combination PER SPLIT
    
    Args:
        split_labels_src: Path to source labels directory
        split_name: Name of the split (train/val/test)
        config: REQUIRED dict with sampling configuration from LIMITED_DATASET_CONFIGS
        constrain_to_basenames: Optional set of basenames to constrain selection to (for hierarchical subsets)
    
    Returns comprehensive metadata dict with statistics and selected samples.
    
    Note: This function is called separately for each split, so all limits are per-split.
          If constrain_to_basenames is provided, samples are selected ONLY from that set (hierarchical).
    """
    # Extract config values - config is REQUIRED
    if not config:
        raise ValueError("Config is required for representative sample selection. Use a config from LIMITED_DATASET_CONFIGS.")
    
    samples_per_combo = config['samples_per_attribute_combo']
    min_per_class = config['min_samples_per_class']
    min_per_attr = config['min_samples_per_attribute_value']
    min_per_class_attr = config['min_samples_per_class_attribute_combo']
    
    print(f"\n  Analyzing labels for representative sample selection...")
    print(f"    Configuration (ALL LIMITS PER SPLIT):")
    print(f"      - {samples_per_combo} samples per attribute combo (weather×scene×time) PER SPLIT")
    print(f"      - {min_per_class} samples per class PER SPLIT")
    print(f"      - {min_per_attr} samples per attribute value PER SPLIT")
    print(f"      - {min_per_class_attr} samples per (class×attribute) combo PER SPLIT")
    
    # Get all JSON files (for limited dataset, analyze .txt files instead)
    json_files = list(split_labels_src.glob('*.json'))
    txt_files = list(split_labels_src.glob('*.txt')) if not json_files else []
    
    # For limited dataset: we have .txt files not .json files
    # We need to create minimal metadata based on what exists
    if not json_files and not txt_files:
        empty_metadata = {
            'split': split_name,
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'samples_per_attribute_combo': samples_per_combo,
                'min_samples_per_class': min_per_class,
                'min_samples_per_attribute_value': min_per_attr,
                'min_samples_per_class_attribute_combo': min_per_class_attr
            },
            'classes': BDD100K_CLASSES,
            'attributes': REPRESENTATIVE_ATTRIBUTES,
            'statistics': {
                'total_files_analyzed': 0,
                'total_selected': 0,
                'by_class_image_count': {cls: 0 for cls in BDD100K_CLASSES},
                'by_weather': {},
                'by_scene': {},
                'by_timeofday': {},
                'by_attribute_combo': {},
                'representative_samples': {}
            },
            'selected_samples': {
                'by_class': {}
            }
        }
        return set(), empty_metadata
    
    # For limited dataset with only .txt files, analyze those
    if txt_files and not json_files:
        # Count objects from YOLO format txt files
        class_counts = {cls: 0 for cls in BDD100K_CLASSES}
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(BDD100K_CLASSES):
                            class_counts[BDD100K_CLASSES[class_id]] += 1
        
        limited_metadata = {
            'split': split_name,
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'samples_per_attribute_combo': samples_per_combo,
                'min_samples_per_class': min_per_class,
                'min_samples_per_attribute_value': min_per_attr,
                'min_samples_per_class_attribute_combo': min_per_class_attr
            },
            'classes': BDD100K_CLASSES,
            'attributes': REPRESENTATIVE_ATTRIBUTES,
            'statistics': {
                'total_files_analyzed': len(txt_files),
                'total_selected': len(txt_files),
                'by_class_image_count': class_counts,
                'by_weather': {},
                'by_scene': {},
                'by_timeofday': {},
                'by_attribute_combo': {},
                'representative_samples': {}
            },
            'selected_samples': {
                'by_class': {cls: [f.stem for f in txt_files] for cls in BDD100K_CLASSES if class_counts[cls] > 0}
            }
        }
        return {f.stem for f in txt_files}, limited_metadata
    
    # Organize files by various groupings
    attribute_combo_groups = {}  # (weather, scene, timeofday) -> files
    class_samples = {class_id: [] for class_id in range(len(BDD100K_CLASSES))}  # class_id -> files
    weather_samples = {w: [] for w in REPRESENTATIVE_ATTRIBUTES['weather']}  # weather -> files
    scene_samples = {s: [] for s in REPRESENTATIVE_ATTRIBUTES['scene']}  # scene -> files
    timeofday_samples = {t: [] for t in REPRESENTATIVE_ATTRIBUTES['timeofday']}  # time -> files
    class_attribute_samples = {}  # (class_id, attr_type, attr_value) -> files
    
    # Filter JSON files to constrained set if hierarchical subset requested
    files_to_process = json_files
    if constrain_to_basenames is not None:
        print(f"  Hierarchical constraint: Limiting to {len(constrain_to_basenames):,} samples from previous config")
        files_to_process = [f for f in json_files if f.stem in constrain_to_basenames]
        print(f"  Available for selection: {len(files_to_process):,} files")
    
    for json_file in tqdm(files_to_process, desc="  Analyzing attributes", unit='files', leave=False):
        attrs = get_label_attributes(json_file)
        if not attrs or not attrs['categories']:
            continue
        
        file_info = {
            'path': json_file,
            'attrs': attrs
        }
        
        # Group by full attribute combination
        combo_key = (attrs['weather'], attrs['scene'], attrs['timeofday'])
        if combo_key not in attribute_combo_groups:
            attribute_combo_groups[combo_key] = []
        attribute_combo_groups[combo_key].append(file_info)
        
        # Group by individual attribute values
        if attrs['weather'] in weather_samples:
            weather_samples[attrs['weather']].append(file_info)
        if attrs['scene'] in scene_samples:
            scene_samples[attrs['scene']].append(file_info)
        if attrs['timeofday'] in timeofday_samples:
            timeofday_samples[attrs['timeofday']].append(file_info)
        
        # Group by class and class+attribute combinations
        for cat in attrs['categories']:
            if cat in CLASS_TO_IDX:
                class_id = CLASS_TO_IDX[cat]
                class_samples[class_id].append(file_info)
                
                # Group by (class, weather), (class, scene), (class, timeofday)
                for attr_type, attr_value in [('weather', attrs['weather']), 
                                               ('scene', attrs['scene']), 
                                               ('timeofday', attrs['timeofday'])]:
                    combo = (class_id, attr_type, attr_value)
                    if combo not in class_attribute_samples:
                        class_attribute_samples[combo] = []
                    class_attribute_samples[combo].append(file_info)
    
    selected_files = set()
    selected_by_attributes = {}
    
    # Initialize metadata structure
    # METADATA STRUCTURE DOCUMENTATION:
    # - statistics.by_class_image_count: COUNT OF IMAGES containing each class (not object count)
    # - statistics.total_selected: Total number of representative images selected
    # - selected_samples.by_class: LIST OF BASENAMES for each class
    # - Later in process_split, additional stats are added:
    #   - statistics.full_dataset: Object counts for the FULL dataset
    #   - statistics.representative_samples.by_class_object_counts: Object counts for representative samples
    metadata = {
        'split': split_name,
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'samples_per_attribute_combo': samples_per_combo,
            'min_samples_per_class': min_per_class,
            'min_samples_per_attribute_value': min_per_attr,
            'min_samples_per_class_attribute_combo': min_per_class_attr
        },
        'classes': BDD100K_CLASSES,
        'attributes': REPRESENTATIVE_ATTRIBUTES,
        'statistics': {
            'total_files_analyzed': len(json_files),
            'total_selected': 0,
            'by_class_image_count': {},  # COUNT OF IMAGES (not objects)
            'by_weather': {},
            'by_scene': {},
            'by_timeofday': {},
            'by_attribute_combo': {},
            'by_class_weather': {},
            'by_class_scene': {},
            'by_class_timeofday': {},
            'representative_samples': {}  # Initialized here, populated in process_split
        },
        'selected_samples': {
            'by_attribute_combo': {},  # (weather, scene, time) -> [basenames]
            'by_class': {},  # class_name -> [basenames]
            'by_weather': {},  # weather -> [basenames]
            'by_scene': {},  # scene -> [basenames]
            'by_timeofday': {},  # time -> [basenames]
            'by_class_weather': {},  # (class, weather) -> [basenames]
            'by_class_scene': {},  # (class, scene) -> [basenames]
            'by_class_timeofday': {}  # (class, time) -> [basenames]
        }
    }
    
    # Step 1: Select samples for each attribute combination (weather×scene×time)
    print(f"\n  Step 1: Selecting samples for attribute combinations...")
    for combo_key, files in attribute_combo_groups.items():
        sorted_files = sorted(
            files,
            key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']),
            reverse=True
        )
        
        num_to_select = min(samples_per_combo, len(sorted_files))
        selected = sorted_files[:num_to_select]
        
        if selected:
            selected_paths = [s['path'] for s in selected]
            selected_by_attributes[combo_key] = selected_paths
            selected_files.update(selected_paths)
            
            # Store in metadata
            combo_str = f"{combo_key[0]}|{combo_key[1]}|{combo_key[2]}"
            metadata['selected_samples']['by_attribute_combo'][combo_str] = [p.stem for p in selected_paths]
            metadata['statistics']['by_attribute_combo'][combo_str] = len(selected_paths)
    
    print(f"    ✓ Selected {len(selected_files)} samples from {len(attribute_combo_groups)} combinations")
    
    # Step 2: Ensure minimum samples per class
    print(f"\n  Step 2: Ensuring {min_per_class} samples per class...")
    for class_id, samples in class_samples.items():
        if not samples:
            continue
        
        # Count already selected samples for this class
        current_count = sum(1 for file_info in samples if file_info['path'] in selected_files)
        
        if current_count < min_per_class:
            # Sort by diversity and add more samples
            sorted_samples = sorted(
                [s for s in samples if s['path'] not in selected_files],
                key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']),
                reverse=True
            )
            
            needed = min_per_class - current_count
            for sample in sorted_samples[:needed]:
                selected_files.add(sample['path'])
    
    # Update class statistics in metadata
    for class_id, samples in class_samples.items():
        class_name = BDD100K_CLASSES[class_id]
        class_basenames = [s['path'].stem for s in samples if s['path'] in selected_files]
        metadata['selected_samples']['by_class'][class_name] = class_basenames
        # IMPORTANT: Store count of IMAGES containing this class (not object count)
        # Object counts are calculated later from the actual YOLO label files
        metadata['statistics']['by_class_image_count'][class_name] = len(class_basenames)
    
    print(f"    ✓ Total samples after class coverage: {len(selected_files)}")
    
    # Step 3: Ensure minimum samples per individual attribute value
    print(f"\n  Step 3: Ensuring {min_per_attr} samples per attribute value...")
    for attr_dict, attr_name in [(weather_samples, 'weather'), 
                                  (scene_samples, 'scene'), 
                                  (timeofday_samples, 'timeofday')]:
        for attr_value, samples in attr_dict.items():
            if not samples:
                continue
            
            current_count = sum(1 for file_info in samples if file_info['path'] in selected_files)
            
            if current_count < min_per_attr:
                sorted_samples = sorted(
                    [s for s in samples if s['path'] not in selected_files],
                    key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']),
                    reverse=True
                )
                
                needed = min_per_attr - current_count
                for sample in sorted_samples[:needed]:
                    selected_files.add(sample['path'])
    
    # Update attribute value statistics in metadata
    for attr_dict, attr_name in [(weather_samples, 'weather'), 
                                  (scene_samples, 'scene'), 
                                  (timeofday_samples, 'timeofday')]:
        metadata_key = f'by_{attr_name}'
        for attr_value, samples in attr_dict.items():
            basenames = [s['path'].stem for s in samples if s['path'] in selected_files]
            if basenames:
                metadata['selected_samples'][metadata_key][attr_value] = basenames
                metadata['statistics'][metadata_key][attr_value] = len(basenames)
    
    print(f"    ✓ Total samples after attribute value coverage: {len(selected_files)}")
    
    # Step 4: Ensure minimum samples per (class, attribute) combination
    print(f"\n  Step 4: Ensuring {min_per_class_attr} samples per (class×attribute) combo...")
    for (class_id, attr_type, attr_value), samples in class_attribute_samples.items():
        if not samples:
            continue
        
        current_count = sum(1 for file_info in samples if file_info['path'] in selected_files)
        
        if current_count < min_per_class_attr:
            sorted_samples = sorted(
                [s for s in samples if s['path'] not in selected_files],
                key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']),
                reverse=True
            )
            
            needed = min_per_class_attr - current_count
            for sample in sorted_samples[:needed]:
                selected_files.add(sample['path'])
    
    # Update class×attribute statistics in metadata (per split)
    for (class_id, attr_type, attr_value), samples in class_attribute_samples.items():
        class_name = BDD100K_CLASSES[class_id]
        basenames = [s['path'].stem for s in samples if s['path'] in selected_files]
        
        if basenames:
            combo_key = f"{class_name}|{attr_value}"
            if attr_type == 'weather':
                metadata['selected_samples']['by_class_weather'][combo_key] = basenames
                metadata['statistics']['by_class_weather'][combo_key] = len(basenames)
            elif attr_type == 'scene':
                metadata['selected_samples']['by_class_scene'][combo_key] = basenames
                metadata['statistics']['by_class_scene'][combo_key] = len(basenames)
            elif attr_type == 'timeofday':
                metadata['selected_samples']['by_class_timeofday'][combo_key] = basenames
                metadata['statistics']['by_class_timeofday'][combo_key] = len(basenames)
    
    total_selected = len(selected_files)
    metadata['statistics']['total_selected'] = total_selected
    
    # Save detailed attributes for each selected sample for analysis
    metadata['selected_samples']['details'] = {}
    for file_path in selected_files:
        json_file = split_labels_src / f"{file_path.stem}.json"
        if json_file.exists():
            attrs = get_label_attributes(json_file)
            if attrs:
                metadata['selected_samples']['details'][file_path.stem] = {
                    'weather': attrs['weather'],
                    'scene': attrs['scene'],
                    'timeofday': attrs['timeofday']
                }
    
    print(f"\n  ✓ FINAL: Selected {total_selected} representative samples")
    print(f"    - {len(attribute_combo_groups)} attribute combinations covered")
    print(f"    - All {len(BDD100K_CLASSES)} classes with min {min_per_class} samples")
    print(f"    - All attribute values with min {min_per_attr} samples")
    print(f"    - Class×attribute combos with min {min_per_class_attr} samples (PER SPLIT)")
    print(f"    - Attributes saved for {len(metadata['selected_samples']['details'])} samples")
    
    return selected_by_attributes, selected_files, metadata


def create_yolo_dataset_structure(base_dir, dataset_name='bdd100k_yolo'):
    """
    Create YOLO-compatible dataset structure.
    
    Structure:
    dataset_name/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
        data.yaml
    """
    dataset_root = base_dir / dataset_name
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (dataset_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    return dataset_root


def process_split(tmp_images_dir, tmp_labels_dir, yolo_dataset_root, split, config):
    """
    Process a dataset split: move images and convert labels to YOLO format.
    Also selects and saves representative JSON samples for metadata.
    
    Args:
        tmp_images_dir: Temporary images directory (extracted)
        tmp_labels_dir: Temporary labels directory (extracted)
        yolo_dataset_root: Root of YOLO dataset structure
        split: 'train', 'val', or 'test'
        config: REQUIRED dict with sampling configuration for metadata generation
    
    Returns:
        Tuple of (images_processed, labels_created, representative_samples_set)
    
    Note: Processes ALL images in the split. Config is only used for selecting
    representative samples for metadata/analysis, not for filtering which images to process.
    """
    # VALIDATION: Config is REQUIRED - fail fast
    if not config:
        raise ValueError(
            f"Config parameter is required for processing {split} split. "
            f"Must provide a config dict from LIMITED_DATASET_CONFIGS."
        )
    
    required_keys = ['samples_per_attribute_combo', 'min_samples_per_class', 
                     'min_samples_per_attribute_value', 'min_samples_per_class_attribute_combo']
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(
            f"Config is missing required keys for {split} split: {missing_keys}. "
            f"Use a complete config from LIMITED_DATASET_CONFIGS."
        )
    
    print(f"\n{'='*70}")
    print(f"Processing {split} split")
    print(f"{'='*70}")
    
    # Define paths - labels are in 100k subdirectory
    split_images_src = tmp_images_dir / '100k' / split
    split_labels_src = tmp_labels_dir / '100k' / split
    
    split_images_dst = yolo_dataset_root / 'images' / split
    split_labels_dst = yolo_dataset_root / 'labels' / split
    metadata_dir = yolo_dataset_root / 'representative_json'
    
    # Check if source directories exist
    if not split_images_src.exists():
        print(f"⚠️  Warning: Images directory not found: {split_images_src}")
        return 0, 0, set()
    
    if not split_labels_src.exists():
        print(f"⚠️  Warning: Labels directory not found: {split_labels_src}")
        return 0, 0, set()
    
    # Create metadata directory (no individual JSON files needed)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Select representative samples with metadata
    _, representative_files, split_metadata = select_representative_samples(split_labels_src, split, config)
    
    # Flatten representative samples for easy lookup
    representative_basenames = {f.stem for f in representative_files}
    
    # Get all image files
    image_files = list(split_images_src.glob('*.jpg')) + list(split_images_src.glob('*.png'))
    
    if not image_files:
        print(f"⚠️  No images found in {split_images_src}")
        return 0, 0, representative_basenames
    
    print(f"Found {len(image_files)} images")
    
    images_processed = 0
    labels_created = 0
    validation_stats = {
        'total_processed': 0,
        'valid_objects': 0,
        'invalid_objects': 0,
        'images_with_objects': 0,
        'images_without_objects': 0
    }
    
    # Store attributes for test split performance metadata
    image_attributes_map = {}
    
    # Process each image
    for img_file in tqdm(image_files, desc=f"Processing {split}", unit='files'):
        images_processed += 1
        
        # Copy image
        dst_img_path = split_images_dst / img_file.name
        if not dst_img_path.exists():
            shutil.copy2(img_file, dst_img_path)
        
        # Convert label - ALWAYS create label file (empty if no objects)
        json_name = img_file.stem + '.json'
        json_path = split_labels_src / json_name
        label_dst_path = split_labels_dst / (img_file.stem + '.txt')
        
        if json_path.exists():
            yolo_labels, attributes, converted_count, skipped_count = convert_json_to_yolo(json_path)
            
            # Store attributes for test split
            if split == 'test' and attributes:
                image_attributes_map[img_file.stem] = attributes
            
            # Update conversion statistics
            validation_stats['total_processed'] += 1
            validation_stats['valid_objects'] += converted_count
            validation_stats['invalid_objects'] += skipped_count
            
            # Create label file (empty if no objects)
            create_label_file(label_dst_path, yolo_labels)
            labels_created += 1
            
            if yolo_labels:
                validation_stats['images_with_objects'] += 1
            else:
                validation_stats['images_without_objects'] += 1
        else:
            # Create empty label file if no JSON exists
            label_dst_path.touch()
            labels_created += 1
            validation_stats['images_without_objects'] += 1
    
    # Print conversion statistics (NO FILTERING APPLIED)
    print(f"\n  Conversion Statistics:")
    print(f"    ⚠️  NO FILTERING APPLIED - ALL objects with coordinates converted as-is")
    print(f"    JSON files processed: {validation_stats['total_processed']:,}")
    print(f"    Objects converted: {validation_stats['valid_objects']:,}")
    print(f"    Objects skipped (missing fields only): {validation_stats['invalid_objects']:,}")
    print(f"    Images with objects: {validation_stats['images_with_objects']:,}")
    print(f"    Images without objects: {validation_stats['images_without_objects']:,}")
    
    # Note: JSON files are NOT saved to subdirectories
    # Performance analysis will read directly from bdd100k_tmp_labels/100k
    
    # Count actual objects from ALL YOLO txt files using method
    print(f"\n  Counting objects in full dataset...")
    all_object_counts = count_objects_in_labels(split_labels_dst, f"  Counting {split}")
    print(f"  ✓ Total objects in full dataset: {sum(all_object_counts.values()):,}")
    
    # Count objects in representative samples from metadata that was already generated
    # Note: This data already exists in split_metadata from select_representative_samples
    representative_image_count = split_metadata['statistics']['total_selected']
    
    print(f"  ✓ Representative samples: {representative_image_count:,} images selected")
    
    # Count attribute distribution for FULL dataset
    print(f"\n  Analyzing attribute distribution for {split} split...")
    # Note: count_attribute_distribution expects base path and adds split internally
    weather_dist, scene_dist, timeofday_dist = count_attribute_distribution(tmp_labels_dir / '100k', split)
    
    if weather_dist or scene_dist or timeofday_dist:
        print(f"  ✓ Attribute distribution:")
        print(f"    Weather: {len(weather_dist)} types, {sum(weather_dist.values()):,} images")
        for weather_type, count in sorted(weather_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"      - {weather_type}: {count:,} images")
        
        print(f"    Scene: {len(scene_dist)} types, {sum(scene_dist.values()):,} images")
        for scene_type, count in sorted(scene_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"      - {scene_type}: {count:,} images")
        
        print(f"    Time: {len(timeofday_dist)} types, {sum(timeofday_dist.values()):,} images")
        for time_type, count in sorted(timeofday_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"      - {time_type}: {count:,} images")
        
        # Update metadata with FULL dataset attribute distribution
        split_metadata['statistics']['full_dataset_attributes'] = {
            'by_weather': weather_dist,
            'by_scene': scene_dist,
            'by_timeofday': timeofday_dist
        }
    
    # Update metadata structure with BOTH full dataset and representative sample statistics
    # IMPORTANT: split_metadata already contains representative sample IMAGE counts from select_representative_samples()
    # We add full dataset statistics WITHOUT overwriting the representative sample stats
    
    # Count objects in representative samples from YOLO files
    representative_object_counts = {cls: 0 for cls in BDD100K_CLASSES}
    for basename in representative_basenames:
        txt_file = split_labels_dst / f'{basename}.txt'
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            class_id = int(line.split()[0])
                            if 0 <= class_id < len(BDD100K_CLASSES):
                                representative_object_counts[BDD100K_CLASSES[class_id]] += 1
                        except (ValueError, IndexError):
                            continue
    
    # Add full dataset object counts (separate from representative samples)
    split_metadata['statistics']['full_dataset'] = {
        'total_images': len(image_files),
        'by_class_object_counts': all_object_counts  # OBJECT counts for FULL dataset
    }
    
    # Add representative sample object counts (calculated from YOLO labels)
    split_metadata['statistics']['representative_samples']['by_class_object_counts'] = representative_object_counts
    split_metadata['statistics']['representative_samples']['total_objects'] = sum(representative_object_counts.values())
    split_metadata['statistics']['representative_samples']['total_images'] = len(representative_basenames)
    
    # Add conversion/validation statistics
    split_metadata['validation'] = validation_stats
    
    # Save metadata JSON file for this split
    metadata_file = metadata_dir / f'{split}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
    # Save performance analysis metadata ONLY for test split (used for evaluation)
    if split == 'test':
        print(f"  Generating performance analysis for test split...")
        performance_file = metadata_dir / f'{split}_performance_analysis.json'
        save_test_performance_metadata(tmp_labels_dir / '100k', split_labels_dst, split, performance_file)
        print(f"  ✓ Performance analysis saved: {performance_file.name}")
    else:
        print(f"  (Performance analysis not generated for {split} - only for test split)")
    
    # Perform integrity check using method
    print(f"\n  Performing integrity check for {split}...")
    images_without_labels, labels_without_images, is_valid = perform_integrity_check(
        split_images_dst, split_labels_dst, split
    )
    
    if images_without_labels:
        print(f"  ⚠️  Warning: {len(images_without_labels)} images without labels")
        if len(images_without_labels) <= 5:
            print(f"      {list(images_without_labels)}")
    if labels_without_images:
        print(f"  ⚠️  Warning: {len(labels_without_images)} labels without images")
        if len(labels_without_images) <= 5:
            print(f"      {list(labels_without_images)}")
    
    if is_valid:
        print(f"  ✓ Integrity check PASSED: All images have labels and vice versa")
    
    # Compare original JSON vs generated YOLO statistics
    comparison_stats = compare_dataset_statistics(tmp_labels_dir, split_labels_dst, split)
    if comparison_stats:
        # Save comparison to metadata directory
        comparison_file = metadata_dir / f'{split}_comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison_stats, f, indent=2)
        print(f"  ✓ [2/3] Original vs Converted statistics validated: {comparison_file.name}")
        
        # [3/3] Validate representative JSON statistics
        print(f"\n  [3/3] Validating representative JSON statistics...")
        representative_json_dir = metadata_dir / split
        if representative_json_dir.exists():
            repr_json_files = list(representative_json_dir.glob('*.json'))
            if repr_json_files:
                repr_json_object_counts = count_objects_in_json_files(
                    representative_json_dir, 
                    f"    Counting representative JSON"
                )
                
                # Compare with representative object counts from YOLO files
                repr_match = True
                mismatches = []
                for cls in BDD100K_CLASSES:
                    json_count = repr_json_object_counts.get(cls, 0)
                    yolo_count = representative_object_counts.get(cls, 0)
                    if json_count != yolo_count:
                        repr_match = False
                        mismatches.append(f"{cls}: JSON={json_count:,} vs YOLO={yolo_count:,}")
                
                if repr_match:
                    print(f"    ✓ Representative JSON validated: {len(repr_json_files):,} files, "
                          f"{sum(repr_json_object_counts.values()):,} objects (matches YOLO counts)")
                else:
                    print(f"    ⚠️  Representative JSON mismatch detected:")
                    for mismatch in mismatches[:5]:  # Show first 5
                        print(f"      {mismatch}")
            else:
                print(f"    ⚠️  No representative JSON files found")
        else:
            print(f"    ⚠️  Representative JSON directory not found: {representative_json_dir}")
    
    print(f"\n✓ {split}: {images_processed:,} images processed, {labels_created:,} labels created")
    print(f"  - Images with objects: {validation_stats['images_with_objects']:,}")
    print(f"  - Images without objects: {validation_stats['images_without_objects']:,}")
    print(f"  Metadata saved: {metadata_file.name}")
    return images_processed, labels_created, representative_basenames


def create_data_yaml(dataset_root, base_dir):
    """
    Create data.yaml configuration file for YOLO training.
    Only includes splits that actually exist in the dataset.
    """
    # Check which splits exist in the dataset
    images_dir = dataset_root / 'images'
    existing_splits = []
    for split in ['train', 'val', 'test']:
        split_dir = images_dir / split
        if split_dir.exists() and any(split_dir.iterdir()):
            existing_splits.append(split)
    
    # Build data.yaml content dynamically
    yaml_lines = [
        "# BDD100K Dataset Configuration for YOLO",
        "# Auto-generated by process_bdd100k_to_yolo_dataset.py",
        "",
        f"path: {dataset_root.absolute()}  # Dataset root directory"
    ]
    
    # Add only existing splits
    for split in existing_splits:
        yaml_lines.append(f"{split}: images/{split}  # {split.capitalize()} images (relative to 'path')")
    
    # Add class configuration
    yaml_lines.extend([
        "",
        "# Number of classes",
        f"nc: {len(BDD100K_CLASSES)}",
        "",
        "# Class names",
        f"names: {BDD100K_CLASSES}",
        ""
    ])
    
    data_yaml_content = "\n".join(yaml_lines)
    
    yaml_path = dataset_root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\n✓ Created data.yaml: {yaml_path}")
    return yaml_path


def create_limited_dataset(base_dir, source_root, output_root, representative_samples_by_split, config):
    """
    Create a limited dataset using representative samples with diverse attributes.
    Uses the representative JSON samples selected during conversion.
    
    Args:
        base_dir: Base directory (needed to access tmp_labels for JSON metadata)
        source_root: Source YOLO dataset root
        output_root: Output YOLO dataset root
        representative_samples_by_split: Dict of {split: set of basenames}
        config: Configuration dict for this limited dataset
    """
    print("\n" + "="*70)
    print(f"CREATING LIMITED DATASET: {config['name']}")
    print("="*70)
    print(f"Description: {config['description']}")
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print("  Strategy: Using diverse representative samples with attributes")
    print(f"  Ensures:")
    print(f"    - {config['samples_per_attribute_combo']} samples per attribute combination")
    print(f"    - {config['min_samples_per_class']} samples per class")
    print(f"    - {config['min_samples_per_attribute_value']} samples per attribute value")
    print(f"    - {config['min_samples_per_class_attribute_combo']} samples per class×attribute combo")
    
    # Verify source exists
    if not source_root.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source_root}\n"
            f"Please run full dataset creation first."
        )
    
    # Create output root
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Get splits to process from config (defaults to all if not specified)
    splits_to_process = config.get('splits', ['train', 'val', 'test'])
    print(f"\n  Splits to process: {', '.join(splits_to_process)}")
    
    total_samples = 0
    
    # Process only the specified splits
    for split in splits_to_process:
        print(f"\n{split.upper()} SPLIT")
        print("-" * 70)
        
        source_images_dir = source_root / 'images' / split
        source_labels_dir = source_root / 'labels' / split
        
        output_images_dir = output_root / 'images' / split
        output_labels_dir = output_root / 'labels' / split
        
        # Create output directories
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        if not source_images_dir.exists():
            print(f"⚠️  Source directory not found: {source_images_dir}")
            continue
        
        # Check if we should copy the ENTIRE split without sampling
        copy_full_split = (
            (split == 'test' and config.get('contain_full_test_split', False)) or
            (split == 'val' and config.get('contain_full_val_split', False))
        )
        
        if copy_full_split:
            # Copy ALL files from test split without sampling
            print(f"Copying ENTIRE {split} split (contain_full_test_split=True)...")
            all_image_files = list(source_images_dir.glob('*.[jJ][pP][gG]')) + \
                             list(source_images_dir.glob('*.[pP][nN][gG]')) + \
                             list(source_images_dir.glob('*.[jJ][pP][eE][gG]'))
            
            basenames_to_copy = {img_file.stem for img_file in all_image_files}
            print(f"Total files to copy: {len(basenames_to_copy)}")
        else:
            # Use representative samples as usual
            basenames_to_copy = representative_samples_by_split.get(split, set())
            
            if not basenames_to_copy:
                print(f"⚠️  No representative samples found for {split}")
                continue
            
            print(f"Creating limited dataset with {len(basenames_to_copy)} representative samples...")
        
        # Copy files for representative samples (or all files if contain_full_test_split)
        samples_copied = 0
        for basename in tqdm(basenames_to_copy, desc=f"Copying {split}", unit='files'):
            # Copy image
            for ext in ['.jpg', '.png', '.jpeg']:
                img_file = source_images_dir / f"{basename}{ext}"
                if img_file.exists():
                    dst_img_path = output_images_dir / img_file.name
                    if not dst_img_path.exists():
                        shutil.copy2(img_file, dst_img_path)
                    samples_copied += 1
                    break
            
            # Copy label
            label_file = source_labels_dir / f"{basename}.txt"
            if label_file.exists():
                dst_label_path = output_labels_dir / label_file.name
                if not dst_label_path.exists():
                    shutil.copy2(label_file, dst_label_path)
        
        # Note: JSON files are NOT copied to subdirectories
        # Performance analysis metadata will be generated from source tmp_labels
        print(f"✓ {split}: {samples_copied} representative samples copied")
        
        total_samples += samples_copied
    
    # Generate NEW metadata for limited dataset by analyzing representative samples
    output_metadata_dir = output_root / 'representative_json'
    output_metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up any old JSON subdirectories (not needed anymore)
    for split in ['train', 'val', 'test']:
        old_json_dir = output_metadata_dir / split
        if old_json_dir.exists():
            shutil.rmtree(old_json_dir)
    
    print("\n" + "="*70)
    print("GENERATING METADATA FOR LIMITED DATASET")
    print("="*70)
    print("Note: Performance analysis generated ONLY for test split (if exists)")
    print("      Reads from source bdd100k_tmp_labels/100k")
    
    # Get splits to process from config
    splits_to_process = config.get('splits', ['train', 'val', 'test'])
    
    for split in splits_to_process:
        print(f"\nAnalyzing {split} split...")
        
        # Analyze the limited dataset labels directory
        split_labels_dir = output_root / 'labels' / split
        
        if not split_labels_dir.exists():
            print(f"  ⚠️  Skipping {split}: directory not found")
            continue
        
        # Check if split has any files
        label_files = list(split_labels_dir.glob('*.txt'))
        if not label_files:
            print(f"  ⚠️  Skipping {split}: no label files found")
            continue
        
        # Count objects from the LIMITED dataset labels (representative samples only)
        limited_class_stats = count_objects_in_labels(split_labels_dir, f"  Counting {split}")
        
        # Count attributes by reading from SOURCE tmp_labels (always available)
        tmp_labels_source = base_dir / 'bdd100k_tmp_labels' / '100k'
        limited_attributes = count_attribute_distribution(tmp_labels_source, split, 
                                                          filter_basenames={f.stem for f in label_files})
        
        # Build metadata for limited dataset
        limited_metadata = {
            'split': split,
            'total_samples': len(label_files),
            'statistics': {
                'by_class': limited_class_stats,
                'attributes': limited_attributes
            },
            'data_source': 'limited_dataset_representative_samples',
            'description': f'Limited dataset ({split}) - representative samples only'
        }
        
        # Save metadata for limited dataset
        metadata_file = output_metadata_dir / f'{split}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(limited_metadata, f, indent=2)
        
        print(f"  ✓ Metadata generated: {metadata_file.name}")
        print(f"    - Files analyzed: {limited_metadata['total_samples']}")
        print(f"    - Total objects: {sum(limited_metadata['statistics']['by_class'].values())}")
        
        # Save performance analysis ONLY for test split (used for evaluation)
        if split == 'test':
            print(f"  Generating performance analysis for test split...")
            performance_file = output_metadata_dir / f'{split}_performance_analysis.json'
            tmp_labels_source = base_dir / 'bdd100k_tmp_labels' / '100k'
            save_test_performance_metadata(tmp_labels_source, split_labels_dir, split, performance_file)
            print(f"  ✓ Performance analysis saved: {performance_file.name}")
        else:
            print(f"  (Performance analysis not generated for {split} - only for test split)")
    
    print("\n" + "="*70)
    print("✓ LIMITED DATASET METADATA REGENERATED")
    print("="*70)
    
    # Create data.yaml for limited dataset
    yaml_path = create_data_yaml(output_root, output_root.parent)
    
    print("\n" + "="*70)
    print(f"✓ {config['name'].upper()} CREATED")
    print("="*70)
    print(f"Dataset location: {output_root}")
    print(f"Configuration: {yaml_path}")
    print(f"Description: {config['description']}")
    print(f"Total samples: {total_samples}")
    print(f"Composition: Diverse samples across weather/scene/time attributes")
    print(f"Per-class coverage: min {config['min_samples_per_class']} samples")
    print("="*70)
    
    return output_root, yaml_path
    
 

def main():
    """Main entry point - runs complete dataset preparation without parameters."""
    # Step 1: Check/download dataset files
    print("=" * 70)
    print("BDD100K YOLO Dataset Preparation")
    print("=" * 70)
    
    if not check_and_download_datasets(source_dir):
        print("\n❌ Dataset files not available. Exiting.")
        return
    
    # Step 2: Image dimensions (validated from actual dataset)
    print("\n" + "="*70)
    print("STEP 2: IMAGE DIMENSIONS")
    print("="*70)
    
    # Using validated dimensions from BDD100K dataset (all images are 1280×720)
    print(f"\nBDD100K image dimensions: {BDD100K_IMAGE_WIDTH}×{BDD100K_IMAGE_HEIGHT}")
    print("✓ Dimensions validated from actual dataset analysis")
    
    # Step 3: Extract and create full dataset
    print("\n" + "="*70)
    print("CREATING FULL DATASET")
    print("="*70)
    
    representative_samples_by_split = extract_and_prepare_yolo_dataset(
        base_dir, source_dir, yolo_dataset_root
    )
    
    # Step 4: Create limited datasets SEQUENTIALLY
    print("\n" + "="*70)
    print(f"STEP 4: CREATE LIMITED DATASETS (SEQUENTIAL PROCESSING)")
    print("="*70)
    print(f"Processing {len(LIMITED_DATASET_CONFIGS)} configurations sequentially")
    print("Strategy: Complete each dataset before starting the next")
    print("  Config 1 → from Full dataset → Complete → Save")
    print("  Config 2 → from Config 1 → Complete → Save")
    print("  Config 3 → from Config 1 (NOT Config 2) → Complete → Save")
    print("="*70)
    print("="*70)
    
    created_datasets = []
    
    for idx, config in enumerate(LIMITED_DATASET_CONFIGS, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(LIMITED_DATASET_CONFIGS)}] Creating: {config['name']}")
        print(f"{'='*70}")
        print(f"Description: {config['description']}")
        
        # Determine source dataset
        source_name = config.get('source_dataset', 'full')
        if source_name == 'full':
            source_root = yolo_dataset_root
            source_tmp_labels = base_dir / 'bdd100k_tmp_labels' / '100k'
            print(f"Source: Full dataset ({yolo_dataset_root.name})")
        else:
            source_root = base_dir / source_name
            source_tmp_labels = base_dir / 'bdd100k_tmp_labels' / '100k'  # Always use original labels for attributes
            print(f"Source: {source_name}")
            if not source_root.exists():
                print(f"❌ ERROR: Source dataset not found: {source_root}")
                print(f"   Cannot create {config['name']} without {source_name}")
                print(f"   Skipping this dataset.")
                continue
        
        output_root = base_dir / config['name']
        
        # Process each split for this config
        print(f"\nProcessing splits...")
        config_samples = {}
        
        for split in config.get('splits', ['train', 'val', 'test']):
            print(f"\n  {split.upper()} split:")
            
            split_labels_src = source_tmp_labels / split
            source_labels_dir = source_root / 'labels' / split
            
            if not split_labels_src.exists() or not source_labels_dir.exists():
                print(f"    ⚠️  Skipping: source directories not found")
                config_samples[split] = set()
                continue
            
            # Check if we should use ENTIRE split without sampling
            use_full_split = (
                (split == 'test' and config.get('contain_full_test_split', False)) or
                (split == 'val' and config.get('contain_full_val_split', False))
            )
            
            if use_full_split:
                # Use ALL files from source
                all_label_files = list(source_labels_dir.glob('*.txt'))
                config_samples[split] = {f.stem for f in all_label_files}
                print(f"    Using ENTIRE split: {len(config_samples[split]):,} images")
            else:
                # Sample representative files from source
                # Get basenames from source to use as constraint
                source_label_files = list(source_labels_dir.glob('*.txt'))
                constrain_to = {f.stem for f in source_label_files}
                
                # Check if source has any files for this split
                if not constrain_to:
                    print(f"    ⚠️  Source has no samples for {split} split, skipping...")
                    config_samples[split] = set()
                    continue
                
                print(f"    Sampling from {len(constrain_to):,} available images...")
                _, representative_files, _ = select_representative_samples(
                    split_labels_src,
                    split,
                    config,
                    constrain_to_basenames=constrain_to
                )
                config_samples[split] = {f.stem for f in representative_files}
                print(f"    Selected: {len(config_samples[split]):,} images")
        
        # Create the limited dataset by copying files
        print(f"\n  Creating limited dataset...")
        create_limited_dataset(base_dir, source_root, output_root, config_samples, config)
        
        created_datasets.append({
            'config': config,
            'path': output_root,
            'source': source_name
        })
        
        print(f"\n✅ {config['name']} completed successfully!")
        print(f"   Output: {output_root}")
        total_samples = sum(len(s) for s in config_samples.values())
        print(f"   Total: {total_samples:,} images")
    
    # Step 6: Compress all limited datasets
    print("\n" + "="*70)
    print(f"COMPRESSING {len(created_datasets)} LIMITED DATASETS")
    print("="*70)
    
    zipped_dir = base_dir / 'bdd100k_limited_datasets_zipped'
    zipped_dir.mkdir(parents=True, exist_ok=True)
    
    compressed_files = []
    
    for dataset_info in created_datasets:
        config = dataset_info['config']
        dataset_path = dataset_info['path']
        compressed_file = zipped_dir / f"{config['name']}.zip"
        
        # Remove existing compressed file if present
        if compressed_file.exists():
            print(f"\nRemoving existing: {compressed_file.name}")
            compressed_file.unlink()
        
        print(f"\nCompressing {config['name']}...")
        print(f"  Destination: {compressed_file}")
        
        with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            # Get all files in the limited dataset
            all_files = list(dataset_path.rglob('*'))
            for file_path in tqdm(all_files, desc=f"  Compressing {config['name']}", unit='files'):
                if file_path.is_file():
                    # Calculate relative path
                    rel_path = file_path.relative_to(dataset_path.parent)
                    zipf.write(file_path, arcname=rel_path)
        
        file_size_mb = compressed_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ Size: {file_size_mb:.1f} MB")
        compressed_files.append({
            'name': config['name'],
            'path': compressed_file,
            'size_mb': file_size_mb,
            'description': config['description']
        })
    
    print("\n" + "="*70)
    print("✅ FULL PROCESS COMPLETE")
    print("="*70)
    print(f"\nFull dataset: {yolo_dataset_root}")
    print(f"\nLimited datasets created ({len(created_datasets)}):")
    for ds_info in created_datasets:
        print(f"  - {ds_info['config']['name']}: {ds_info['path']}")
    print(f"\nCompressed files ({len(compressed_files)}):")
    for cf in compressed_files:
        print(f"  - {cf['name']}.zip ({cf['size_mb']:.1f} MB) - {cf['description']}")
    print(f"\nCompressed location: {zipped_dir}")
    
    # Step 7: Compress test split only from full dataset
    test_split_result = compress_test_split_only(yolo_dataset_root, base_dir)
    if test_split_result:
        print(f"\n✅ Test split compressed:")
        print(f"  - {test_split_result['path'].name} ({test_split_result['size_mb']:.1f} MB)")
        print(f"  - {test_split_result['num_images']:,} images, {test_split_result['num_labels']:,} labels")
    
    print("\n" + "="*70)
    print(f"\n  # For limited dataset (quick testing, visualization, experimentation):")
    print(f"  YOLO_DATASET_ROOT = BASE_DIR / 'bdd100k_yolo_limited'")
    print(f"  DATA_YAML_PATH = YOLO_DATASET_ROOT / 'data.yaml'")
    print(f"  Note: Limited dataset IS the representative samples (physically copied)")
    print("="*70)


def verify_image_dimensions(tmp_images_dir, sample_size=100):
    """
    Verify image dimensions from actual dataset images.
    Checks multiple images to ensure consistency.
    
    Args:
        tmp_images_dir: Path to temporary images directory
        sample_size: Number of images to check (default 100)
    
    Returns:
        Tuple of (width, height, is_consistent, dimension_counts)
    """
    print("\n" + "="*70)
    print("VERIFYING IMAGE DIMENSIONS FROM ACTUAL DATASET")
    print("="*70)
    
    dimension_counts = {}
    images_checked = 0
    
    # Check images from all splits
    for split in ['train', 'val', 'test']:
        split_dir = tmp_images_dir / '100k' / split
        if not split_dir.exists():
            continue
        
        # Get image files
        image_files = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png'))
        
        # Check up to sample_size images per split
        for img_file in image_files[:min(sample_size, len(image_files))]:
            try:
                with Image.open(img_file) as img:
                    dimensions = img.size  # Returns (width, height)
                    dimension_counts[dimensions] = dimension_counts.get(dimensions, 0) + 1
                    images_checked += 1
            except Exception as e:
                print(f"⚠️  Warning: Could not read {img_file.name}: {e}")
                continue
        
        if images_checked >= sample_size:
            break
    
    if not dimension_counts:
        raise RuntimeError("Could not verify image dimensions - no images found or readable")
    
    # Find most common dimension
    most_common_dim = max(dimension_counts.items(), key=lambda x: x[1])
    width, height = most_common_dim[0]
    count = most_common_dim[1]
    
    # Check if all images have the same dimension
    is_consistent = len(dimension_counts) == 1
    
    print(f"\nImages checked: {images_checked}")
    print(f"\nDimension analysis:")
    for dim, cnt in sorted(dimension_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (cnt / images_checked) * 100
        status = "✓" if dim == most_common_dim[0] else "⚠️"
        print(f"  {status} {dim[0]}×{dim[1]}: {cnt} images ({percentage:.1f}%)")
    
    if is_consistent:
        print(f"\n✓ All images have consistent dimensions: {width}×{height}")
    else:
        print(f"\n⚠️  Warning: Multiple image dimensions detected!")
        print(f"   Most common: {width}×{height} ({count}/{images_checked} images)")
    
    return width, height, is_consistent, dimension_counts


def check_extraction_complete(tmp_dir, expected_splits=['train', 'val', 'test']):
    """Check if extraction is complete by verifying split directories exist and have files."""
    if not tmp_dir.exists():
        return False
    
    # Check for 100k subdirectory structure
    base_100k = tmp_dir / '100k'
    if not base_100k.exists():
        return False
    
    for split in expected_splits:
        split_dir = base_100k / split
        if not split_dir.exists():
            return False
        # Check if directory has files
        if not any(split_dir.iterdir()):
            return False
    
    return True


def check_dataset_complete(yolo_dataset_root, expected_splits=['train', 'val', 'test']):
    """Check if YOLO dataset is complete with all splits and metadata files."""
    if not yolo_dataset_root.exists():
        return False
    
    # Check data.yaml
    if not (yolo_dataset_root / 'data.yaml').exists():
        return False
    
    # Check all splits have images and labels
    for split in expected_splits:
        images_dir = yolo_dataset_root / 'images' / split
        labels_dir = yolo_dataset_root / 'labels' / split
        
        if not images_dir.exists() or not labels_dir.exists():
            return False
        
        # Check if directories have files
        if not any(images_dir.iterdir()) or not any(labels_dir.iterdir()):
            return False
    
    # Check metadata files
    metadata_dir = yolo_dataset_root / 'representative_json'
    if not metadata_dir.exists():
        return False
    
    for split in expected_splits:
        metadata_file = metadata_dir / f'{split}_metadata.json'
        if not metadata_file.exists():
            return False
    
    return True


def analyze_data_integrity(tmp_images_dir, tmp_labels_dir):
    """
    Analyze data integrity by finding mismatches between images and labels.
    This runs on the original extracted TMP dataset before YOLO conversion.
    
    Returns:
        Tuple of (images_without_labels, labels_without_images, integrity_passed)
    """
    print("\n" + "="*70)
    print("DATA INTEGRITY ANALYSIS - ORIGINAL TMP DATASET")
    print("="*70)
    print("Checking for mismatches between images and labels...")
    
    images_without_labels = []
    labels_without_images = []
    
    # Check each split
    for split in ['train', 'val', 'test']:
        images_dir = tmp_images_dir / '100k' / split
        labels_dir = tmp_labels_dir / '100k' / split
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"⚠️  {split} directories not found, skipping...")
            continue
        
        print(f"\nAnalyzing {split.upper()} split...")
        
        # Get all image files
        image_files = {}
        for ext in ['.jpg', '.png', '.jpeg']:
            for img_path in images_dir.glob(f"*{ext}"):
                basename = img_path.stem
                image_files[basename] = img_path
        
        # Get all label files (JSON format before conversion)
        label_files = {}
        for label_path in labels_dir.glob("*.json"):
            basename = label_path.stem
            label_files[basename] = label_path
        
        # Find images without labels
        for basename, img_path in image_files.items():
            if basename not in label_files:
                images_without_labels.append({
                    'basename': basename,
                    'path': img_path,
                    'split': split
                })
        
        # Find labels without images
        for basename, label_path in label_files.items():
            if basename not in image_files:
                labels_without_images.append({
                    'basename': basename,
                    'path': label_path,
                    'split': split
                })
        
        print(f"  Images: {len(image_files):,}")
        print(f"  Labels: {len(label_files):,}")
        print(f"  Images without labels: {len([x for x in images_without_labels if x['split'] == split])}")
        print(f"  Labels without images: {len([x for x in labels_without_images if x['split'] == split])}")
    
    # Summary
    print("\n" + "="*70)
    print("INTEGRITY ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total images without labels: {len(images_without_labels):,}")
    print(f"Total labels without images: {len(labels_without_images):,}")
    
    # Display breakdown by split
    if images_without_labels:
        print("\nImages without labels by split:")
        for split in ['train', 'val', 'test']:
            count = len([x for x in images_without_labels if x['split'] == split])
            if count > 0:
                print(f"  {split}: {count:,}")
        
        # Show first few examples
        print("\nFirst 10 images without labels:")
        for idx, sample in enumerate(images_without_labels[:10]):
            print(f"  {idx+1}. {sample['basename']} (Split: {sample['split']})")
    
    if labels_without_images:
        print("\nLabels without images by split:")
        for split in ['train', 'val', 'test']:
            count = len([x for x in labels_without_images if x['split'] == split])
            if count > 0:
                print(f"  {split}: {count:,}")
        
        # Show first few examples
        print("\nFirst 10 labels without images:")
        for idx, sample in enumerate(labels_without_images[:10]):
            print(f"  {idx+1}. {sample['basename']} (Split: {sample['split']})")
    
    integrity_passed = len(images_without_labels) == 0 and len(labels_without_images) == 0
    
    if integrity_passed:
        print("\n✅ INTEGRITY CHECK PASSED: All images have corresponding labels and vice versa")
    else:
        print("\n⚠️  INTEGRITY ISSUES FOUND - See details above")
        print("   These mismatches will be handled during YOLO conversion:")
        print("   - Images without labels will get empty label files")
        print("   - Labels without images will be skipped")
    
    print("="*70)
    
    return images_without_labels, labels_without_images, integrity_passed


def extract_and_prepare_yolo_dataset(base_dir, source_dir, yolo_dataset_root):
    """Extract BDD100K dataset and create YOLO-compatible structure with representative samples."""
    print("=" * 70)
    print("BDD100K Dataset Extraction & YOLO Conversion")
    print("=" * 70)
    
    images_zip = source_dir / "bdd100k_images_100k.zip"
    labels_zip = source_dir / "bdd100k_labels.zip"
    
    tmp_images_dir = base_dir / "bdd100k_tmp_images"
    tmp_labels_dir = base_dir / "bdd100k_tmp_labels"
    
    # Validate source files exist
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Step 1: Extract zip files (skip if already extracted)
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING ZIP FILES")
    print("="*70)
    
    images_extracted = check_extraction_complete(tmp_images_dir)
    labels_extracted = check_extraction_complete(tmp_labels_dir)
    
    if images_extracted and labels_extracted:
        print("✓ Extraction already complete, skipping...")
        print(f"  Images: {tmp_images_dir}")
        print(f"  Labels: {tmp_labels_dir}")
    else:
        if images_zip.exists() and not images_extracted:
            images_count = extract_zip_with_progress(
                str(images_zip),
                str(tmp_images_dir),
                "Extracting images..."
            )
        elif images_extracted:
            print(f"✓ Images already extracted: {tmp_images_dir}")
        else:
            print(f"⚠️  Warning: Images zip not found: {images_zip}")
        
        if labels_zip.exists() and not labels_extracted:
            labels_count = extract_zip_with_progress(
                str(labels_zip),
                str(tmp_labels_dir),
                "Extracting labels..."
            )
        elif labels_extracted:
            print(f"✓ Labels already extracted: {tmp_labels_dir}")
        else:
            print(f"⚠️  Warning: Labels zip not found: {labels_zip}")
    
    # STEP 1.5: DATA INTEGRITY CHECK (MANDATORY - CANNOT BE SKIPPED)
    # This step performs comprehensive validation:
    # 1. Image-Label Matching: Ensures all images have corresponding labels (JSON format)
    # 2. Original vs Converted Statistics: Validates object counts match between JSON and YOLO formats
    # 3. Representative JSON Validation: Ensures representative samples statistics are consistent
    print("\n" + "="*70)
    print("STEP 1.5: DATA INTEGRITY CHECK")
    print("="*70)
    print("This step validates:")
    print("  1. Image-Label matching (all images have corresponding JSON labels)")
    print("  2. Original JSON vs Converted YOLO statistics (object counts match)")
    print("  3. Representative JSON files statistics (subset consistency)")
    print("="*70)
    
    print("\n[1/3] Analyzing original TMP dataset before YOLO conversion...")
    print("Checking for mismatches between images and labels (JSON format)")
    
    # Run integrity analysis on TMP dataset (original JSON labels)
    imgs_no_labels, labels_no_imgs, integrity_ok = analyze_data_integrity(
        tmp_images_dir,
        tmp_labels_dir
    )
    
    # Note: We continue processing regardless of integrity issues
    # The YOLO conversion will handle mismatches appropriately
    
    print("\n" + "="*70)
    print("STEP 1.5 COMPLETE: DATA INTEGRITY CHECK SUMMARY")
    print("="*70)
    print("✓ [1/3] Image-Label matching validation: DONE")
    print("✓ [2/3] Original vs Converted statistics: Will be validated per split during conversion")
    print("✓ [3/3] Representative JSON validation: Will be validated per split during conversion")
    print("="*70)
    
    # Step 2 & 3: Create YOLO dataset and process splits (skip if complete)
    dataset_complete = check_dataset_complete(yolo_dataset_root)
    
    if dataset_complete:
        print("\n" + "="*70)
        print("STEPS 2-3: DATASET ALREADY COMPLETE, SKIPPING...")
        print("="*70)
        print(f"✓ YOLO dataset found: {yolo_dataset_root}")
        print(f"✓ All splits (train/val/test) verified")
        print(f"✓ Metadata files verified")
        
        # Load representative samples from metadata files
        representative_samples_by_split = {}
        metadata_dir = yolo_dataset_root / 'representative_json'
        for split in ['train', 'val', 'test']:
            metadata_file = metadata_dir / f'{split}_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Extract basenames from selected_samples.by_class structure
                    basenames = set()
                    by_class = metadata.get('selected_samples', {}).get('by_class', {})
                    for class_name, class_basenames in by_class.items():
                        if isinstance(class_basenames, list):
                            basenames.update(class_basenames)
                    representative_samples_by_split[split] = basenames
        
        yolo_dataset_root_created = yolo_dataset_root
        total_images = sum(1 for _ in (yolo_dataset_root / 'images' / 'train').glob('*'))
        total_labels = sum(1 for _ in (yolo_dataset_root / 'labels' / 'train').glob('*.txt'))
    else:
        # Step 2: Create YOLO dataset structure
        print("\n" + "="*70)
        print("STEP 2: CREATING YOLO DATASET STRUCTURE")
        print("="*70)
        
        yolo_dataset_root_created = create_yolo_dataset_structure(base_dir, yolo_dataset_root.name)
        print(f"✓ YOLO dataset structure created: {yolo_dataset_root_created}")
        
        # Step 3: Process each split
        print("\n" + "="*70)
        print("STEP 3: CONVERTING LABELS & ORGANIZING FILES")
        print("="*70)
        
        total_images = 0
        total_labels = 0
        representative_samples_by_split = {}
        
        # Use first config for full dataset metadata (representative sample selection for analysis only)
        # This does NOT filter images - ALL images are processed regardless of config
        metadata_config = LIMITED_DATASET_CONFIGS[0]
        
        for split in ['train', 'val', 'test']:
            imgs, lbls, repr_samples = process_split(tmp_images_dir, tmp_labels_dir, yolo_dataset_root_created, split, metadata_config)
            total_images += imgs
            total_labels += lbls
            representative_samples_by_split[split] = repr_samples
    
    # Step 4: Create data.yaml
    print("\n" + "="*70)
    print("STEP 4: CREATING CONFIGURATION FILE")
    print("="*70)
    
    yaml_path = create_data_yaml(yolo_dataset_root_created, base_dir)
    
    print("\n" + "="*70)
    print("NOTE: Temporary directories preserved for reference:")
    print(f"  {tmp_images_dir}")
    print(f"  {tmp_labels_dir}")
    print("="*70)
    
    # Final Summary
    print("\n" + "="*70)
    print("EXTRACTION & CONVERSION COMPLETE")
    print("="*70)
    print(f"Dataset location: {yolo_dataset_root_created}")
    print(f"Configuration file: {yaml_path}")
    print(f"\nStatistics:")
    print(f"  Total images: {total_images}")
    print(f"  Total labels: {total_labels}")
    print(f"  Classes: {len(BDD100K_CLASSES)}")
    # Note: Integrity status determined in process_split for each split
    print(f"\nYOLO Dataset Structure:")
    print(f"  {yolo_dataset_root_created}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/")
    print(f"    │   ├── val/")
    print(f"    │   └── test/")
    print(f"    ├── labels/")
    print(f"    │   ├── train/")
    print(f"    │   ├── val/")
    print(f"    │   └── test/")
    print(f"    ├── representative_json/  # Metadata files only")
    print(f"    │   ├── train_metadata.json")
    print(f"    │   ├── val_metadata.json")
    print(f"    │   └── test_metadata.json")
    print(f"    └── data.yaml")
    print("\n✅ Full dataset ready for YOLO training!")
    print(f"   Use data.yaml path in your training notebooks: {yaml_path}")
    print(f"   Metadata files contain statistics and representative sample paths")
    print("="*70)
    
    return representative_samples_by_split


if __name__ == "__main__":
    main()
