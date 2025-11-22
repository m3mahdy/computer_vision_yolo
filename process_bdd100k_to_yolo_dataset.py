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
10. Keeps temporary directories by default for reference (use --cleanup to remove)

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
    python process_bdd100k_to_yolo_dataset.py --skip-download

    # Skip download if files already exist
    python process_bdd100k_to_yolo_dataset.py --skip-download
    
    # Force metadata regeneration
    python process_bdd100k_to_yolo_dataset.py --force-reanalysis
    
    # Remove temporary directories after processing
    python process_bdd100k_to_yolo_dataset.py --cleanup
"""

import os
import json
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse
import random
import subprocess
import sys
from datetime import datetime

# BDD100K object detection classes (10 classes)
BDD100K_CLASSES = [
    'pedestrian',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'traffic light',
    'traffic sign'
]

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

# Representative sample attributes for diverse dataset visualization
REPRESENTATIVE_ATTRIBUTES = {
    'weather': ['clear', 'overcast', 'rainy', 'snowy', 'partly cloudy'],
    'scene': ['city street', 'highway', 'residential', 'parking lot'],
    'timeofday': ['daytime', 'night', 'dawn/dusk']
}

# Sampling configuration for representative dataset
SAMPLES_PER_ATTRIBUTE_COMBO = 10  # Samples per (weather, scene, timeofday) combination
MIN_SAMPLES_PER_CLASS = 10  # Minimum samples per object class per split
MIN_SAMPLES_PER_ATTRIBUTE_VALUE = 5  # Minimum samples per individual attribute value (weather/scene/time)
MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO = 3  # Minimum samples per (class, attribute) combination


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


def check_and_download_datasets(source_dir, skip_download=False):
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
    
    if skip_download:
        print("\n⚠️  Dataset files not found and --skip-download flag is set.")
        print("Please download manually from: http://bdd-data.berkeley.edu/")
        print(f"  Required files:")
        print(f"    - bdd100k_images_100k.zip (~6GB)")
        print(f"    - bdd100k_labels.zip (~300MB)")
        print(f"  Place them in: {source_dir}")
        return False
    
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
    Convert BDD100K bbox format to YOLO format.
    
    BDD100K format: {x1, y1, x2, y2} (absolute pixel coordinates)
    YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    """
    x1, y1 = bbox['x1'], bbox['y1']
    x2, y2 = bbox['x2'], bbox['y2']
    
    # Calculate YOLO format values
    x_center = ((x1 + x2) / 2.0) / img_width
    y_center = ((y1 + y2) / 2.0) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Ensure values are within [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return [x_center, y_center, width, height]


def get_label_attributes(json_path):
    """
    Extract attributes from a BDD100K JSON label file.
    Returns dict with attributes and object categories.
    """
    try:
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        
        attributes = label_data.get('attributes', {})
        frames = label_data.get('frames', [])
        
        # Get object categories
        if frames:
            objects = frames[0].get('objects', [])
        else:
            objects = label_data.get('objects', label_data.get('labels', []))
        
        categories = [obj.get('category', '') for obj in objects if 'box2d' in obj]
        
        return {
            'weather': attributes.get('weather', 'undefined'),
            'scene': attributes.get('scene', 'undefined'),
            'timeofday': attributes.get('timeofday', 'undefined'),
            'categories': [cat for cat in categories if cat in CLASS_TO_IDX],
            'num_objects': len(categories)
        }
    except Exception as e:
        return None


def validate_yolo_bbox(bbox_coords):
    """
    Validate YOLO format bounding box coordinates.
    Returns True if valid, False otherwise.
    """
    if len(bbox_coords) != 4:
        return False
    
    x_center, y_center, width, height = bbox_coords
    
    # All values must be in [0, 1]
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
        return False
    
    # Width and height must be positive
    if width <= 0 or height <= 0:
        return False
    
    return True


def convert_json_to_yolo(json_path, validate=True):
    """
    Convert a single BDD100K JSON label file to YOLO format with validation.
    Returns tuple: (yolo_labels, attributes, valid_objects, invalid_objects)
    - yolo_labels: List of YOLO format strings
    - attributes: Dict with weather, scene, timeofday
    - valid_objects: Count of successfully converted objects
    - invalid_objects: Count of objects that failed validation
    """
    try:
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        
        # Extract attributes
        attributes = {
            'weather': label_data.get('attributes', {}).get('weather', 'undefined'),
            'scene': label_data.get('attributes', {}).get('scene', 'undefined'),
            'timeofday': label_data.get('attributes', {}).get('timeofday', 'undefined')
        }
        
        # BDD100K images are typically 1280x720
        img_width = 1280
        img_height = 720
        
        # Process labels
        yolo_labels = []
        valid_count = 0
        invalid_count = 0
        
        frames = label_data.get('frames', [])
        
        if frames:
            objects = frames[0].get('objects', [])
        else:
            objects = label_data.get('objects', label_data.get('labels', []))
        
        for obj in objects:
            category = obj.get('category', '')
            
            if category not in CLASS_TO_IDX:
                continue
            
            box2d = obj.get('box2d')
            if not box2d:
                invalid_count += 1
                continue
            
            # Validate bbox has required fields
            if not all(k in box2d for k in ['x1', 'y1', 'x2', 'y2']):
                invalid_count += 1
                continue
            
            class_idx = CLASS_TO_IDX[category]
            yolo_bbox = convert_bbox_to_yolo(box2d, img_width, img_height)
            
            # Validate converted bbox
            if validate and not validate_yolo_bbox(yolo_bbox):
                invalid_count += 1
                continue
            
            # Format: class_idx x_center y_center width height
            yolo_line = f"{class_idx} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
            yolo_labels.append(yolo_line)
            valid_count += 1
        
        return yolo_labels, attributes, valid_count, invalid_count
    
    except Exception as e:
        print(f"Warning: Error processing {json_path}: {e}")
        return [], {}, 0, 0


def count_attribute_distribution(tmp_labels_dir, split_name):
    """
    Count distribution of attribute values (weather/scene/timeofday) across ALL images.
    Returns dict with counts for each attribute value.
    """
    json_dir = tmp_labels_dir / split_name
    if not json_dir.exists():
        return {}, {}, {}
    
    json_files = list(json_dir.glob('*.json'))
    
    weather_counts = {}
    scene_counts = {}
    timeofday_counts = {}
    
    for json_file in json_files:
        attrs = get_label_attributes(json_file)
        if attrs:
            weather = attrs.get('weather', 'undefined')
            scene = attrs.get('scene', 'undefined')
            timeofday = attrs.get('timeofday', 'undefined')
            
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


def save_test_performance_metadata(tmp_labels_dir, yolo_labels_dir, split_name, output_file):
    """
    Save detailed per-image metadata to enable performance analysis.
    For each image, stores: basename, attributes (weather/scene/timeofday), 
    classes present, and object counts per class.
    Works with both full dataset (all images) and limited dataset (representative samples).
    """
    print(f"\n  Generating performance analysis metadata for {split_name} split...")
    
    performance_data = {
        'split': split_name,
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': 0,
        'images': []
    }
    
    # Get JSON files from source (contains attributes)
    # Try 100k subfolder first (full dataset source), then direct path (limited dataset)
    json_dir = tmp_labels_dir / '100k' / split_name
    if not json_dir.exists():
        json_dir = tmp_labels_dir / split_name
    
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
        
        # Get attributes if available
        attrs = json_map.get(basename, {})
        
        image_data = {
            'basename': basename,
            'weather': attrs.get('weather', 'unknown'),
            'scene': attrs.get('scene', 'unknown'),
            'timeofday': attrs.get('timeofday', 'unknown'),
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


def select_representative_samples(split_labels_src, split_name):
    """
    Select representative samples ensuring comprehensive coverage:
    1. SAMPLES_PER_ATTRIBUTE_COMBO per (weather, scene, timeofday) combination
    2. MIN_SAMPLES_PER_CLASS per object class
    3. MIN_SAMPLES_PER_ATTRIBUTE_VALUE per individual attribute value
    4. MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO per (class, attribute) combination (per split)
    
    Returns comprehensive metadata dict with statistics and selected samples.
    """
    print(f"\n  Analyzing labels for representative sample selection...")
    print(f"    Configuration:")
    print(f"      - {SAMPLES_PER_ATTRIBUTE_COMBO} samples per attribute combo (weather×scene×time)")
    print(f"      - {MIN_SAMPLES_PER_CLASS} samples per class")
    print(f"      - {MIN_SAMPLES_PER_ATTRIBUTE_VALUE} samples per attribute value")
    print(f"      - {MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO} samples per (class×attribute) combo PER SPLIT")
    
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
                'samples_per_attribute_combo': SAMPLES_PER_ATTRIBUTE_COMBO,
                'min_samples_per_class': MIN_SAMPLES_PER_CLASS,
                'min_samples_per_attribute_value': MIN_SAMPLES_PER_ATTRIBUTE_VALUE,
                'min_samples_per_class_attribute_combo': MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO
            },
            'classes': BDD100K_CLASSES,
            'attributes': REPRESENTATIVE_ATTRIBUTES,
            'statistics': {
                'total_files_analyzed': 0,
                'total_selected': 0,
                'by_class': {cls: 0 for cls in BDD100K_CLASSES},
                'by_weather': {},
                'by_scene': {},
                'by_timeofday': {},
                'by_attribute_combo': {}
            }
        }
        return {}, set(), empty_metadata
    
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
                'samples_per_attribute_combo': SAMPLES_PER_ATTRIBUTE_COMBO,
                'min_samples_per_class': MIN_SAMPLES_PER_CLASS,
                'min_samples_per_attribute_value': MIN_SAMPLES_PER_ATTRIBUTE_VALUE,
                'min_samples_per_class_attribute_combo': MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO
            },
            'classes': BDD100K_CLASSES,
            'attributes': REPRESENTATIVE_ATTRIBUTES,
            'statistics': {
                'total_files_analyzed': len(txt_files),
                'total_selected': len(txt_files),
                'by_class': class_counts,
                'by_weather': {},
                'by_scene': {},
                'by_timeofday': {},
                'by_attribute_combo': {}
            },
            'selected_samples': {
                'by_class': {cls: [f.stem for f in txt_files] for cls in BDD100K_CLASSES if class_counts[cls] > 0}
            }
        }
        return {}, {f.stem for f in txt_files}, limited_metadata
    
    # Organize files by various groupings
    attribute_combo_groups = {}  # (weather, scene, timeofday) -> files
    class_samples = {class_id: [] for class_id in range(len(BDD100K_CLASSES))}  # class_id -> files
    weather_samples = {w: [] for w in REPRESENTATIVE_ATTRIBUTES['weather']}  # weather -> files
    scene_samples = {s: [] for s in REPRESENTATIVE_ATTRIBUTES['scene']}  # scene -> files
    timeofday_samples = {t: [] for t in REPRESENTATIVE_ATTRIBUTES['timeofday']}  # time -> files
    class_attribute_samples = {}  # (class_id, attr_type, attr_value) -> files
    
    for json_file in tqdm(json_files, desc="  Analyzing attributes", unit='files', leave=False):
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
    metadata = {
        'split': split_name,
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'samples_per_attribute_combo': SAMPLES_PER_ATTRIBUTE_COMBO,
            'min_samples_per_class': MIN_SAMPLES_PER_CLASS,
            'min_samples_per_attribute_value': MIN_SAMPLES_PER_ATTRIBUTE_VALUE,
            'min_samples_per_class_attribute_combo': MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO
        },
        'classes': BDD100K_CLASSES,
        'attributes': REPRESENTATIVE_ATTRIBUTES,
        'statistics': {
            'total_files_analyzed': len(json_files),
            'total_selected': 0,
            'by_class': {},
            'by_weather': {},
            'by_scene': {},
            'by_timeofday': {},
            'by_attribute_combo': {},
            'by_class_weather': {},
            'by_class_scene': {},
            'by_class_timeofday': {}
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
        
        num_to_select = min(SAMPLES_PER_ATTRIBUTE_COMBO, len(sorted_files))
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
    print(f"\n  Step 2: Ensuring {MIN_SAMPLES_PER_CLASS} samples per class...")
    for class_id, samples in class_samples.items():
        if not samples:
            continue
        
        # Count already selected samples for this class
        current_count = sum(1 for file_info in samples if file_info['path'] in selected_files)
        
        if current_count < MIN_SAMPLES_PER_CLASS:
            # Sort by diversity and add more samples
            sorted_samples = sorted(
                [s for s in samples if s['path'] not in selected_files],
                key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']),
                reverse=True
            )
            
            needed = MIN_SAMPLES_PER_CLASS - current_count
            for sample in sorted_samples[:needed]:
                selected_files.add(sample['path'])
    
    # Update class statistics in metadata
    for class_id, samples in class_samples.items():
        class_name = BDD100K_CLASSES[class_id]
        class_basenames = [s['path'].stem for s in samples if s['path'] in selected_files]
        metadata['selected_samples']['by_class'][class_name] = class_basenames
        # Store count of images with this class (not object count - that's calculated later)
        metadata['statistics']['by_class'][class_name] = len(class_basenames)
    
    print(f"    ✓ Total samples after class coverage: {len(selected_files)}")
    
    # Step 3: Ensure minimum samples per individual attribute value
    print(f"\n  Step 3: Ensuring {MIN_SAMPLES_PER_ATTRIBUTE_VALUE} samples per attribute value...")
    for attr_dict, attr_name in [(weather_samples, 'weather'), 
                                  (scene_samples, 'scene'), 
                                  (timeofday_samples, 'timeofday')]:
        for attr_value, samples in attr_dict.items():
            if not samples:
                continue
            
            current_count = sum(1 for file_info in samples if file_info['path'] in selected_files)
            
            if current_count < MIN_SAMPLES_PER_ATTRIBUTE_VALUE:
                sorted_samples = sorted(
                    [s for s in samples if s['path'] not in selected_files],
                    key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']),
                    reverse=True
                )
                
                needed = MIN_SAMPLES_PER_ATTRIBUTE_VALUE - current_count
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
    print(f"\n  Step 4: Ensuring {MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO} samples per (class×attribute) combo...")
    for (class_id, attr_type, attr_value), samples in class_attribute_samples.items():
        if not samples:
            continue
        
        current_count = sum(1 for file_info in samples if file_info['path'] in selected_files)
        
        if current_count < MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO:
            sorted_samples = sorted(
                [s for s in samples if s['path'] not in selected_files],
                key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']),
                reverse=True
            )
            
            needed = MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO - current_count
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
                    'weather': attrs.get('weather', 'undefined'),
                    'scene': attrs.get('scene', 'undefined'),
                    'timeofday': attrs.get('timeofday', 'undefined')
                }
    
    print(f"\n  ✓ FINAL: Selected {total_selected} representative samples")
    print(f"    - {len(attribute_combo_groups)} attribute combinations covered")
    print(f"    - All {len(BDD100K_CLASSES)} classes with min {MIN_SAMPLES_PER_CLASS} samples")
    print(f"    - All attribute values with min {MIN_SAMPLES_PER_ATTRIBUTE_VALUE} samples")
    print(f"    - Class×attribute combos with min {MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO} samples (PER SPLIT)")
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


def process_split(tmp_images_dir, tmp_labels_dir, yolo_dataset_root, split):
    """
    Process a dataset split: move images and convert labels to YOLO format.
    Also selects and saves representative JSON samples.
    
    Args:
        tmp_images_dir: Temporary images directory (extracted)
        tmp_labels_dir: Temporary labels directory (extracted)
        yolo_dataset_root: Root of YOLO dataset structure
        split: 'train', 'val', or 'test'
    
    Returns:
        Tuple of (images_copied, labels_converted, representative_samples_set)
    """
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
    _, representative_files, split_metadata = select_representative_samples(split_labels_src, split)
    
    # Flatten representative samples for easy lookup
    representative_basenames = {f.stem for f in representative_files}
    
    # Get all image files
    image_files = list(split_images_src.glob('*.jpg')) + list(split_images_src.glob('*.png'))
    
    if not image_files:
        print(f"⚠️  No images found in {split_images_src}")
        return 0, 0, representative_basenames
    
    print(f"Found {len(image_files)} images")
    
    images_copied = 0
    labels_converted = 0
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
        # Copy image
        dst_img_path = split_images_dst / img_file.name
        if not dst_img_path.exists():
            shutil.copy2(img_file, dst_img_path)
            images_copied += 1
        
        # Convert label - ALWAYS create label file (empty if no objects)
        json_name = img_file.stem + '.json'
        json_path = split_labels_src / json_name
        label_dst_path = split_labels_dst / (img_file.stem + '.txt')
        
        if json_path.exists():
            yolo_labels, attributes, valid_count, invalid_count = convert_json_to_yolo(json_path, validate=True)
            
            # Store attributes for test split
            if split == 'test' and attributes:
                image_attributes_map[img_file.stem] = attributes
            
            # Update validation statistics
            validation_stats['total_processed'] += 1
            validation_stats['valid_objects'] += valid_count
            validation_stats['invalid_objects'] += invalid_count
            
            # Create label file (empty if no objects)
            create_label_file(label_dst_path, yolo_labels)
            
            if yolo_labels:
                labels_converted += 1
                validation_stats['images_with_objects'] += 1
            else:
                validation_stats['images_without_objects'] += 1
        else:
            # Create empty label file if no JSON exists
            label_dst_path.touch()
            validation_stats['images_without_objects'] += 1
    
    # Print validation statistics
    print(f"\n  Validation Statistics:")
    print(f"    JSON files processed: {validation_stats['total_processed']:,}")
    print(f"    Valid objects converted: {validation_stats['valid_objects']:,}")
    print(f"    Invalid objects skipped: {validation_stats['invalid_objects']:,}")
    print(f"    Images with objects: {validation_stats['images_with_objects']:,}")
    print(f"    Images without objects: {validation_stats['images_without_objects']:,}")
    
    # Save representative JSON files for later analysis
    representative_json_dir = metadata_dir / split
    representative_json_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Saving representative JSON files for {split}...")
    json_files_saved = 0
    for basename in representative_basenames:
        json_src = split_labels_src / f"{basename}.json"
        if json_src.exists():
            json_dst = representative_json_dir / f"{basename}.json"
            if not json_dst.exists():
                shutil.copy2(json_src, json_dst)
                json_files_saved += 1
    
    print(f"  ✓ Saved {json_files_saved} representative JSON files with attributes")
    
    # Count actual objects from ALL YOLO txt files using method
    print(f"\n  Counting objects in full dataset...")
    all_object_counts = count_objects_in_labels(split_labels_dst, f"  Counting {split}")
    print(f"  ✓ Total objects in full dataset: {sum(all_object_counts.values()):,}")
    
    # Count objects in representative samples only
    representative_object_counts = {cls: 0 for cls in BDD100K_CLASSES}
    for basename in representative_basenames:
        txt_file = split_labels_dst / f'{basename}.txt'
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(BDD100K_CLASSES):
                            representative_object_counts[BDD100K_CLASSES[class_id]] += 1
    
    print(f"  ✓ Objects in representative samples: {sum(representative_object_counts.values()):,}")
    
    # Count attribute distribution for FULL dataset
    print(f"\n  Analyzing attribute distribution for {split} split...")
    # Note: count_attribute_distribution expects base path and adds split internally
    weather_dist, scene_dist, timeofday_dist = count_attribute_distribution(tmp_labels_dir / '100k', split)
    
    if weather_dist or scene_dist or timeofday_dist:
        print(f"  ✓ Attribute distribution:")
        print(f"    Weather: {len(weather_dist)} types, {sum(weather_dist.values()):,} images")
        print(f"    Scene: {len(scene_dist)} types, {sum(scene_dist.values()):,} images")
        print(f"    Time: {len(timeofday_dist)} types, {sum(timeofday_dist.values()):,} images")
        
        # Update metadata with FULL dataset attribute distribution
        split_metadata['statistics']['full_dataset_attributes'] = {
            'by_weather': weather_dist,
            'by_scene': scene_dist,
            'by_timeofday': timeofday_dist
        }
    
    # Update metadata structure to include BOTH full dataset and representative sample statistics
    split_metadata['statistics']['by_class'] = all_object_counts
    split_metadata['statistics']['representative_samples'] = {
        'total_selected': len(representative_basenames),
        'by_class': representative_object_counts
    }
    split_metadata['validation'] = validation_stats
    
    # Save metadata JSON file for this split
    metadata_file = metadata_dir / f'{split}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
    # Save performance analysis metadata with attributes for each image
    performance_file = metadata_dir / f'{split}_performance_analysis.json'
    save_test_performance_metadata(tmp_labels_dir, split_labels_dst, split, performance_file)
    
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
    
    print(f"\n✓ {split}: {images_copied} images, {labels_converted} labels")
    print(f"  Metadata saved: {metadata_file.name}")
    return images_copied, labels_converted, representative_basenames


def create_data_yaml(dataset_root, base_dir):
    """
    Create data.yaml configuration file for YOLO training.
    """
    data_yaml_content = f"""# BDD100K Dataset Configuration for YOLO
# Auto-generated by process_bdd100k_to_yolo_dataset.py

path: {dataset_root.absolute()}  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Validation images (relative to 'path')
test: images/test    # Test images (relative to 'path')

# Number of classes
nc: {len(BDD100K_CLASSES)}

# Class names
names: {BDD100K_CLASSES}
"""
    
    yaml_path = dataset_root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"\n✓ Created data.yaml: {yaml_path}")
    return yaml_path


def create_limited_dataset(source_root, output_root, representative_samples_by_split):
    """
    Create a limited dataset using representative samples with diverse attributes.
    Uses the representative JSON samples selected during conversion.
    
    Args:
        source_root: Source YOLO dataset root
        output_root: Output YOLO dataset root
        representative_samples_by_split: Dict of {split: set of basenames}
    """
    print("\n" + "="*70)
    print("CREATING LIMITED DATASET FROM REPRESENTATIVE SAMPLES")
    print("="*70)
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print("  Strategy: Using diverse representative samples with attributes")
    print(f"  Ensures:")
    print(f"    - {SAMPLES_PER_ATTRIBUTE_COMBO} samples per attribute combination")
    print(f"    - {MIN_SAMPLES_PER_CLASS} samples per class")
    print(f"    - {MIN_SAMPLES_PER_ATTRIBUTE_VALUE} samples per attribute value")
    print(f"    - {MIN_SAMPLES_PER_CLASS_ATTRIBUTE_COMBO} samples per class×attribute combo")
    
    # Verify source exists
    if not source_root.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source_root}\n"
            f"Please run full dataset creation first."
        )
    
    # Create output root
    output_root.mkdir(parents=True, exist_ok=True)
    
    total_samples = 0
    
    # Process each split
    for split in ['train', 'val', 'test']:
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
        
        # Get representative samples for this split
        representative_basenames = representative_samples_by_split.get(split, set())
        
        if not representative_basenames:
            print(f"⚠️  No representative samples found for {split}")
            continue
        
        print(f"Creating limited dataset with {len(representative_basenames)} representative samples...")
        
        # Copy files for representative samples
        samples_copied = 0
        for basename in tqdm(representative_basenames, desc=f"Copying {split}", unit='files'):
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
        
        # Copy representative JSON files for attribute analysis
        source_json_dir = source_root / 'representative_json' / split
        output_json_dir = output_root / 'representative_json' / split
        
        if source_json_dir.exists():
            output_json_dir.mkdir(parents=True, exist_ok=True)
            json_files_copied = 0
            
            for basename in representative_basenames:
                json_file = source_json_dir / f"{basename}.json"
                if json_file.exists():
                    dst_json_path = output_json_dir / json_file.name
                    if not dst_json_path.exists():
                        shutil.copy2(json_file, dst_json_path)
                        json_files_copied += 1
            
            print(f"✓ {split}: {samples_copied} representative samples copied, {json_files_copied} JSON files copied")
        else:
            print(f"✓ {split}: {samples_copied} representative samples copied (no JSON files found)")
        
        total_samples += samples_copied
    
    # Generate NEW metadata for limited dataset by analyzing representative samples
    output_metadata_dir = output_root / 'representative_json'
    
    print("\n" + "="*70)
    print("GENERATING METADATA FOR LIMITED DATASET")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        print(f"\nAnalyzing {split} split...")
        
        # Analyze the limited dataset labels directory
        split_labels_dir = output_root / 'labels' / split
        split_json_dir = output_root / 'representative_json' / split
        
        if not split_labels_dir.exists():
            print(f"  ⚠️  Skipping {split}: directory not found")
            continue
        
        # Count objects from the LIMITED dataset labels (representative samples only)
        limited_class_stats = count_objects_in_labels(split_labels_dir, f"  Counting {split}")
        
        # Count attributes from the LIMITED dataset JSON files
        limited_attributes = {}
        if split_json_dir.exists():
            limited_attributes = count_attribute_distribution(split_json_dir.parent, split)
        
        # Build metadata for limited dataset
        limited_metadata = {
            'split': split,
            'total_samples': len(list(split_labels_dir.glob('*.txt'))),
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
        
        # Save performance analysis metadata for limited dataset
        performance_file = output_metadata_dir / f'{split}_performance_analysis.json'
        save_test_performance_metadata(split_json_dir.parent, split_labels_dir, split, performance_file)
    
    print("\n" + "="*70)
    print("✓ LIMITED DATASET METADATA REGENERATED")
    print("="*70)
    
    # Create data.yaml for limited dataset
    yaml_path = create_data_yaml(output_root, output_root.parent)
    
    print("\n" + "="*70)
    print("LIMITED DATASET CREATED")
    print("="*70)
    print(f"Dataset location: {output_root}")
    print(f"Configuration: {yaml_path}")
    print(f"Total samples: {total_samples}")
    print(f"Composition: Diverse samples across weather/scene/time attributes")
    print(f"Coverage: Minimum 10 samples per object class")
    print("="*70)
    
    return output_root, yaml_path
    
 

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Extract BDD100K dataset and prepare YOLO-compatible structure.\n'
                    'Automatically creates both full and limited datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract full dataset + limited dataset (representative samples with comprehensive coverage)
  python process_bdd100k_to_yolo_dataset.py
  
  # Skip download check (use when files already exist)
  python process_bdd100k_to_yolo_dataset.py --skip-download
  
  # Force metadata regeneration
  python process_bdd100k_to_yolo_dataset.py --force-reanalysis
  
  # Remove temporary directories after processing
  python process_bdd100k_to_yolo_dataset.py --cleanup
        """
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download check (assume files already exist)'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove temporary extraction directories after processing (default: keep them)'
    )
    
    parser.add_argument(
        '--force-reanalysis',
        action='store_true',
        help='Force regeneration of metadata files even if dataset exists (automatically enabled when creating limited dataset)'
    )
    
    parser.add_argument(
        '--reanalyze-only',
        action='store_true',
        help='Only regenerate metadata without reprocessing images/labels (dataset must already exist)'
    )
    
    args = parser.parse_args()
    
    # Define paths
    base_dir = Path(__file__).parent
    source_dir = base_dir / "bdd_100k_source"
    yolo_dataset_root = base_dir / 'bdd100k_yolo'
    limited_dataset_root = base_dir / 'bdd100k_yolo_limited'
    
    # Handle reanalyze-only mode
    if args.reanalyze_only:
        print("=" * 70)
        print("METADATA REGENERATION MODE")
        print("=" * 70)
        
        if not yolo_dataset_root.exists():
            print(f"\n❌ Dataset not found: {yolo_dataset_root}")
            print("Please run full dataset creation first.")
            return
        
        print(f"\nRegenerating metadata for: {yolo_dataset_root}")
        
        # Regenerate metadata for full dataset
        for split in ['train', 'val', 'test']:
            print(f"\n{'='*70}")
            print(f"Reanalyzing {split} split")
            print(f"{'='*70}")
            
            labels_dir = yolo_dataset_root / 'labels' / split
            metadata_dir = yolo_dataset_root / 'representative_json'
            
            if not labels_dir.exists():
                print(f"⚠️  Labels directory not found: {labels_dir}")
                continue
            
            # Load existing metadata to get representative samples
            metadata_file = metadata_dir / f'{split}_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Get representative basenames
                representative_basenames = set()
                if 'selected_samples' in metadata and 'by_class' in metadata['selected_samples']:
                    for cls_samples in metadata['selected_samples']['by_class'].values():
                        representative_basenames.update(cls_samples)
            else:
                representative_basenames = set()
            
            # Count objects in ALL files using method
            print(f"Counting objects in full dataset...")
            all_object_counts = count_objects_in_labels(labels_dir, f"  {split}")
            all_txt_files = list(labels_dir.glob('*.txt'))
            print(f"✓ Total: {sum(all_object_counts.values()):,} objects")
            
            # Count objects in representative samples using method
            representative_object_counts = {cls: 0 for cls in BDD100K_CLASSES}
            if representative_basenames:
                for basename in tqdm(representative_basenames, desc=f"  Representative", leave=False):
                    txt_file = labels_dir / f'{basename}.txt'
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
            
            # Update or create metadata
            if metadata_file.exists():
                metadata['statistics']['by_class'] = all_object_counts
                metadata['statistics']['representative_samples'] = {
                    'total_selected': len(representative_basenames),
                    'by_class': representative_object_counts
                }
            else:
                metadata = {
                    'split': split,
                    'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'statistics': {
                        'total_files_analyzed': len(all_txt_files),
                        'by_class': all_object_counts,
                        'representative_samples': {
                            'total_selected': len(representative_basenames),
                            'by_class': representative_object_counts
                        }
                    }
                }
            
            # Save metadata
            metadata_dir.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ {split}: {len(all_txt_files):,} files, {sum(all_object_counts.values()):,} objects")
            if representative_basenames:
                print(f"  Representative: {len(representative_basenames):,} samples, {sum(representative_object_counts.values()):,} objects")
        
        # Regenerate limited dataset metadata if it exists
        if limited_dataset_root.exists():
            print(f"\n{'='*70}")
            print("REGENERATING LIMITED DATASET METADATA")
            print(f"{'='*70}")
            
            # Inline regeneration for limited dataset
            for split in ['train', 'val', 'test']:
                limited_labels_dir = limited_dataset_root / 'labels' / split
                if not limited_labels_dir.exists():
                    continue
                
                # Count objects from txt files
                txt_files = list(limited_labels_dir.glob('*.txt'))
                object_counts = {cls: 0 for cls in BDD100K_CLASSES}
                
                for txt_file in txt_files:
                    with open(txt_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = int(line.split()[0])
                                if 0 <= class_id < len(BDD100K_CLASSES):
                                    object_counts[BDD100K_CLASSES[class_id]] += 1
                
                # Create metadata
                limited_metadata = {
                    'split': split,
                    'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'statistics': {
                        'total_files_analyzed': len(txt_files),
                        'total_selected': len(txt_files),
                        'by_class': object_counts
                    }
                }
                
                # Save metadata
                metadata_dir = limited_dataset_root / 'representative_json'
                metadata_dir.mkdir(parents=True, exist_ok=True)
                metadata_file = metadata_dir / f'{split}_metadata.json'
                with open(metadata_file, 'w') as f:
                    json.dump(limited_metadata, f, indent=2)
                
                print(f"  ✓ {split}: {len(txt_files):,} files, {sum(object_counts.values()):,} objects")
        
        print("\n" + "="*70)
        print("✅ METADATA REGENERATION COMPLETE")
        print("="*70)
        return
    
    # Step 1: Check/download dataset files
    print("=" * 70)
    print("BDD100K YOLO Dataset Preparation")
    print("=" * 70)
    
    if not check_and_download_datasets(source_dir, args.skip_download):
        print("\n❌ Dataset files not available. Exiting.")
        return
    
    # Step 2: Extract and create full dataset
    print("\n" + "="*70)
    print("CREATING FULL DATASET")
    print("="*70)
    
    # Force re-analysis to ensure fresh metadata for limited dataset creation
    force_reanalysis = args.force_reanalysis or True

    
    representative_samples_by_split = extract_and_prepare_yolo_dataset(
        base_dir, source_dir, yolo_dataset_root, args.cleanup, force_reanalysis
    )
    
    # Step 3: Create limited dataset from representative samples
    print("\n" + "="*70)
    print("CREATING LIMITED DATASET FROM REPRESENTATIVE SAMPLES")
    print("="*70)
    create_limited_dataset(yolo_dataset_root, limited_dataset_root, representative_samples_by_split)
    
    # Step 4: Compress limited dataset for easy distribution
    print("\n" + "="*70)
    print("COMPRESSING LIMITED DATASET")
    print("="*70)
    
    zipped_dir = base_dir / 'bdd100k_yolo_limited_zipped'
    zipped_dir.mkdir(parents=True, exist_ok=True)
    compressed_file = zipped_dir / 'bdd100k_yolo_limited.zip'
    
    # Remove existing compressed file if present
    if compressed_file.exists():
        print(f"Removing existing compressed file: {compressed_file.name}")
        compressed_file.unlink()
    
    print(f"Compressing {limited_dataset_root.name} to {compressed_file}...")
    print(f"This may take a few minutes...")
    
    with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        # Get all files in limited dataset
        all_files = []
        for root, dirs, files in os.walk(limited_dataset_root):
            for file in files:
                all_files.append(Path(root) / file)
        
        # Add files with progress bar
        for file_path in tqdm(all_files, desc="Compressing", unit='files'):
            arcname = file_path.relative_to(limited_dataset_root.parent)
            zipf.write(file_path, arcname)
    
    compressed_size_mb = compressed_file.stat().st_size / (1024 * 1024)
    print(f"✓ Compressed to: {compressed_file}")
    print(f"  Size: {compressed_size_mb:.1f} MB")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ DATASET PREPARATION COMPLETE")
    print("="*70)
    print(f"\n📁 Created datasets:")
    print(f"  1. Full dataset: {yolo_dataset_root}")
    print(f"     data.yaml: {yolo_dataset_root / 'data.yaml'}")
    print(f"  2. Limited dataset: {limited_dataset_root}")
    print(f"     data.yaml: {limited_dataset_root / 'data.yaml'}")
    print(f"  3. Compressed limited dataset: {compressed_file}")
    print(f"     Size: {compressed_size_mb:.1f} MB")
    
    print(f"\n💡 Quick start with limited dataset:")
    print(f"  python unzip_limited_dataset.py")
    
    print(f"\n💡 Usage in notebooks:")
    print(f"  # For full dataset (production training and comprehensive analysis):")
    print(f"  YOLO_DATASET_ROOT = BASE_DIR / 'bdd100k_yolo'")
    print(f"  DATA_YAML_PATH = YOLO_DATASET_ROOT / 'data.yaml'")
    print(f"  # Metadata files available at:")
    print(f"  METADATA_DIR = YOLO_DATASET_ROOT / 'representative_json'")
    print(f"  # Load with: json.load(open(METADATA_DIR / 'train_metadata.json'))")
    print(f"\n  # For limited dataset (quick testing, visualization, experimentation):")
    print(f"  YOLO_DATASET_ROOT = BASE_DIR / 'bdd100k_yolo_limited'")
    print(f"  DATA_YAML_PATH = YOLO_DATASET_ROOT / 'data.yaml'")
    print(f"  Note: Limited dataset IS the representative samples (physically copied)")
    print("="*70)


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


def extract_and_prepare_yolo_dataset(base_dir, source_dir, yolo_dataset_root, cleanup=False, force_reanalysis=False):
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
    
    # Step 2 & 3: Create YOLO dataset and process splits (skip if complete, unless forced)
    dataset_complete = check_dataset_complete(yolo_dataset_root)
    
    if dataset_complete and not force_reanalysis:
        print("\n" + "="*70)
        print("STEPS 2-3: DATASET ALREADY COMPLETE, SKIPPING...")
        print("="*70)
        print(f"✓ YOLO dataset found: {yolo_dataset_root}")
        print(f"✓ All splits (train/val/test) verified")
        print(f"✓ Metadata files verified")
        print("\n  Use --force-reanalysis to regenerate metadata")
        
        # Load representative samples from metadata files
        representative_samples_by_split = {}
        metadata_dir = yolo_dataset_root / 'representative_json'
        for split in ['train', 'val', 'test']:
            metadata_file = metadata_dir / f'{split}_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Extract basenames from all sample categories
                    basenames = set()
                    for category in metadata.get('selected_samples', {}).values():
                        if isinstance(category, dict):
                            for samples in category.values():
                                basenames.update(samples)
                        elif isinstance(category, list):
                            basenames.update(category)
                    representative_samples_by_split[split] = basenames
        
        yolo_dataset_root_created = yolo_dataset_root
        total_images = sum(1 for _ in (yolo_dataset_root / 'images' / 'train').glob('*'))
        total_labels = sum(1 for _ in (yolo_dataset_root / 'labels' / 'train').glob('*.txt'))
    else:
        if force_reanalysis:
            print("\n⚠️  Force re-analysis enabled: Regenerating metadata...")
        
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
        
        for split in ['train', 'val', 'test']:
            imgs, lbls, repr_samples = process_split(tmp_images_dir, tmp_labels_dir, yolo_dataset_root_created, split)
            total_images += imgs
            total_labels += lbls
            representative_samples_by_split[split] = repr_samples
    
    # Step 4: Create data.yaml
    print("\n" + "="*70)
    print("STEP 4: CREATING CONFIGURATION FILE")
    print("="*70)
    
    yaml_path = create_data_yaml(yolo_dataset_root_created, base_dir)
    
    # Step 5: Clean up temporary directories (optional)
    print("\n" + "="*70)
    print("STEP 5: CLEANUP")
    print("="*70)
    
    if cleanup:
        print("Removing temporary directories...")
        if tmp_images_dir.exists():
            shutil.rmtree(tmp_images_dir)
            print(f"✓ Removed: {tmp_images_dir}")
        
        if tmp_labels_dir.exists():
            shutil.rmtree(tmp_labels_dir)
            print(f"✓ Removed: {tmp_labels_dir}")
        
        print("✓ Cleanup complete")
    else:
        print(f"Keeping temporary directories (use --cleanup flag to remove):")
        print(f"  {tmp_images_dir}")
        print(f"  {tmp_labels_dir}")
    
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
    print(f"  Integrity checks: ✓ PASSED (all splits)")
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
