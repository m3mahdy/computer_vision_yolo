#!/usr/bin/env python3
"""
Validate All Hardcoded Values in process_bdd100k_to_yolo_dataset.py

This script validates:
1. Class names match actual BDD100K categories
2. Attribute values match actual dataset
3. Image dimensions are consistent
4. Number of classes is correct
"""

import json
from pathlib import Path
from collections import Counter
from PIL import Image
import sys


# Hardcoded values from the script (to validate)
EXPECTED_CLASSES = [
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

EXPECTED_ATTRIBUTES = {
    'weather': ['clear', 'foggy', 'overcast', 'partly cloudy', 'rainy', 'snowy', 'undefined'],
    'scene': ['city street', 'gas stations', 'highway', 'parking lot', 'residential', 'tunnel', 'undefined'],
    'timeofday': ['daytime', 'night', 'dawn/dusk', 'undefined']
}


def validate_classes(base_dir: Path):
    """Validate that class names match actual BDD100K categories."""
    print("=" * 80)
    print("1. VALIDATING CLASS NAMES")
    print("=" * 80)
    
    labels_dir = base_dir / 'bdd100k_tmp_labels' / '100k'
    
    # Collect all unique categories from actual data
    all_categories = set()
    detection_categories = set()  # Only detection classes (no prefix)
    file_count = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue
        
        print(f"\nProcessing {split} split...")
        split_files = 0
        
        for json_file in split_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Handle different BDD100K label formats
                frames = data.get('frames', [])
                if frames:
                    objects = frames[0].get('objects', [])
                else:
                    objects = data.get('objects', data.get('labels', []))
                
                for obj in objects:
                    category = obj.get('category', '')
                    if category:
                        all_categories.add(category)
                        # Detection classes have no prefix (no '/')
                        if '/' not in category:
                            detection_categories.add(category)
                
                file_count += 1
                split_files += 1
                
                # Progress indicator
                if split_files % 10000 == 0:
                    print(f"  Processed {split_files:,} files...")
                
            except Exception as e:
                print(f"  Warning: Could not read {json_file.name}: {e}")
                continue
        
        print(f"  Completed {split} split: {split_files:,} files")
    
    print(f"\nTotal label files checked: {file_count:,}")
    print(f"\nAll categories found ({len(all_categories)}): {sorted(all_categories)}")
    print(f"\nDetection categories (no prefix) found ({len(detection_categories)}): {sorted(detection_categories)}")
    print(f"Segmentation categories (area/*, lane/*) found: {len(all_categories) - len(detection_categories)}")
    print(f"\nExpected detection classes ({len(EXPECTED_CLASSES)}): {EXPECTED_CLASSES}")
    
    # Check for mismatches (only compare detection classes)
    missing_in_actual = set(EXPECTED_CLASSES) - detection_categories
    extra_in_actual = detection_categories - set(EXPECTED_CLASSES)
    
    if missing_in_actual:
        print(f"\n❌ MISSING in actual data: {missing_in_actual}")
    
    if extra_in_actual:
        print(f"\n⚠️  EXTRA detection classes in actual data (not in hardcoded list): {extra_in_actual}")
    
    if not missing_in_actual and not extra_in_actual:
        print("\n✓ Detection class names MATCH perfectly!")
        return True
    else:
        print("\n❌ Detection class names MISMATCH detected!")
        return False


def validate_attributes(base_dir: Path):
    """Validate attribute values match actual dataset."""
    print("\n" + "=" * 80)
    print("2. VALIDATING ATTRIBUTE VALUES")
    print("=" * 80)
    
    labels_dir = base_dir / 'bdd100k_tmp_labels' / '100k'
    
    # Collect actual attribute values with counts
    actual_attributes = {
        'weather': {},
        'scene': {},
        'timeofday': {}
    }
    
    file_count = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue
        
        print(f"\nProcessing {split} split...")
        split_files = 0
        
        for json_file in split_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                attrs = data.get('attributes', {})
                for attr_name in ['weather', 'scene', 'timeofday']:
                    if attr_name in attrs:
                        value = attrs[attr_name]
                        actual_attributes[attr_name][value] = actual_attributes[attr_name].get(value, 0) + 1
                
                file_count += 1
                split_files += 1
                
                # Progress indicator
                if split_files % 10000 == 0:
                    print(f"  Processed {split_files:,} files...")
                
            except Exception as e:
                print(f"  Warning: Could not read {json_file.name}: {e}")
                continue
        
        print(f"  Completed {split} split: {split_files:,} files")
    
    print(f"\nTotal label files checked: {file_count:,}\n")
    
    all_match = True
    
    for attr_name, expected_values in EXPECTED_ATTRIBUTES.items():
        actual_value_counts = actual_attributes[attr_name]
        actual_values = sorted(actual_value_counts.keys())
        expected_sorted = sorted(expected_values)
        
        print(f"{attr_name.upper()}:")
        print(f"  Expected ({len(expected_sorted)}): {expected_sorted}")
        print(f"  Actual   ({len(actual_values)}): {actual_values}")
        
        # Show counts for each value
        print(f"  Value frequencies:")
        for value in sorted(actual_value_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {value[0]}: {value[1]:,} occurrences")
        
        missing = set(expected_sorted) - set(actual_values)
        extra = set(actual_values) - set(expected_sorted)
        
        if missing:
            print(f"  ❌ Missing in actual: {missing}")
            all_match = False
        if extra:
            print(f"  ⚠️  Extra in actual: {extra}")
            all_match = False
        if not missing and not extra:
            print(f"  ✓ MATCH")
        print()
    
    if all_match:
        print("✓ All attribute values validated!")
        return True
    else:
        print("❌ Attribute values MISMATCH detected!")
        return False


def validate_image_dimensions(base_dir: Path):
    """Validate image dimensions are consistent."""
    print("=" * 80)
    print("3. VALIDATING IMAGE DIMENSIONS")
    print("=" * 80)
    
    images_dir = base_dir / 'bdd100k_tmp_images' / '100k'
    
    dimension_counts = Counter()
    image_count = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = images_dir / split
        if not split_dir.exists():
            continue
        
        print(f"\nProcessing {split} split...")
        split_images = 0
        
        for img_file in split_dir.glob('*.jpg'):
            try:
                img = Image.open(img_file)
                dimension_counts[(img.width, img.height)] += 1
                image_count += 1
                split_images += 1
                
                # Progress indicator
                if split_images % 10000 == 0:
                    print(f"  Processed {split_images:,} images...")
                
            except Exception as e:
                print(f"  Warning: Could not read {img_file.name}: {e}")
                continue
        
        print(f"  Completed {split} split: {split_images:,} images")
    
    print(f"\nTotal images checked: {image_count:,}\n")
    
    if len(dimension_counts) == 0:
        print("❌ No images found to validate!")
        return False
    
    print("Dimensions found:")
    for (width, height), count in dimension_counts.most_common():
        percentage = (count / image_count * 100)
        print(f"  {width} × {height}: {count:,} images ({percentage:.1f}%)")
    
    if len(dimension_counts) == 1:
        width, height = list(dimension_counts.keys())[0]
        print(f"\n✓ All images have consistent dimensions: {width} × {height}")
        print(f"  Recommended: BDD100K_IMAGE_WIDTH = {width}")
        print(f"  Recommended: BDD100K_IMAGE_HEIGHT = {height}")
        return True
    else:
        print(f"\n⚠️  Multiple dimensions found! Dataset has inconsistent image sizes.")
        print(f"  Most common: {dimension_counts.most_common(1)[0][0]}")
        return False


def validate_dataset_structure(base_dir: Path):
    """Validate expected dataset structure exists."""
    print("\n" + "=" * 80)
    print("4. VALIDATING DATASET STRUCTURE")
    print("=" * 80)
    
    expected_dirs = [
        'bdd100k_tmp_labels/100k/train',
        'bdd100k_tmp_labels/100k/val',
        'bdd100k_tmp_labels/100k/test',
        'bdd100k_tmp_images/100k/train',
        'bdd100k_tmp_images/100k/val',
        'bdd100k_tmp_images/100k/test'
    ]
    
    all_exist = True
    for dir_path in expected_dirs:
        full_path = base_dir / dir_path
        exists = full_path.exists()
        status = "✓" if exists else "❌"
        print(f"  {status} {dir_path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✓ All expected directories exist!")
        return True
    else:
        print("\n❌ Some expected directories are missing!")
        return False


def validate_number_of_classes():
    """Validate the number of classes."""
    print("\n" + "=" * 80)
    print("5. VALIDATING NUMBER OF CLASSES")
    print("=" * 80)
    
    num_classes = len(EXPECTED_CLASSES)
    print(f"Number of classes defined: {num_classes}")
    
    # BDD100K detection task should have 10 classes
    if num_classes == 10:
        print("✓ Correct! BDD100K detection task has 10 classes")
        return True
    else:
        print(f"⚠️  Expected 10 classes for BDD100K detection, but found {num_classes}")
        return False


def main():
    base_dir = Path.cwd()
    
    print("BDD100K HARDCODED VALUES VALIDATION")
    print("Base directory:", base_dir)
    print()
    
    results = {
        'structure': validate_dataset_structure(base_dir),
        'classes': validate_classes(base_dir),
        'attributes': validate_attributes(base_dir),
        'dimensions': validate_image_dimensions(base_dir),
        'num_classes': validate_number_of_classes()
    }
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED!")
        print("All hardcoded values match the actual BDD100K dataset.")
        sys.exit(0)
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("Review the hardcoded values in process_bdd100k_to_yolo_dataset.py")
        sys.exit(1)


if __name__ == '__main__':
    main()
