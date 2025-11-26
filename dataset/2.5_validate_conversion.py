"""
2.5. Validate Conversion Output.

Comprehensive validation of script 2 outputs:
- Verifies object counts match between JSON and YOLO labels
- Validates image and label file counts
- Checks directory structure completeness
- Verifies data.yaml format
- Ensures image-label correspondence

Usage:
    python dataset/2.5_validate_conversion.py
"""

import json
from pathlib import Path
from collections import Counter
import yaml

from bdd100k_config import BDD100K_CLASSES, CLASS_TO_IDX, YOLO_DATASET_ROOT


def count_objects_in_json(json_dir):
    """Count objects by class from BDD100K JSON files."""
    if not json_dir.exists():
        return None
    
    class_counts = Counter()
    total_files = 0
    files_with_objects = 0
    
    for json_file in json_dir.glob('*.json'):
        total_files += 1
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            frames = data.get('frames', [])
            if frames:
                objects = frames[0].get('objects', [])
            else:
                objects = data.get('objects', data.get('labels', []))
            
            file_has_objects = False
            for obj in objects:
                category = obj.get('category', '')
                if category in CLASS_TO_IDX and 'box2d' in obj:
                    class_counts[category] += 1
                    file_has_objects = True
            
            if file_has_objects:
                files_with_objects += 1
        except:
            continue
    
    return {
        'class_counts': class_counts,
        'total_files': total_files,
        'files_with_objects': files_with_objects
    }


def count_objects_in_yolo(labels_dir):
    """Count objects by class from YOLO label files."""
    if not labels_dir.exists():
        return None
    
    class_counts = Counter()
    total_files = 0
    files_with_objects = 0
    
    for label_file in labels_dir.glob('*.txt'):
        total_files += 1
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            file_has_objects = False
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(BDD100K_CLASSES):
                        class_name = BDD100K_CLASSES[class_id]
                        class_counts[class_name] += 1
                        file_has_objects = True
            
            if file_has_objects:
                files_with_objects += 1
        except:
            continue
    
    return {
        'class_counts': class_counts,
        'total_files': total_files,
        'files_with_objects': files_with_objects
    }


def validate_split_object_counts(split_name, json_dir, yolo_labels_dir):
    """Validate object counts match between JSON and YOLO."""
    print(f"\n{split_name.upper()}")
    print("-" * 50)
    
    issues = []
    
    # Count in JSON
    json_stats = count_objects_in_json(json_dir)
    if not json_stats:
        issues.append(f"{split_name}: Source JSON not found")
        print(f"  ⚠️  Source JSON not found: {json_dir}")
        return issues
    
    # Count in YOLO
    yolo_stats = count_objects_in_yolo(yolo_labels_dir)
    if not yolo_stats:
        issues.append(f"{split_name}: YOLO labels not found")
        print(f"  ❌ YOLO labels not found: {yolo_labels_dir}")
        return issues
    
    # Compare file counts
    print(f"\nFile Counts:")
    print(f"  JSON files: {json_stats['total_files']:,}")
    print(f"  YOLO files: {yolo_stats['total_files']:,}")
    
    if json_stats['total_files'] != yolo_stats['total_files']:
        issues.append(f"{split_name}: File count mismatch")
        print(f"  ❌ Mismatch!")
    else:
        print(f"  ✓ Match")
    
    # Compare object counts
    print(f"\nObject Counts:")
    json_total = sum(json_stats['class_counts'].values())
    yolo_total = sum(yolo_stats['class_counts'].values())
    print(f"  JSON total: {json_total:,}")
    print(f"  YOLO total: {yolo_total:,}")
    
    if json_total != yolo_total:
        issues.append(f"{split_name}: Object count mismatch ({json_total} vs {yolo_total})")
        print(f"  ❌ Mismatch!")
    else:
        print(f"  ✓ Match")
    
    # Compare per-class counts
    print(f"\nPer-Class Counts:")
    all_classes = set(json_stats['class_counts'].keys()) | set(yolo_stats['class_counts'].keys())
    mismatches = []
    
    for cls in sorted(all_classes):
        json_count = json_stats['class_counts'].get(cls, 0)
        yolo_count = yolo_stats['class_counts'].get(cls, 0)
        
        if json_count != yolo_count:
            mismatches.append(f"{cls}: {json_count} vs {yolo_count}")
            print(f"  ❌ {cls}: JSON={json_count:,}, YOLO={yolo_count:,}")
        else:
            print(f"  ✓ {cls}: {json_count:,}")
    
    if mismatches:
        issues.append(f"{split_name}: Class count mismatches - {', '.join(mismatches)}")
    
    return issues


def validate_images_exist(split_name, images_dir):
    """Validate images directory exists and has files."""
    print(f"\n{split_name.upper()} - Images")
    print("-" * 50)
    
    issues = []
    
    if not images_dir.exists():
        issues.append(f"{split_name}: Images directory not found")
        print(f"  ❌ Not found: {images_dir}")
        return issues
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    print(f"  Image files: {len(image_files):,}")
    
    if len(image_files) == 0:
        issues.append(f"{split_name}: No images found")
        print(f"  ❌ No images")
    else:
        print(f"  ✓ Images present")
    
    return issues


def validate_image_label_correspondence(split_name, images_dir, labels_dir):
    """Validate each image has corresponding label and vice versa."""
    print(f"\n{split_name.upper()} - Image/Label Correspondence")
    print("-" * 50)
    
    issues = []
    
    if not images_dir.exists() or not labels_dir.exists():
        return issues
    
    image_basenames = {f.stem for f in images_dir.glob('*') if f.suffix in ['.jpg', '.png', '.jpeg']}
    label_basenames = {f.stem for f in labels_dir.glob('*.txt')}
    
    missing_labels = image_basenames - label_basenames
    missing_images = label_basenames - image_basenames
    
    print(f"  Images: {len(image_basenames):,}")
    print(f"  Labels: {len(label_basenames):,}")
    
    if missing_labels:
        issues.append(f"{split_name}: {len(missing_labels)} images without labels")
        print(f"  ❌ {len(missing_labels)} images missing labels")
    
    if missing_images:
        issues.append(f"{split_name}: {len(missing_images)} labels without images")
        print(f"  ❌ {len(missing_images)} labels missing images")
    
    if not missing_labels and not missing_images:
        print(f"  ✓ Perfect correspondence")
    
    return issues


def validate_data_yaml(yaml_path):
    """Validate data.yaml file."""
    print(f"\nDATA.YAML Validation")
    print("-" * 50)
    
    issues = []
    
    if not yaml_path.exists():
        issues.append("data.yaml not found")
        print(f"  ❌ Not found: {yaml_path}")
        return issues
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Check required fields
        if 'path' not in data:
            issues.append("data.yaml: Missing 'path'")
            print(f"  ❌ Missing 'path'")
        else:
            print(f"  ✓ path: {data['path']}")
        
        if 'nc' not in data or data['nc'] != len(BDD100K_CLASSES):
            issues.append(f"data.yaml: nc should be {len(BDD100K_CLASSES)}")
            print(f"  ❌ nc: {data.get('nc')} (expected {len(BDD100K_CLASSES)})")
        else:
            print(f"  ✓ nc: {data['nc']}")
        
        if 'names' not in data or data['names'] != BDD100K_CLASSES:
            issues.append("data.yaml: names mismatch")
            print(f"  ❌ names mismatch")
        else:
            print(f"  ✓ names: {len(data['names'])} classes")
        
        # Check splits
        splits_found = []
        for split in ['train', 'val', 'test']:
            if split in data:
                splits_found.append(split)
        
        if len(splits_found) == 3:
            print(f"  ✓ Splits: {', '.join(splits_found)}")
        else:
            issues.append(f"data.yaml: Missing splits")
            print(f"  ⚠️  Splits: {', '.join(splits_found)}")
        
    except Exception as e:
        issues.append(f"data.yaml: Parse error - {e}")
        print(f"  ❌ Parse error: {e}")
    
    return issues


def main():
    """Main validation function."""
    base_dir = Path(__file__).parent.parent
    dataset_root = YOLO_DATASET_ROOT
    tmp_labels_dir = base_dir / 'bdd100k_tmp_labels' / '100k'
    
    print("="*70)
    print("VALIDATE SCRIPT 2 CONVERSION OUTPUT")
    print("="*70)
    print(f"Dataset: {dataset_root}")
    
    if not dataset_root.exists():
        print(f"\n❌ Dataset not found: {dataset_root}")
        print("Run script 2 first")
        return
    
    all_issues = []
    
    # Validate data.yaml
    all_issues.extend(validate_data_yaml(dataset_root / 'data.yaml'))
    
    # Validate each split
    for split in ['train', 'val', 'test']:
        json_dir = tmp_labels_dir / split
        yolo_labels_dir = dataset_root / 'labels' / split
        images_dir = dataset_root / 'images' / split
        
        # Object count validation
        print(f"\n{'='*70}")
        print(f"[1/3] OBJECT COUNTS VALIDATION - {split.upper()}")
        print(f"{'='*70}")
        all_issues.extend(validate_split_object_counts(split, json_dir, yolo_labels_dir))
        
        # Images validation
        print(f"\n{'='*70}")
        print(f"[2/3] IMAGES VALIDATION - {split.upper()}")
        print(f"{'='*70}")
        all_issues.extend(validate_images_exist(split, images_dir))
        
        # Correspondence validation
        print(f"\n{'='*70}")
        print(f"[3/3] CORRESPONDENCE VALIDATION - {split.upper()}")
        print(f"{'='*70}")
        all_issues.extend(validate_image_label_correspondence(split, images_dir, yolo_labels_dir))
    
    # Final summary
    print(f"\n{'='*70}")
    if not all_issues:
        print("✅ ALL VALIDATION PASSED")
        print("="*70)
        print("\nConversion is complete and correct!")
        print("Next step: Run script 3 to create limited datasets")
    else:
        print(f"⚠️  {len(all_issues)} ISSUES FOUND")
        print("="*70)
        print("\nIssues:")
        for issue in all_issues:
            print(f"  - {issue}")
        print("\nFix issues before proceeding")
    print("="*70)


if __name__ == '__main__':
    main()
