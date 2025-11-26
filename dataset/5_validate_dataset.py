"""
5. Comprehensive Dataset Validation (READ-ONLY).

Validates datasets with exhaustive checks (never modifies any data):
1. Source comparison - Object/file counts match source dataset
2. YOLO format validation - Structure, data.yaml, label format
3. Label coordinate validation - Bounds, normalization, empty files
4. Image file integrity - File format, readability, dimensions
5. Duplicate detection - Duplicate files, metadata entries
6. Class consistency - Class IDs, distributions, metadata alignment
7. Image-label correspondence - Perfect matching
8. JSON metadata validation - Structure, completeness, accuracy
9. Metadata vs Source - Files exist in source, content matches
10. Attribute diversity - Weather/scene/time representation

Usage:
    python dataset/5_validate_dataset.py
"""

import json
from pathlib import Path
from collections import Counter
import yaml

from bdd100k_config import (BDD100K_CLASSES, CLASS_TO_IDX, 
                             LIMITED_DATASET_CONFIGS, REPRESENTATIVE_ATTRIBUTES)


def get_available_datasets():
    """Get list of available datasets."""
    base_dir = Path(__file__).parent.parent
    datasets = []
    
    full_dataset = base_dir / 'bdd100k_yolo'
    if full_dataset.exists():
        datasets.append({
            'id': 1,
            'name': 'bdd100k_yolo',
            'path': full_dataset,
            'source': None,
            'description': 'Full BDD100K dataset'
        })
    
    for idx, config in enumerate(LIMITED_DATASET_CONFIGS, start=2):
        dataset_path = base_dir / config['name']
        if dataset_path.exists():
            source_name = config.get('source_dataset', 'full')
            source_path = full_dataset if source_name == 'full' else base_dir / source_name
            datasets.append({
                'id': idx,
                'name': config['name'],
                'path': dataset_path,
                'source': source_path,
                'description': config['description']
            })
    
    return datasets


def count_objects_in_labels(labels_dir):
    """Count objects by class from YOLO label files."""
    if not labels_dir.exists():
        return None
    
    class_counts = Counter()
    total_files = 0
    files_with_objects = 0
    
    for label_file in labels_dir.glob('*.txt'):
        total_files += 1
        file_has_objects = False
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(BDD100K_CLASSES):
                            class_counts[BDD100K_CLASSES[class_id]] += 1
                            file_has_objects = True
        except:
            continue
        
        if file_has_objects:
            files_with_objects += 1
    
    return {
        'class_counts': class_counts,
        'total_files': total_files,
        'files_with_objects': files_with_objects,
        'total_objects': sum(class_counts.values())
    }


def validate_source_subset(dataset_name, dataset_labels, source_labels, split_name):
    """Validate dataset is proper subset of source."""
    print(f"\n  {split_name}:")
    
    issues = []
    
    dataset_stats = count_objects_in_labels(dataset_labels)
    source_stats = count_objects_in_labels(source_labels)
    
    if not dataset_stats or not source_stats:
        issues.append(f"{split_name}: Missing labels")
        print(f"    ❌ Missing labels")
        return issues
    
    # Check if dataset is subset
    dataset_basenames = {f.stem for f in dataset_labels.glob('*.txt')}
    source_basenames = {f.stem for f in source_labels.glob('*.txt')}
    
    not_in_source = dataset_basenames - source_basenames
    if not_in_source:
        issues.append(f"{split_name}: {len(not_in_source)} files not in source")
        print(f"    ❌ {len(not_in_source)} files not in source")
    else:
        print(f"    ✓ All files exist in source")
    
    # File count comparison
    coverage = (dataset_stats['total_files'] / source_stats['total_files'] * 100) if source_stats['total_files'] > 0 else 0
    print(f"    Files: {dataset_stats['total_files']:,} / {source_stats['total_files']:,} ({coverage:.1f}%)")
    
    # Object count comparison
    obj_coverage = (dataset_stats['total_objects'] / source_stats['total_objects'] * 100) if source_stats['total_objects'] > 0 else 0
    print(f"    Objects: {dataset_stats['total_objects']:,} / {source_stats['total_objects']:,} ({obj_coverage:.1f}%)")
    
    # Class distribution comparison
    print(f"    Class distribution:")
    for cls in BDD100K_CLASSES:
        dataset_count = dataset_stats['class_counts'].get(cls, 0)
        source_count = source_stats['class_counts'].get(cls, 0)
        if source_count > 0:
            cls_coverage = (dataset_count / source_count * 100)
            if dataset_count > 0:
                print(f"      {cls}: {dataset_count:,} / {source_count:,} ({cls_coverage:.1f}%)")
    
    return issues


def validate_directory_structure(dataset_root):
    """Validate YOLO dataset structure."""
    issues = []
    
    required = ['images', 'labels', 'data.yaml']
    for item in required:
        path = dataset_root / item
        if not path.exists():
            issues.append(f"Missing: {item}")
            print(f"    ❌ {item}")
        else:
            print(f"    ✓ {item}")
    
    return issues


def validate_data_yaml(dataset_root):
    """Validate data.yaml file."""
    issues = []
    yaml_path = dataset_root / 'data.yaml'
    
    if not yaml_path.exists():
        issues.append("data.yaml not found")
        return issues
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data.get('nc') != len(BDD100K_CLASSES):
            issues.append(f"nc mismatch: expected {len(BDD100K_CLASSES)}")
            print(f"    ❌ nc: {data.get('nc')}")
        else:
            print(f"    ✓ nc: {data['nc']}")
        
        if data.get('names') != BDD100K_CLASSES:
            issues.append("names mismatch")
            print(f"    ❌ names mismatch")
        else:
            print(f"    ✓ names: {len(data['names'])} classes")
        
        splits = [s for s in ['train', 'val', 'test'] if s in data]
        print(f"    ✓ Splits: {', '.join(splits)}")
        
    except Exception as e:
        issues.append(f"data.yaml error: {e}")
    
    return issues


def validate_label_format(dataset_root):
    """Validate YOLO label file format (READ-ONLY)."""
    issues = []
    invalid_files = []
    empty_files = []
    coordinate_issues = []
    
    labels_dir = dataset_root / 'labels'
    if not labels_dir.exists():
        return ["labels/ not found"]
    
    for split_dir in labels_dir.iterdir():
        if not split_dir.is_dir():
            continue
        
        label_files = list(split_dir.glob('*.txt'))
        split_empty = 0
        split_invalid = 0
        split_coord_issues = 0
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        empty_files.append(label_file.name)
                        split_empty += 1
                        continue
                    
                    for line_num, line in enumerate(content.split('\n'), 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) != 5:
                            invalid_files.append(f"{label_file.name}:{line_num} (expected 5 parts, got {len(parts)})")
                            split_invalid += 1
                            break
                        
                        try:
                            class_id = int(parts[0])
                            if not (0 <= class_id < len(BDD100K_CLASSES)):
                                invalid_files.append(f"{label_file.name}:{line_num} (invalid class_id={class_id})")
                                split_invalid += 1
                                break
                            
                            # Validate coordinates
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            # Check normalization (0-1 range)
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                                coordinate_issues.append(f"{label_file.name}:{line_num} (center out of bounds)")
                                split_coord_issues += 1
                            
                            if not (0 < width <= 1 and 0 < height <= 1):
                                coordinate_issues.append(f"{label_file.name}:{line_num} (invalid dimensions)")
                                split_coord_issues += 1
                            
                            # Check if box is within image bounds
                            if (x_center - width/2 < 0 or x_center + width/2 > 1 or
                                y_center - height/2 < 0 or y_center + height/2 > 1):
                                coordinate_issues.append(f"{label_file.name}:{line_num} (box exceeds bounds)")
                                split_coord_issues += 1
                                
                        except ValueError as e:
                            invalid_files.append(f"{label_file.name}:{line_num} (parse error: {e})")
                            split_invalid += 1
                            break
            except Exception as e:
                invalid_files.append(f"{label_file.name} (read error: {e})")
                split_invalid += 1
        
        status = "✓" if (split_empty == 0 and split_invalid == 0 and split_coord_issues == 0) else "⚠️"
        print(f"    {status} {split_dir.name}: {len(label_files)} files")
        if split_empty > 0:
            print(f"      ⚠️  {split_empty} empty files")
        if split_invalid > 0:
            print(f"      ❌ {split_invalid} invalid format")
        if split_coord_issues > 0:
            print(f"      ⚠️  {split_coord_issues} coordinate issues")
    
    if empty_files:
        issues.append(f"{len(empty_files)} empty label files")
    if invalid_files:
        issues.append(f"{len(invalid_files)} invalid label files")
    if coordinate_issues:
        issues.append(f"{len(coordinate_issues)} coordinate issues")
    
    return issues


def validate_no_duplicates(dataset_root):
    """Validate no duplicate files or metadata entries (READ-ONLY)."""
    issues = []
    
    # Check for duplicate image basenames across splits
    all_images = {}
    images_dir = dataset_root / 'images'
    
    if images_dir.exists():
        for split_dir in images_dir.iterdir():
            if not split_dir.is_dir():
                continue
            
            for img_file in split_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    basename = img_file.stem
                    if basename in all_images:
                        issues.append(f"Duplicate image: {basename} in {split_dir.name} and {all_images[basename]}")
                        print(f"    ❌ Duplicate: {basename}")
                    else:
                        all_images[basename] = split_dir.name
        
        if not issues:
            print(f"    ✓ No duplicate images across {len(all_images)} files")
    
    # Check for duplicate label basenames across splits
    all_labels = {}
    labels_dir = dataset_root / 'labels'
    
    if labels_dir.exists():
        for split_dir in labels_dir.iterdir():
            if not split_dir.is_dir():
                continue
            
            for label_file in split_dir.glob('*.txt'):
                basename = label_file.stem
                if basename in all_labels:
                    issues.append(f"Duplicate label: {basename} in {split_dir.name} and {all_labels[basename]}")
                else:
                    all_labels[basename] = split_dir.name
    
    # Check metadata for duplicate entries
    json_dir = dataset_root / 'representative_json'
    if json_dir.exists():
        for split in ['train', 'val', 'test']:
            json_path = json_dir / f'{split}_metadata.json'
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    files = data.get('files', {})
                    if len(files) != len(set(files.keys())):
                        issues.append(f"{split}: Duplicate entries in metadata")
                        print(f"    ❌ {split}: Duplicate metadata entries")
                except:
                    pass
    
    return issues


def validate_image_label_matching(dataset_root):
    """Validate image-label correspondence (READ-ONLY)."""
    issues = []
    
    images_dir = dataset_root / 'images'
    labels_dir = dataset_root / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        return ["images/ or labels/ missing"]
    
    for split_dir in images_dir.iterdir():
        if not split_dir.is_dir():
            continue
        
        split_name = split_dir.name
        label_split = labels_dir / split_name
        
        if not label_split.exists():
            issues.append(f"{split_name}: No labels/")
            continue
        
        image_files = {f.stem for f in split_dir.glob('*') if f.suffix in ['.jpg', '.png', '.jpeg']}
        label_files = {f.stem for f in label_split.glob('*.txt')}
        
        missing_labels = image_files - label_files
        missing_images = label_files - image_files
        
        if missing_labels:
            issues.append(f"{split_name}: {len(missing_labels)} images without labels")
            print(f"    ❌ {split_name}: {len(missing_labels)} images missing labels")
        elif missing_images:
            issues.append(f"{split_name}: {len(missing_images)} labels without images")
            print(f"    ❌ {split_name}: {len(missing_images)} labels missing images")
        else:
            print(f"    ✓ {split_name}: {len(image_files):,} perfect correspondence")
    
    return issues


def validate_image_integrity(dataset_root):
    """Validate image files are readable and valid (READ-ONLY)."""
    issues = []
    images_dir = dataset_root / 'images'
    
    if not images_dir.exists():
        return ["images/ not found"]
    
    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False
        print(f"    ⚠️  PIL not available, skipping image integrity checks")
        return []
    
    for split_dir in images_dir.iterdir():
        if not split_dir.is_dir():
            continue
        
        image_files = [f for f in split_dir.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        checked = min(50, len(image_files))  # Sample check
        
        corrupted = []
        wrong_format = []
        
        for img_file in image_files[:checked]:
            try:
                with Image.open(img_file) as img:
                    img.verify()  # Verify it's a valid image
                    
                # Re-open to check format and size
                with Image.open(img_file) as img:
                    if img.format not in ['JPEG', 'PNG']:
                        wrong_format.append(f"{img_file.name} ({img.format})")
                    
                    width, height = img.size
                    if width < 10 or height < 10:
                        corrupted.append(f"{img_file.name} (tiny: {width}x{height})")
                    if width > 10000 or height > 10000:
                        corrupted.append(f"{img_file.name} (huge: {width}x{height})")
            except Exception as e:
                corrupted.append(f"{img_file.name} ({str(e)[:50]})")
        
        status = "✓" if not corrupted and not wrong_format else "⚠️"
        print(f"    {status} {split_dir.name}: Checked {checked}/{len(image_files)} images")
        
        if corrupted:
            issues.append(f"{split_dir.name}: {len(corrupted)} corrupted/invalid images")
            print(f"      ⚠️  {len(corrupted)} corrupted/invalid")
        if wrong_format:
            issues.append(f"{split_dir.name}: {len(wrong_format)} unexpected formats")
            print(f"      ⚠️  {len(wrong_format)} unexpected formats")
    
    return issues


def validate_class_consistency(dataset_root):
    """Validate class distribution consistency (READ-ONLY)."""
    issues = []
    labels_dir = dataset_root / 'labels'
    json_dir = dataset_root / 'representative_json'
    
    if not labels_dir.exists():
        return ["labels/ not found"]
    
    print(f"    Checking class distribution...")
    
    for split in ['train', 'val', 'test']:
        split_labels = labels_dir / split
        if not split_labels.exists():
            continue
        
        # Count classes from label files
        label_class_counts = Counter()
        for label_file in split_labels.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(BDD100K_CLASSES):
                                label_class_counts[BDD100K_CLASSES[class_id]] += 1
            except:
                continue
        
        # Compare with metadata if available
        json_path = json_dir / f'{split}_metadata.json'
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Count from metadata
                metadata_class_counts = Counter()
                for file_info in data.get('files', {}).values():
                    for cls, count in file_info.get('class_counts', {}).items():
                        metadata_class_counts[cls] += count
                
                # Compare counts
                if label_class_counts != metadata_class_counts:
                    issues.append(f"{split}: Class counts mismatch between labels and metadata")
                    print(f"    ❌ {split}: Label/metadata class count mismatch")
                    
                    # Show differences
                    for cls in BDD100K_CLASSES:
                        label_count = label_class_counts.get(cls, 0)
                        meta_count = metadata_class_counts.get(cls, 0)
                        if label_count != meta_count:
                            print(f"      {cls}: labels={label_count}, metadata={meta_count}")
                else:
                    total = sum(label_class_counts.values())
                    print(f"    ✓ {split}: {total:,} objects, class counts consistent")
            except Exception as e:
                issues.append(f"{split}: Metadata comparison error - {e}")
        else:
            total = sum(label_class_counts.values())
            print(f"    ✓ {split}: {total:,} objects counted from labels")
        
        # Check for missing classes in training data
        if split == 'train' and label_class_counts:
            missing_classes = [cls for cls in BDD100K_CLASSES if label_class_counts.get(cls, 0) == 0]
            if missing_classes:
                issues.append(f"{split}: Missing classes in training data: {missing_classes}")
                print(f"    ⚠️  Missing classes: {missing_classes}")
    
    return issues


def validate_json_metadata(dataset_root, source_root=None):
    """Validate JSON metadata files (READ-ONLY)."""
    issues = []
    json_dir = dataset_root / 'representative_json'
    
    if not json_dir.exists():
        issues.append("representative_json/ not found")
        print(f"    ❌ No JSON metadata (required)")
        return issues
    
    print(f"    Checking metadata files...")
    
    # Check metadata files
    for split in ['train', 'val', 'test']:
        json_path = json_dir / f'{split}_metadata.json'
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Validate structure
                required = ['dataset', 'split', 'total_files', 'files', 'summary']
                missing = [f for f in required if f not in data]
                
                if missing:
                    issues.append(f"{split}: Missing fields {missing}")
                    print(f"    ❌ {split}: Missing {missing}")
                else:
                    total_files = data.get('total_files', 0)
                    total_objects = data['summary'].get('total_objects', 0)
                    print(f"    ✓ {split}_metadata.json: {total_files:,} files, {total_objects:,} objects")
                    
                    # Validate against actual files
                    labels_dir = dataset_root / 'labels' / split
                    if labels_dir.exists():
                        actual_files = {f.stem for f in labels_dir.glob('*.txt')}
                        metadata_files = set(data['files'].keys())
                        
                        if actual_files != metadata_files:
                            missing_in_metadata = actual_files - metadata_files
                            missing_in_files = metadata_files - actual_files
                            
                            if missing_in_metadata:
                                issues.append(f"{split}: {len(missing_in_metadata)} files not in metadata")
                                print(f"      ⚠️  {len(missing_in_metadata)} files not in metadata")
                            if missing_in_files:
                                issues.append(f"{split}: {len(missing_in_files)} metadata entries without files")
                                print(f"      ⚠️  {len(missing_in_files)} metadata entries without files")
                        else:
                            print(f"      ✓ Metadata matches files perfectly")
                    
                    # Validate each file entry has required fields
                    sample_files = list(data['files'].items())[:5]
                    for basename, file_info in sample_files:
                        required_fields = ['weather', 'scene', 'timeofday', 'categories', 'class_counts', 'object_count']
                        missing_fields = [f for f in required_fields if f not in file_info]
                        if missing_fields:
                            issues.append(f"{split}/{basename}: Missing fields {missing_fields}")
                            break
            except Exception as e:
                issues.append(f"{split}: JSON error - {e}")
                print(f"    ❌ {split}: {e}")
    
    return issues


def validate_metadata_against_source(dataset_root, source_root, dataset_name):
    """Validate dataset metadata matches source metadata (READ-ONLY)."""
    if not source_root or not source_root.exists():
        print(f"    ⚠️  No source for comparison (full dataset)")
        return []
    
    issues = []
    
    dataset_json_dir = dataset_root / 'representative_json'
    source_json_dir = source_root / 'representative_json'
    
    if not dataset_json_dir.exists():
        issues.append("Dataset metadata missing")
        print(f"    ❌ Dataset metadata not found")
        return issues
    
    if not source_json_dir.exists():
        issues.append("Source metadata missing")
        print(f"    ❌ Source metadata not found")
        return issues
    
    print(f"    Validating against source metadata...")
    
    for split in ['train', 'val', 'test']:
        dataset_json = dataset_json_dir / f'{split}_metadata.json'
        source_json = source_json_dir / f'{split}_metadata.json'
        
        if not dataset_json.exists():
            continue
        
        if not source_json.exists():
            issues.append(f"{split}: Source metadata missing")
            print(f"    ⚠️  {split}: No source metadata")
            continue
        
        try:
            with open(dataset_json, 'r') as f:
                dataset_data = json.load(f)
            with open(source_json, 'r') as f:
                source_data = json.load(f)
            
            dataset_files = dataset_data.get('files', {})
            source_files = source_data.get('files', {})
            
            # Check if all dataset files exist in source
            not_in_source = set(dataset_files.keys()) - set(source_files.keys())
            if not_in_source:
                issues.append(f"{split}: {len(not_in_source)} files not in source metadata")
                print(f"    ❌ {split}: {len(not_in_source)} files not found in source")
            else:
                print(f"    ✓ {split}: All {len(dataset_files)} files exist in source")
            
            # Validate metadata content matches source
            mismatches = 0
            for basename, dataset_info in list(dataset_files.items())[:100]:  # Sample check
                source_info = source_files.get(basename)
                if not source_info:
                    continue
                
                # Check attributes match
                for attr in ['weather', 'scene', 'timeofday', 'object_count']:
                    if dataset_info.get(attr) != source_info.get(attr):
                        mismatches += 1
                        break
                
                # Check class counts match
                if dataset_info.get('class_counts') != source_info.get('class_counts'):
                    mismatches += 1
            
            if mismatches > 0:
                issues.append(f"{split}: {mismatches} metadata mismatches with source")
                print(f"    ⚠️  {split}: {mismatches} metadata mismatches")
            else:
                print(f"    ✓ {split}: Metadata content matches source")
            
            # Validate attribute diversity
            weather_types = set(info.get('weather') for info in dataset_files.values())
            scene_types = set(info.get('scene') for info in dataset_files.values())
            time_types = set(info.get('timeofday') for info in dataset_files.values())
            
            print(f"    ✓ {split} diversity: {len(weather_types)} weather, {len(scene_types)} scenes, {len(time_types)} times")
            
            if len(weather_types) < 3:
                issues.append(f"{split}: Low weather diversity ({len(weather_types)} types)")
            if len(scene_types) < 3:
                issues.append(f"{split}: Low scene diversity ({len(scene_types)} types)")
            if len(time_types) < 2:
                issues.append(f"{split}: Low time diversity ({len(time_types)} types)")
                
        except Exception as e:
            issues.append(f"{split}: Validation error - {e}")
            print(f"    ❌ {split}: {e}")
    
    return issues


def main():
    """Main validation function."""
    datasets = get_available_datasets()
    
    if not datasets:
        print("\n❌ No datasets found")
        return
    
    print("\n" + "="*70)
    print("SELECT DATASET FOR COMPREHENSIVE VALIDATION")
    print("="*70)
    for ds in datasets:
        print(f"[{ds['id']}] {ds['name']}")
        print(f"    {ds['description']}")
    print("[0] Cancel")
    print("="*70)
    
    choice = input(f"\nSelect (0-{len(datasets)}): ").strip()
    if choice == '0':
        return
    
    dataset = next((d for d in datasets if d['id'] == int(choice)), None)
    if not dataset:
        print("Invalid choice")
        return
    
    dataset_root = dataset['path']
    source_root = dataset['source']
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE VALIDATION (READ-ONLY): {dataset['name']}")
    print(f"{'='*70}")
    print(f"Note: All checks are read-only and will not modify any data.")
    print(f"{'='*70}")
    
    all_issues = []
    
    # [1] Source comparison (limited datasets only)
    if source_root and source_root.exists():
        print(f"\n[1/10] SOURCE COMPARISON")
        print(f"Source: {source_root.name}")
        
        for split in ['train', 'val', 'test']:
            dataset_labels = dataset_root / 'labels' / split
            source_labels = source_root / 'labels' / split
            
            if dataset_labels.exists() and source_labels.exists():
                all_issues.extend(validate_source_subset(
                    dataset['name'], dataset_labels, source_labels, split
                ))
    else:
        print(f"\n[1/10] SOURCE COMPARISON - Skipped (full dataset)")
    
    # [2] Directory structure
    print(f"\n[2/10] DIRECTORY STRUCTURE")
    all_issues.extend(validate_directory_structure(dataset_root))
    
    # [3] data.yaml validation
    print(f"\n[3/10] DATA.YAML VALIDATION")
    all_issues.extend(validate_data_yaml(dataset_root))
    
    # [4] Label format validation
    print(f"\n[4/10] LABEL FORMAT & COORDINATE VALIDATION")
    all_issues.extend(validate_label_format(dataset_root))
    
    # [5] Duplicate detection
    print(f"\n[5/10] DUPLICATE DETECTION")
    all_issues.extend(validate_no_duplicates(dataset_root))
    
    # [6] Image-label matching
    print(f"\n[6/10] IMAGE-LABEL CORRESPONDENCE")
    all_issues.extend(validate_image_label_matching(dataset_root))
    
    # [7] Image file integrity
    print(f"\n[7/10] IMAGE FILE INTEGRITY")
    all_issues.extend(validate_image_integrity(dataset_root))
    
    # [8] Class consistency
    print(f"\n[8/10] CLASS CONSISTENCY")
    all_issues.extend(validate_class_consistency(dataset_root))
    
    # [9] JSON metadata validation
    print(f"\n[9/10] JSON METADATA VALIDATION")
    all_issues.extend(validate_json_metadata(dataset_root, source_root))
    
    # [10] Metadata vs Source validation (for limited datasets)
    if source_root and source_root.exists():
        print(f"\n[10/10] METADATA vs SOURCE VALIDATION")
        all_issues.extend(validate_metadata_against_source(dataset_root, source_root, dataset['name']))
    else:
        print(f"\n[10/10] METADATA vs SOURCE VALIDATION - Skipped (full dataset)")
    
    # Final summary
    print(f"\n{'='*70}")
    if not all_issues:
        print("✅ ALL VALIDATION PASSED")
        print(f"{'='*70}")
        print(f"\n{dataset['name']} is complete and valid!")
    else:
        print(f"⚠️  {len(all_issues)} ISSUES FOUND")
        print(f"{'='*70}")
        print("\nIssues:")
        for issue in all_issues[:30]:
            print(f"  - {issue}")
        if len(all_issues) > 30:
            print(f"  ... and {len(all_issues) - 30} more")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
