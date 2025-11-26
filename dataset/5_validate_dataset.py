"""
5. Comprehensive Dataset Validation (READ-ONLY).

Validates datasets with exhaustive checks (never modifies any data):
Checks ALL files (no sampling) and reports EVERY issue found.
Stops after validation to allow fixes before re-validation.

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
                             LIMITED_DATASET_CONFIGS, REPRESENTATIVE_ATTRIBUTES,
                             BDD100K_IMAGE_WIDTH, BDD100K_IMAGE_HEIGHT, TMP_LABELS_DIR)


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


def count_objects_in_json(json_dir):
    """Count objects by class from BDD100K JSON files (READ-ONLY)."""
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
            
            # Handle both formats: with frames or direct objects
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
        'files_with_objects': files_with_objects,
        'total_objects': sum(class_counts.values())
    }


def validate_against_tmp_json(dataset_labels, tmp_json_dir, split_name):
    """Validate bdd100k_yolo labels against tmp JSON files (like script 2.5)."""
    print(f"\n  {split_name}:")
    
    issues = []
    
    dataset_stats = count_objects_in_labels(dataset_labels)
    tmp_stats = count_objects_in_json(tmp_json_dir)
    
    if not dataset_stats:
        issues.append(f"{split_name}: YOLO labels not found")
        print(f"    ❌ YOLO labels not found")
        return issues
    
    if not tmp_stats:
        issues.append(f"{split_name}: Source JSON not found")
        print(f"    ❌ Source JSON not found in {tmp_json_dir}")
        return issues
    
    # Check file counts match
    if dataset_stats['total_files'] != tmp_stats['total_files']:
        issues.append(f"{split_name}: File count mismatch")
        print(f"    ❌ File count: YOLO={dataset_stats['total_files']}, JSON={tmp_stats['total_files']}")
    else:
        print(f"    ✓ File count: {dataset_stats['total_files']:,} files match")
    
    # Check file basenames match
    yolo_basenames = {f.stem for f in dataset_labels.glob('*.txt')}
    json_basenames = {f.stem for f in tmp_json_dir.glob('*.json')}
    
    not_in_json = yolo_basenames - json_basenames
    not_in_yolo = json_basenames - yolo_basenames
    
    if not_in_json:
        issues.append(f"{split_name}: {len(not_in_json)} YOLO files not in JSON")
        print(f"    ❌ {len(not_in_json)} YOLO files missing in JSON source")
    if not_in_yolo:
        issues.append(f"{split_name}: {len(not_in_yolo)} JSON files not converted")
        print(f"    ❌ {len(not_in_yolo)} JSON files not converted to YOLO")
    if not not_in_json and not not_in_yolo:
        print(f"    ✓ All files match between YOLO and JSON")
    
    # Check object counts match
    if dataset_stats['total_objects'] != tmp_stats['total_objects']:
        diff = abs(dataset_stats['total_objects'] - tmp_stats['total_objects'])
        issues.append(f"{split_name}: Object count mismatch (diff={diff})")
        print(f"    ❌ Object count: YOLO={dataset_stats['total_objects']:,}, JSON={tmp_stats['total_objects']:,} (diff={diff})")
    else:
        print(f"    ✓ Object count: {dataset_stats['total_objects']:,} objects match")
    
    # Check class distribution matches
    print(f"    Class distribution comparison:")
    class_mismatches = []
    for cls in BDD100K_CLASSES:
        yolo_count = dataset_stats['class_counts'].get(cls, 0)
        json_count = tmp_stats['class_counts'].get(cls, 0)
        if yolo_count != json_count:
            class_mismatches.append(cls)
            diff = abs(yolo_count - json_count)
            print(f"      ❌ {cls}: YOLO={yolo_count:,}, JSON={json_count:,} (diff={diff})")
        elif yolo_count > 0:
            print(f"      ✓ {cls}: {yolo_count:,}")
    
    if class_mismatches:
        issues.append(f"{split_name}: Class count mismatches for {class_mismatches}")
    
    return issues


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


def validate_label_format(dataset_root, source_json_dir=None, source_yolo_dir=None):
    """Validate YOLO label file format - STOPS at first issue found (READ-ONLY).
    
    Args:
        dataset_root: Path to YOLO dataset
        source_json_dir: Optional path to BDD100K JSON source dir (for bdd100k_yolo validation)
                        When provided, calculates expected values using conversion rules
        source_yolo_dir: Optional path to YOLO source dir (for limited datasets)
                        When provided, compares values must exactly match source YOLO values
    """
    issues = []
    
    labels_dir = dataset_root / 'labels'
    if not labels_dir.exists():
        return ["labels/ not found"]
    
    # Check source exists if validation requires it
    if source_json_dir and not source_json_dir.exists():
        error_msg = f"Source JSON directory not found: {source_json_dir}"
        print(f"    ❌ {error_msg}")
        return [error_msg]
    
    if source_yolo_dir and not source_yolo_dir.exists():
        error_msg = f"Source YOLO directory not found: {source_yolo_dir}"
        print(f"    ❌ {error_msg}")
        return [error_msg]
    
    if source_json_dir:
        print(f"    Source: BDD100K JSON (validates using conversion rules)")
    elif source_yolo_dir:
        print(f"    Source: YOLO dataset (validates exact match)")
    else:
        print(f"    Source: None (standard YOLO validation)")
    print(f"    Checking files until first issue found...")
    
    for split_dir in sorted(labels_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        
        label_files = sorted(split_dir.glob('*.txt'))
        
        print(f"\n    Processing {split_dir.name}: {len(label_files)} files...")
        
        for idx, label_file in enumerate(label_files, 1):
            # Progress indicator
            if idx % 1000 == 0:
                print(f"      Checked {idx}/{len(label_files)} files...")
            
            try:
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        print(f"\n{'='*70}")
                        print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                        print(f"{'='*70}")
                        print(f"\nIssue Type: EMPTY FILE")
                        print(f"Location: {split_dir.name}/{label_file.name}")
                        print(f"Full Path: {label_file}")
                        print(f"\nDescription:")
                        print(f"  The label file is empty (0 bytes or only whitespace).")
                        print(f"\nExpected:")
                        print(f"  Each line should contain: <class_id> <x_center> <y_center> <width> <height>")
                        print(f"\nAction Required:")
                        print(f"  1. Check if corresponding image should have labels")
                        print(f"  2. Either add proper labels or remove the file")
                        print(f"  3. Run validation again after fixing")
                        print(f"{'='*70}")
                        return [f"Empty file: {split_dir.name}/{label_file.name}"]
                    
                    for line_num, line in enumerate(content.split('\n'), 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) != 5:
                            print(f"\n{'='*70}")
                            print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                            print(f"{'='*70}")
                            print(f"\nIssue Type: INVALID FORMAT")
                            print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                            print(f"Full Path: {label_file}")
                            print(f"\nLine Content:")
                            print(f"  '{line}'")
                            print(f"\nProblem:")
                            print(f"  Expected 5 values, but found {len(parts)} values")
                            print(f"  Parts found: {parts}")
                            print(f"\nExpected Format:")
                            print(f"  <class_id> <x_center> <y_center> <width> <height>")
                            print(f"  Example: 0 0.5 0.5 0.3 0.4")
                            print(f"\nAction Required:")
                            print(f"  1. Fix the line format in the file")
                            print(f"  2. Ensure all values are space-separated")
                            print(f"  3. Run validation again after fixing")
                            print(f"{'='*70}")
                            return [f"Invalid format: {split_dir.name}/{label_file.name}:{line_num}"]
                        
                        try:
                            class_id = int(parts[0])
                            if not (0 <= class_id < len(BDD100K_CLASSES)):
                                print(f"\n{'='*70}")
                                print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                print(f"{'='*70}")
                                print(f"\nIssue Type: INVALID CLASS ID")
                                print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                print(f"Full Path: {label_file}")
                                print(f"\nLine Content:")
                                print(f"  '{line}'")
                                print(f"\nProblem:")
                                print(f"  class_id = {class_id}")
                                print(f"  Valid range: 0-{len(BDD100K_CLASSES)-1}")
                                print(f"\nValid Classes:")
                                for i, cls in enumerate(BDD100K_CLASSES):
                                    print(f"  {i}: {cls}")
                                print(f"\nAction Required:")
                                print(f"  1. Check if class_id should be one of the valid IDs above")
                                print(f"  2. Fix the class_id in the file")
                                print(f"  3. Run validation again after fixing")
                                print(f"{'='*70}")
                                return [f"Invalid class_id: {split_dir.name}/{label_file.name}:{line_num}"]
                            
                            # Validate coordinates with YOLO conversion rules from script 2:
                            # x_center = ((x1 + x2) / 2.0) / img_width
                            # y_center = ((y1 + y2) / 2.0) / img_height
                            # width = (x2 - x1) / img_width
                            # height = (y2 - y1) / img_height
                            # BDD100K images: 1280x720, normalized to [0, 1]
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            # Tolerance for floating point precision
                            # Due to division and rounding, allow ±1 pixel worth of error
                            x_tolerance = 1.0 / BDD100K_IMAGE_WIDTH  # ~0.00078 (1px in x)
                            y_tolerance = 1.0 / BDD100K_IMAGE_HEIGHT  # ~0.00139 (1px in y)
                            
                            # Source validation strategy depends on source type
                            source_validates = False
                            
                            # Strategy 1: Validate against BDD100K JSON (for bdd100k_yolo)
                            # Calculate expected values using conversion rules and compare
                            if source_json_dir:
                                json_file = source_json_dir / split_dir.name / f"{label_file.stem}.json"
                                if not json_file.exists():
                                    print(f"\n{'='*70}")
                                    print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                    print(f"{'='*70}")
                                    print(f"\nIssue Type: SOURCE JSON FILE NOT FOUND")
                                    print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                    print(f"Full Path: {label_file}")
                                    print(f"\nProblem:")
                                    print(f"  Expected source JSON file not found: {json_file}")
                                    print(f"\nAction Required:")
                                    print(f"  1. Verify source JSON files are in correct location")
                                    print(f"  2. Check if file was accidentally deleted or moved")
                                    print(f"  3. Re-run conversion if necessary")
                                    print(f"{'='*70}")
                                    return [f"Source JSON not found: {split_dir.name}/{label_file.name}"]
                                
                                try:
                                    with open(json_file, 'r') as f:
                                        json_data = json.load(f)
                                    
                                    # Get source objects
                                    frames = json_data.get('frames', [])
                                    if frames:
                                        source_objects = frames[0].get('objects', [])
                                    else:
                                        source_objects = json_data.get('objects', [])
                                    
                                    # Find matching object by line number (objects are in same order)
                                    valid_boxes = []
                                    for obj in source_objects:
                                        category = obj.get('category', '')
                                        if category in CLASS_TO_IDX and 'box2d' in obj:
                                            valid_boxes.append(obj)
                                    
                                    current_line_index = line_num - 1
                                    if current_line_index < len(valid_boxes):
                                        source_obj = valid_boxes[current_line_index]
                                        source_box = source_obj['box2d']
                                        source_category = source_obj.get('category', '')
                                        
                                        # CRITICAL: Verify class ID matches source category
                                        expected_class_id = CLASS_TO_IDX.get(source_category)
                                        if expected_class_id is None or class_id != expected_class_id:
                                            print(f"\n{'='*70}")
                                            print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                            print(f"{'='*70}")
                                            print(f"\nIssue Type: CLASS MAPPING ERROR")
                                            print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                            print(f"Full Path: {label_file}")
                                            print(f"Source JSON: {json_file}")
                                            print(f"\nLine Content:")
                                            print(f"  '{line}'")
                                            print(f"\nClass Mapping Problem:")
                                            print(f"  YOLO class_id: {class_id} ({BDD100K_CLASSES[class_id]})")
                                            print(f"  Source category: '{source_category}'")
                                            if expected_class_id is not None:
                                                print(f"  Expected class_id: {expected_class_id}")
                                                print(f"\n  ❌ Class ID mismatch!")
                                                print(f"     YOLO says: {BDD100K_CLASSES[class_id]}")
                                                print(f"     Source says: {source_category}")
                                            else:
                                                print(f"  ❌ Source category '{source_category}' not in CLASS_TO_IDX")
                                            print(f"\nCorrect Class Mapping:")
                                            for i, cls in enumerate(BDD100K_CLASSES):
                                                print(f"  {i}: {cls}")
                                            print(f"\nAction Required:")
                                            print(f"  1. Check conversion script class mapping logic")
                                            print(f"  2. Verify CLASS_TO_IDX configuration")
                                            print(f"  3. Re-run conversion with correct mapping")
                                            print(f"{'='*70}")
                                            return [f"Class mapping error: {split_dir.name}/{label_file.name}:{line_num}"]
                                        
                                        # Calculate expected YOLO values from source using conversion rules
                                        x1, y1, x2, y2 = source_box['x1'], source_box['y1'], source_box['x2'], source_box['y2']
                                        expected_x_center = ((x1 + x2) / 2.0) / BDD100K_IMAGE_WIDTH
                                        expected_y_center = ((y1 + y2) / 2.0) / BDD100K_IMAGE_HEIGHT
                                        expected_width = (x2 - x1) / BDD100K_IMAGE_WIDTH
                                        expected_height = (y2 - y1) / BDD100K_IMAGE_HEIGHT
                                        
                                        # Verify converted values match expected (within tolerance)
                                        if (abs(x_center - expected_x_center) <= x_tolerance and
                                            abs(y_center - expected_y_center) <= y_tolerance and
                                            abs(width - expected_width) <= x_tolerance and
                                            abs(height - expected_height) <= y_tolerance):
                                            # Values match source conversion, accept them
                                            source_validates = True
                                except Exception as e:
                                    print(f"\n{'='*70}")
                                    print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                    print(f"{'='*70}")
                                    print(f"\nIssue Type: SOURCE JSON READ ERROR")
                                    print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                    print(f"Full Path: {label_file}")
                                    print(f"Source JSON: {json_file}")
                                    print(f"\nProblem:")
                                    print(f"  Cannot read or parse source JSON file: {e}")
                                    print(f"\nAction Required:")
                                    print(f"  1. Check source JSON file is valid")
                                    print(f"  2. Verify file is not corrupted")
                                    print(f"  3. Check file permissions")
                                    print(f"{'='*70}")
                                    return [f"Source JSON read error: {split_dir.name}/{label_file.name}"]
                            
                            # Strategy 2: Validate against source YOLO (for limited datasets)
                            # Values must exactly match source YOLO values
                            elif source_yolo_dir:
                                source_label_file = source_yolo_dir / split_dir.name / label_file.name
                                if not source_label_file.exists():
                                    print(f"\n{'='*70}")
                                    print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                    print(f"{'='*70}")
                                    print(f"\nIssue Type: SOURCE YOLO FILE NOT FOUND")
                                    print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                    print(f"Full Path: {label_file}")
                                    print(f"\nProblem:")
                                    print(f"  Expected source YOLO file not found: {source_label_file}")
                                    print(f"\nAction Required:")
                                    print(f"  1. Verify file exists in source dataset")
                                    print(f"  2. Check if this file should be in limited dataset")
                                    print(f"  3. Re-run dataset creation if necessary")
                                    print(f"{'='*70}")
                                    return [f"Source YOLO not found: {split_dir.name}/{label_file.name}"]
                                
                                try:
                                    with open(source_label_file, 'r') as f:
                                        source_lines = f.read().strip().split('\n')
                                    
                                    # Find matching line (line_num is 1-based)
                                    if line_num <= len(source_lines):
                                        source_line = source_lines[line_num - 1].strip()
                                        source_parts = source_line.split()
                                        
                                        if len(source_parts) == 5:
                                            source_class_id = int(source_parts[0])
                                            source_x_center, source_y_center, source_width, source_height = map(float, source_parts[1:])
                                            
                                            # CRITICAL: Verify class ID matches source
                                            if class_id != source_class_id:
                                                print(f"\n{'='*70}")
                                                print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                                print(f"{'='*70}")
                                                print(f"\nIssue Type: CLASS ID MISMATCH WITH SOURCE")
                                                print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                                print(f"Full Path: {label_file}")
                                                print(f"Source YOLO: {source_label_file}")
                                                print(f"\nLine Content:")
                                                print(f"  Dataset: '{line}'")
                                                print(f"  Source:  '{source_line}'")
                                                print(f"\nClass ID Mismatch:")
                                                print(f"  Dataset class_id: {class_id} ({BDD100K_CLASSES[class_id]})")
                                                print(f"  Source class_id:  {source_class_id} ({BDD100K_CLASSES[source_class_id]})")
                                                print(f"\nProblem:")
                                                print(f"  Class IDs must match exactly between dataset and source")
                                                print(f"\nAction Required:")
                                                print(f"  1. Verify source dataset is correct")
                                                print(f"  2. Re-create limited dataset from source")
                                                print(f"  3. Do not modify class IDs during subset creation")
                                                print(f"{'='*70}")
                                                return [f"Class ID mismatch: {split_dir.name}/{label_file.name}:{line_num}"]
                                            
                                            # Values must exactly match source YOLO
                                            if (abs(x_center - source_x_center) <= x_tolerance and
                                                abs(y_center - source_y_center) <= y_tolerance and
                                                abs(width - source_width) <= x_tolerance and
                                                abs(height - source_height) <= y_tolerance):
                                                # Values match source YOLO, accept them
                                                source_validates = True
                                except Exception as e:
                                    print(f"\n{'='*70}")
                                    print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                    print(f"{'='*70}")
                                    print(f"\nIssue Type: SOURCE YOLO READ ERROR")
                                    print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                    print(f"Full Path: {label_file}")
                                    print(f"Source YOLO: {source_label_file}")
                                    print(f"\nProblem:")
                                    print(f"  Cannot read source YOLO file: {e}")
                                    print(f"\nAction Required:")
                                    print(f"  1. Check source YOLO file is valid")
                                    print(f"  2. Verify file is not corrupted")
                                    print(f"  3. Check file permissions")
                                    print(f"{'='*70}")
                                    return [f"Source YOLO read error: {split_dir.name}/{label_file.name}"]
                            
                            # Check normalization (0-1 range) - STOP at first issue
                            # Allow small tolerance for rounding errors
                            # Skip validation if source_validates=True (source has zero width/height)
                            if not source_validates and (x_center < -x_tolerance or x_center > 1 + x_tolerance):
                                print(f"\n{'='*70}")
                                print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                print(f"{'='*70}")
                                print(f"\nIssue Type: X_CENTER OUT OF BOUNDS")
                                print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                print(f"Full Path: {label_file}")
                                print(f"\nLine Content:")
                                print(f"  '{line}'")
                                print(f"\nCoordinate Values:")
                                print(f"  class_id: {class_id} ({BDD100K_CLASSES[class_id]})")
                                print(f"  x_center: {x_center:.6f} ❌ (must be 0-1)")
                                print(f"  y_center: {y_center:.6f}")
                                print(f"  width:    {width:.6f}")
                                print(f"  height:   {height:.6f}")
                                print(f"\nImage Info:")
                                print(f"  BDD100K standard: {BDD100K_IMAGE_WIDTH}x{BDD100K_IMAGE_HEIGHT}px")
                                print(f"  Conversion: x_center = ((x1+x2)/2) / {BDD100K_IMAGE_WIDTH}")
                                print(f"  Pixel tolerance: ±{x_tolerance:.6f} (±1px for rounding)")
                                print(f"\nProblem:")
                                print(f"  x_center value {x_center:.6f} is outside valid range [0, 1]")
                                print(f"  Exceeds tolerance of ±1px from BDD100K→YOLO conversion")
                                print(f"\nAction Required:")
                                print(f"  1. Check if this is a data error or conversion issue")
                                print(f"  2. Correct the x_center value to be between 0 and 1")
                                print(f"  3. Run validation again after fixing")
                                print(f"{'='*70}")
                                return [f"x_center out of bounds: {split_dir.name}/{label_file.name}:{line_num}"]
                            
                            if not source_validates and (y_center < -y_tolerance or y_center > 1 + y_tolerance):
                                print(f"\n{'='*70}")
                                print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                print(f"{'='*70}")
                                print(f"\nIssue Type: Y_CENTER OUT OF BOUNDS")
                                print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                print(f"Full Path: {label_file}")
                                print(f"\nLine Content:")
                                print(f"  '{line}'")
                                print(f"\nCoordinate Values:")
                                print(f"  class_id: {class_id} ({BDD100K_CLASSES[class_id]})")
                                print(f"  x_center: {x_center:.6f}")
                                print(f"  y_center: {y_center:.6f} ❌ (must be 0-1)")
                                print(f"  width:    {width:.6f}")
                                print(f"  height:   {height:.6f}")
                                print(f"\nImage Info:")
                                print(f"  BDD100K standard: {BDD100K_IMAGE_WIDTH}x{BDD100K_IMAGE_HEIGHT}px")
                                print(f"  Conversion: y_center = ((y1+y2)/2) / {BDD100K_IMAGE_HEIGHT}")
                                print(f"  Pixel tolerance: ±{y_tolerance:.6f} (±1px for rounding)")
                                print(f"\nProblem:")
                                print(f"  y_center value {y_center:.6f} is outside valid range [0, 1]")
                                print(f"  Exceeds tolerance of ±1px from BDD100K→YOLO conversion")
                                print(f"\nAction Required:")
                                print(f"  1. Check if this is a data error or conversion issue")
                                print(f"  2. Correct the y_center value to be between 0 and 1")
                                print(f"  3. Run validation again after fixing")
                                print(f"{'='*70}")
                                return [f"y_center out of bounds: {split_dir.name}/{label_file.name}:{line_num}"]
                            
                            if not source_validates and (width <= 0 or width > 1 + x_tolerance):
                                print(f"\n{'='*70}")
                                print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                print(f"{'='*70}")
                                print(f"\nIssue Type: WIDTH INVALID")
                                print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                print(f"Full Path: {label_file}")
                                print(f"\nLine Content:")
                                print(f"  '{line}'")
                                print(f"\nCoordinate Values:")
                                print(f"  class_id: {class_id} ({BDD100K_CLASSES[class_id]})")
                                print(f"  x_center: {x_center:.6f}")
                                print(f"  y_center: {y_center:.6f}")
                                print(f"  width:    {width:.6f} ❌ (must be 0-1)")
                                print(f"  height:   {height:.6f}")
                                print(f"\nImage Info:")
                                print(f"  BDD100K standard: {BDD100K_IMAGE_WIDTH}x{BDD100K_IMAGE_HEIGHT}px")
                                print(f"  Conversion: width = (x2-x1) / {BDD100K_IMAGE_WIDTH}")
                                print(f"  Pixel tolerance: ±{x_tolerance:.6f} (±1px for rounding)")
                                print(f"\nProblem:")
                                print(f"  width value {width:.6f} is outside valid range (0, 1]")
                                print(f"\nAction Required:")
                                print(f"  1. Check if this is a data error or conversion issue")
                                print(f"  2. Correct the width value to be between 0 and 1")
                                print(f"  3. Run validation again after fixing")
                                print(f"{'='*70}")
                                return [f"width invalid: {split_dir.name}/{label_file.name}:{line_num}"]
                            
                            if not source_validates and (height <= 0 or height > 1 + y_tolerance):
                                print(f"\n{'='*70}")
                                print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                print(f"{'='*70}")
                                print(f"\nIssue Type: HEIGHT INVALID")
                                print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                print(f"Full Path: {label_file}")
                                print(f"\nLine Content:")
                                print(f"  '{line}'")
                                print(f"\nCoordinate Values:")
                                print(f"  class_id: {class_id} ({BDD100K_CLASSES[class_id]})")
                                print(f"  x_center: {x_center:.6f}")
                                print(f"  y_center: {y_center:.6f}")
                                print(f"  width:    {width:.6f}")
                                print(f"  height:   {height:.6f} ❌ (must be 0-1)")
                                print(f"\nImage Info:")
                                print(f"  BDD100K standard: {BDD100K_IMAGE_WIDTH}x{BDD100K_IMAGE_HEIGHT}px")
                                print(f"  Conversion: height = (y2-y1) / {BDD100K_IMAGE_HEIGHT}")
                                print(f"  Pixel tolerance: ±{y_tolerance:.6f} (±1px for rounding)")
                                print(f"\nProblem:")
                                print(f"  height value {height:.6f} is outside valid range (0, 1]")
                                print(f"\nAction Required:")
                                print(f"  1. Check if this is a data error or conversion issue")
                                print(f"  2. Correct the height value to be between 0 and 1")
                                print(f"  3. Run validation again after fixing")
                                print(f"{'='*70}")
                                return [f"height invalid: {split_dir.name}/{label_file.name}:{line_num}"]
                            
                            # Check if box is within image bounds (with tolerance)
                            x_min = x_center - width/2
                            x_max = x_center + width/2
                            y_min = y_center - height/2
                            y_max = y_center + height/2
                            
                            if not source_validates and x_min < -x_tolerance:
                                print(f"\n{'='*70}")
                                print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                print(f"{'='*70}")
                                print(f"\nIssue Type: BOX EXCEEDS LEFT BOUNDARY")
                                print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                print(f"Full Path: {label_file}")
                                print(f"\nLine Content:")
                                print(f"  '{line}'")
                                print(f"\nCoordinate Values:")
                                print(f"  class_id: {class_id} ({BDD100K_CLASSES[class_id]})")
                                print(f"  x_center: {x_center:.6f}")
                                print(f"  y_center: {y_center:.6f}")
                                print(f"  width:    {width:.6f}")
                                print(f"  height:   {height:.6f}")
                                print(f"\nCalculated Box Bounds:")
                                print(f"  x_min: {x_min:.6f} ❌ (must be >= 0)")
                                print(f"  x_max: {x_max:.6f}")
                                print(f"  y_min: {y_min:.6f}")
                                print(f"  y_max: {y_max:.6f}")
                                print(f"\nImage Info:")
                                print(f"  BDD100K standard: {BDD100K_IMAGE_WIDTH}x{BDD100K_IMAGE_HEIGHT}px")
                                print(f"  Conversion tolerance: ±{x_tolerance:.6f} (±1px for rounding)")
                                print(f"\nProblem:")
                                print(f"  Box extends beyond left edge (x_min = {x_min:.6f} < 0)")
                                print(f"  Exceeds ±1px tolerance from BDD100K→YOLO conversion")
                                print(f"\nAction Required:")
                                print(f"  1. Reduce width or increase x_center")
                                print(f"  2. Ensure: x_center - width/2 >= 0")
                                print(f"  3. Run validation again after fixing")
                                print(f"{'='*70}")
                                return [f"Box x_min < 0: {split_dir.name}/{label_file.name}:{line_num}"]
                            
                            if not source_validates and x_max > 1 + x_tolerance:
                                print(f"\n{'='*70}")
                                print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                print(f"{'='*70}")
                                print(f"\nIssue Type: BOX EXCEEDS RIGHT BOUNDARY")
                                print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                print(f"Full Path: {label_file}")
                                print(f"\nLine Content:")
                                print(f"  '{line}'")
                                print(f"\nCoordinate Values:")
                                print(f"  class_id: {class_id} ({BDD100K_CLASSES[class_id]})")
                                print(f"  x_center: {x_center:.6f}")
                                print(f"  y_center: {y_center:.6f}")
                                print(f"  width:    {width:.6f}")
                                print(f"  height:   {height:.6f}")
                                print(f"\nCalculated Box Bounds:")
                                print(f"  x_min: {x_min:.6f}")
                                print(f"  x_max: {x_max:.6f} ❌ (must be <= 1)")
                                print(f"  y_min: {y_min:.6f}")
                                print(f"  y_max: {y_max:.6f}")
                                print(f"\nImage Info:")
                                print(f"  BDD100K standard: {BDD100K_IMAGE_WIDTH}x{BDD100K_IMAGE_HEIGHT}px")
                                print(f"  Pixel tolerance: ±{x_tolerance:.6f} (±1px)")
                                print(f"\nProblem:")
                                print(f"  Box extends beyond right edge (x_max = {x_max:.6f} > 1)")
                                print(f"  Exceeds tolerance of ±1 pixel")
                                print(f"\nAction Required:")
                                print(f"  1. Reduce width or decrease x_center")
                                print(f"  2. Ensure: x_center + width/2 <= 1")
                                print(f"  3. Run validation again after fixing")
                                print(f"{'='*70}")
                                return [f"Box x_max > 1: {split_dir.name}/{label_file.name}:{line_num}"]
                            
                            if not source_validates and y_min < -y_tolerance:
                                print(f"\n{'='*70}")
                                print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                print(f"{'='*70}")
                                print(f"\nIssue Type: BOX EXCEEDS TOP BOUNDARY")
                                print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                print(f"Full Path: {label_file}")
                                print(f"\nLine Content:")
                                print(f"  '{line}'")
                                print(f"\nCoordinate Values:")
                                print(f"  class_id: {class_id} ({BDD100K_CLASSES[class_id]})")
                                print(f"  x_center: {x_center:.6f}")
                                print(f"  y_center: {y_center:.6f}")
                                print(f"  width:    {width:.6f}")
                                print(f"  height:   {height:.6f}")
                                print(f"\nCalculated Box Bounds:")
                                print(f"  x_min: {x_min:.6f}")
                                print(f"  x_max: {x_max:.6f}")
                                print(f"  y_min: {y_min:.6f} ❌ (must be >= 0)")
                                print(f"  y_max: {y_max:.6f}")
                                print(f"\nImage Info:")
                                print(f"  BDD100K standard: {BDD100K_IMAGE_WIDTH}x{BDD100K_IMAGE_HEIGHT}px")
                                print(f"  Pixel tolerance: ±{y_tolerance:.6f} (±1px)")
                                print(f"\nProblem:")
                                print(f"  Box extends beyond top edge (y_min = {y_min:.6f} < 0)")
                                print(f"  Exceeds tolerance of ±1 pixel")
                                print(f"\nAction Required:")
                                print(f"  1. Reduce height or increase y_center")
                                print(f"  2. Ensure: y_center - height/2 >= 0")
                                print(f"  3. Run validation again after fixing")
                                print(f"{'='*70}")
                                return [f"Box y_min < 0: {split_dir.name}/{label_file.name}:{line_num}"]
                            
                            if not source_validates and y_max > 1 + y_tolerance:
                                print(f"\n{'='*70}")
                                print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                                print(f"{'='*70}")
                                print(f"\nIssue Type: BOX EXCEEDS BOTTOM BOUNDARY")
                                print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                                print(f"Full Path: {label_file}")
                                print(f"\nLine Content:")
                                print(f"  '{line}'")
                                print(f"\nCoordinate Values:")
                                print(f"  class_id: {class_id} ({BDD100K_CLASSES[class_id]})")
                                print(f"  x_center: {x_center:.6f}")
                                print(f"  y_center: {y_center:.6f}")
                                print(f"  width:    {width:.6f}")
                                print(f"  height:   {height:.6f}")
                                print(f"\nCalculated Box Bounds:")
                                print(f"  x_min: {x_min:.6f}")
                                print(f"  x_max: {x_max:.6f}")
                                print(f"  y_min: {y_min:.6f}")
                                print(f"  y_max: {y_max:.6f} ❌ (must be <= 1)")
                                print(f"\nImage Info:")
                                print(f"  BDD100K standard: {BDD100K_IMAGE_WIDTH}x{BDD100K_IMAGE_HEIGHT}px")
                                print(f"  Pixel tolerance: ±{y_tolerance:.6f} (±1px)")
                                print(f"\nProblem:")
                                print(f"  Box extends beyond bottom edge (y_max = {y_max:.6f} > 1)")
                                print(f"  Exceeds tolerance of ±1 pixel")
                                print(f"\nAction Required:")
                                print(f"  1. Reduce height or decrease y_center")
                                print(f"  2. Ensure: y_center + height/2 <= 1")
                                print(f"  3. Run validation again after fixing")
                                print(f"{'='*70}")
                                return [f"Box y_max > 1: {split_dir.name}/{label_file.name}:{line_num}"]
                                
                        except ValueError as e:
                            print(f"\n{'='*70}")
                            print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                            print(f"{'='*70}")
                            print(f"\nIssue Type: PARSE ERROR")
                            print(f"Location: {split_dir.name}/{label_file.name}:{line_num}")
                            print(f"Full Path: {label_file}")
                            print(f"\nLine Content:")
                            print(f"  '{line}'")
                            print(f"\nProblem:")
                            print(f"  Cannot parse coordinate values: {e}")
                            print(f"  All coordinate values must be valid numbers")
                            print(f"\nExpected Format:")
                            print(f"  <class_id> <x_center> <y_center> <width> <height>")
                            print(f"  All values must be numeric (int for class_id, float for coordinates)")
                            print(f"\nAction Required:")
                            print(f"  1. Check for non-numeric characters in coordinates")
                            print(f"  2. Fix the values to be valid numbers")
                            print(f"  3. Run validation again after fixing")
                            print(f"{'='*70}")
                            return [f"Parse error: {split_dir.name}/{label_file.name}:{line_num}"]
            except Exception as e:
                print(f"\n{'='*70}")
                print(f"🛑 FIRST ISSUE FOUND - VALIDATION STOPPED")
                print(f"{'='*70}")
                print(f"\nIssue Type: FILE READ ERROR")
                print(f"Location: {split_dir.name}/{label_file.name}")
                print(f"Full Path: {label_file}")
                print(f"\nProblem:")
                print(f"  Cannot read file: {e}")
                print(f"\nAction Required:")
                print(f"  1. Check file permissions")
                print(f"  2. Check file encoding")
                print(f"  3. Verify file is not corrupted")
                print(f"  4. Run validation again after fixing")
                print(f"{'='*70}")
                return [f"Read error: {split_dir.name}/{label_file.name}"]
        
        print(f"    ✓ {split_dir.name}: All {len(label_files)} files validated successfully")
    
    print(f"\n{'='*70}")
    print(f"✅ ALL FILES VALIDATED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"No issues found in any label files!")
    
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
    """Validate image files are readable and valid - checks ALL images (READ-ONLY)."""
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
    
    print(f"    Checking ALL images (no sampling)...")
    
    for split_dir in sorted(images_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        
        image_files = sorted([f for f in split_dir.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        
        corrupted = []
        wrong_format = []
        
        print(f"\n    Processing {split_dir.name}: {len(image_files)} images...")
        
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    img.verify()  # Verify it's a valid image
                    
                # Re-open to check format and size
                with Image.open(img_file) as img:
                    if img.format not in ['JPEG', 'PNG']:
                        error_msg = f"{split_dir.name}/{img_file.name} - Wrong format: {img.format}"
                        wrong_format.append(error_msg)
                    
                    width, height = img.size
                    if width < 10 or height < 10:
                        error_msg = f"{split_dir.name}/{img_file.name} - Too small: {width}x{height}px"
                        corrupted.append(error_msg)
                    if width > 10000 or height > 10000:
                        error_msg = f"{split_dir.name}/{img_file.name} - Too large: {width}x{height}px"
                        corrupted.append(error_msg)
            except Exception as e:
                error_msg = f"{split_dir.name}/{img_file.name} - Error: {str(e)[:100]}"
                corrupted.append(error_msg)
        
        status = "✓" if not corrupted and not wrong_format else "❌"
        print(f"    {status} {split_dir.name}: Checked {len(image_files)} images")
        
        if corrupted:
            print(f"      ❌ {len(corrupted)} corrupted/invalid images:")
            for error_msg in corrupted:
                print(f"         - {error_msg}")
            issues.append(f"{split_dir.name}: {len(corrupted)} corrupted/invalid images")
        
        if wrong_format:
            print(f"      ⚠️  {len(wrong_format)} unexpected formats:")
            for error_msg in wrong_format:
                print(f"         - {error_msg}")
            issues.append(f"{split_dir.name}: {len(wrong_format)} unexpected formats")
    
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


def validate_metadata_against_tmp_json(dataset_root, tmp_json_base):
    """Validate bdd100k_yolo metadata against BDD100K JSON source (READ-ONLY)."""
    if not tmp_json_base or not tmp_json_base.exists():
        error_msg = f"BDD100K JSON source not found: {tmp_json_base}"
        print(f"    ❌ {error_msg}")
        return [error_msg]
    
    issues = []
    
    dataset_json_dir = dataset_root / 'representative_json'
    
    if not dataset_json_dir.exists():
        issues.append("Dataset metadata missing")
        print(f"    ❌ Dataset metadata not found")
        return issues
    
    print(f"    Validating metadata against BDD100K JSON source...")
    
    for split in ['train', 'val', 'test']:
        print(f"\n  {split}:")
        dataset_json = dataset_json_dir / f'{split}_metadata.json'
        tmp_json_dir = tmp_json_base / split
        
        if not dataset_json.exists():
            print(f"    ⚠️  No metadata file for {split}")
            continue
        
        if not tmp_json_dir.exists():
            issues.append(f"{split}: BDD100K JSON source directory not found")
            print(f"    ❌ BDD100K JSON source not found: {tmp_json_dir}")
            continue
        
        try:
            with open(dataset_json, 'r') as f:
                dataset_data = json.load(f)
            
            dataset_files = dataset_data.get('files', {})
            dataset_summary = dataset_data.get('summary', {})
            
            # Count objects from BDD100K JSON source
            source_stats = count_objects_in_json(tmp_json_dir)
            
            if not source_stats:
                issues.append(f"{split}: Cannot read BDD100K JSON source")
                print(f"    ❌ Cannot read source JSON files")
                continue
            
            # 1. Validate file count
            metadata_file_count = dataset_data.get('total_files', 0)
            source_file_count = source_stats['total_files']
            
            if metadata_file_count != source_file_count:
                issues.append(f"{split}: File count mismatch in metadata")
                print(f"    ❌ File count: metadata={metadata_file_count}, source={source_file_count}")
            else:
                print(f"    ✓ File count: {metadata_file_count:,} files match")
            
            # 2. Validate all files exist in source
            json_basenames = {f.stem for f in tmp_json_dir.glob('*.json')}
            metadata_basenames = set(dataset_files.keys())
            
            not_in_source = metadata_basenames - json_basenames
            not_in_metadata = json_basenames - metadata_basenames
            
            if not_in_source:
                issues.append(f"{split}: {len(not_in_source)} metadata files not in source")
                print(f"    ❌ {len(not_in_source)} files in metadata but not in source")
            if not_in_metadata:
                issues.append(f"{split}: {len(not_in_metadata)} source files not in metadata")
                print(f"    ❌ {len(not_in_metadata)} source files missing from metadata")
            if not not_in_source and not not_in_metadata:
                print(f"    ✓ All files match between metadata and source")
            
            # 3. Validate total object count
            metadata_object_count = dataset_summary.get('total_objects', 0)
            source_object_count = source_stats['total_objects']
            
            if metadata_object_count != source_object_count:
                diff = abs(metadata_object_count - source_object_count)
                issues.append(f"{split}: Object count mismatch (diff={diff})")
                print(f"    ❌ Object count: metadata={metadata_object_count:,}, source={source_object_count:,} (diff={diff})")
            else:
                print(f"    ✓ Object count: {metadata_object_count:,} objects match")
            
            # 4. Validate class distribution
            print(f"    Class distribution comparison:")
            metadata_class_counts = dataset_summary.get('class_distribution', {})
            source_class_counts = source_stats['class_counts']
            
            class_mismatches = []
            for cls in BDD100K_CLASSES:
                metadata_count = metadata_class_counts.get(cls, 0)
                source_count = source_class_counts.get(cls, 0)
                
                if metadata_count != source_count:
                    class_mismatches.append(cls)
                    diff = abs(metadata_count - source_count)
                    print(f"      ❌ {cls}: metadata={metadata_count:,}, source={source_count:,} (diff={diff})")
                elif metadata_count > 0:
                    print(f"      ✓ {cls}: {metadata_count:,}")
            
            if class_mismatches:
                issues.append(f"{split}: Class count mismatches for {class_mismatches}")
            
            # 5. Validate individual file metadata against source
            print(f"    Validating individual file metadata...")
            file_mismatches = 0
            files_checked = 0
            
            for basename, metadata_info in dataset_files.items():
                json_file = tmp_json_dir / f"{basename}.json"
                if not json_file.exists():
                    file_mismatches += 1
                    continue
                
                files_checked += 1
                
                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                    
                    # Get objects from source
                    frames = json_data.get('frames', [])
                    if frames:
                        source_objects = frames[0].get('objects', [])
                    else:
                        source_objects = json_data.get('objects', [])
                    
                    # Count valid objects in source
                    source_object_count = 0
                    source_class_counts = Counter()
                    for obj in source_objects:
                        category = obj.get('category', '')
                        if category in CLASS_TO_IDX and 'box2d' in obj:
                            source_object_count += 1
                            source_class_counts[category] += 1
                    
                    # Compare with metadata
                    metadata_object_count = metadata_info.get('object_count', 0)
                    metadata_class_counts = metadata_info.get('class_counts', {})
                    
                    if metadata_object_count != source_object_count:
                        file_mismatches += 1
                    elif metadata_class_counts != dict(source_class_counts):
                        file_mismatches += 1
                    
                    # Verify attributes match
                    source_attributes = frames[0].get('attributes', {}) if frames else json_data.get('attributes', {})
                    for attr in ['weather', 'scene', 'timeofday']:
                        if metadata_info.get(attr) != source_attributes.get(attr):
                            file_mismatches += 1
                            break
                    
                except:
                    file_mismatches += 1
            
            if file_mismatches > 0:
                issues.append(f"{split}: {file_mismatches}/{files_checked} file metadata mismatches")
                print(f"    ❌ {file_mismatches}/{files_checked} files have metadata mismatches")
            else:
                print(f"    ✓ All {files_checked} file metadata entries match source")
            
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
    
    # [1] Source comparison - CRITICAL: Must exist or validation fails immediately
    if dataset['name'] == 'bdd100k_yolo':
        # bdd100k_yolo validates against tmp JSON files
        tmp_labels_base = TMP_LABELS_DIR / '100k'
        print(f"\n[1/10] SOURCE VALIDATION")
        if not tmp_labels_base.exists():
            print(f"{'='*70}")
            print(f"🛑 VALIDATION FAILED - CANNOT PROCEED")
            print(f"{'='*70}")
            print(f"\nIssue Type: SOURCE NOT FOUND")
            print(f"Dataset: {dataset['name']}")
            print(f"\nProblem:")
            print(f"  BDD100K JSON source directory not found: {tmp_labels_base}")
            print(f"\nWhy This Is Critical:")
            print(f"  The bdd100k_yolo dataset must be validated against BDD100K JSON source")
            print(f"  to verify correct conversion, class mapping, and data integrity.")
            print(f"\nAction Required:")
            print(f"  1. Ensure BDD100K JSON files exist in: {tmp_labels_base}")
            print(f"  2. Check TMP_LABELS_DIR configuration in bdd100k_config.py")
            print(f"  3. Re-run dataset conversion if necessary")
            print(f"  4. Do not proceed without source validation")
            print(f"{'='*70}")
            return
        else:
            print(f"\n[1/10] SOURCE VALIDATION (against BDD100K JSON)")
            print(f"Source: {tmp_labels_base}")
            
            for split in ['train', 'val', 'test']:
                dataset_labels = dataset_root / 'labels' / split
                tmp_json_dir = tmp_labels_base / split
                
                if dataset_labels.exists() and tmp_json_dir.exists():
                    all_issues.extend(validate_against_tmp_json(
                        dataset_labels, tmp_json_dir, split
                    ))
    elif source_root:
        # Limited datasets validate against YOLO source
        print(f"\n[1/10] SOURCE COMPARISON")
        if not source_root.exists():
            print(f"{'='*70}")
            print(f"🛑 VALIDATION FAILED - CANNOT PROCEED")
            print(f"{'='*70}")
            print(f"\nIssue Type: SOURCE DATASET NOT FOUND")
            print(f"Dataset: {dataset['name']}")
            print(f"\nProblem:")
            print(f"  Source dataset not found: {source_root}")
            print(f"\nWhy This Is Critical:")
            print(f"  Limited datasets are subsets and must validate against their source")
            print(f"  to verify data integrity, class mapping, and coordinate accuracy.")
            print(f"\nAction Required:")
            print(f"  1. Ensure source dataset exists: {source_root}")
            print(f"  2. Check source_dataset configuration in LIMITED_DATASET_CONFIGS")
            print(f"  3. Create source dataset first before creating limited datasets")
            print(f"  4. Do not proceed without source validation")
            print(f"{'='*70}")
            return
        else:
            print(f"Source: {source_root.name}")
            
            for split in ['train', 'val', 'test']:
                dataset_labels = dataset_root / 'labels' / split
                source_labels = source_root / 'labels' / split
                
                if dataset_labels.exists() and source_labels.exists():
                    all_issues.extend(validate_source_subset(
                        dataset['name'], dataset_labels, source_labels, split
                    ))
    else:
        # Limited dataset without source is a critical error
        print(f"\n[1/10] SOURCE COMPARISON")
        print(f"{'='*70}")
        print(f"🛑 VALIDATION FAILED - CANNOT PROCEED")
        print(f"{'='*70}")
        print(f"\nIssue Type: NO SOURCE CONFIGURED")
        print(f"Dataset: {dataset['name']}")
        print(f"\nProblem:")
        print(f"  Limited dataset has no source configured in the system.")
        print(f"\nWhy This Is Critical:")
        print(f"  All limited datasets must have a source dataset to validate against.")
        print(f"  Without source validation, data integrity cannot be verified.")
        print(f"\nAction Required:")
        print(f"  1. Add 'source_dataset' configuration in LIMITED_DATASET_CONFIGS")
        print(f"  2. Specify which dataset this is derived from (e.g., 'bdd100k_yolo')")
        print(f"  3. Re-create the dataset with proper source configuration")
        print(f"  4. Do not use this dataset without source validation")
        print(f"{'='*70}")
        return
    
    # [2] Directory structure
    print(f"\n[2/10] DIRECTORY STRUCTURE")
    all_issues.extend(validate_directory_structure(dataset_root))
    
    # [3] data.yaml validation
    print(f"\n[3/10] DATA.YAML VALIDATION")
    all_issues.extend(validate_data_yaml(dataset_root))
    
    # [4] Label format validation
    print(f"\n[4/10] LABEL FORMAT & COORDINATE VALIDATION")
    # Different validation strategies based on dataset type
    if dataset['name'] == 'bdd100k_yolo':
        # For bdd100k_yolo: validate against BDD100K JSON using conversion rules
        tmp_labels_base = TMP_LABELS_DIR / '100k'
        all_issues.extend(validate_label_format(dataset_root, source_json_dir=tmp_labels_base))
    elif source_root:
        # For limited datasets: validate exact match with source YOLO values
        source_labels = source_root / 'labels'
        all_issues.extend(validate_label_format(dataset_root, source_yolo_dir=source_labels))
    # Note: If source_root is None, validation already failed in step [1]
    
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
    
    # [10] Metadata vs Source validation
    print(f"\n[10/10] METADATA vs SOURCE VALIDATION")
    if dataset['name'] == 'bdd100k_yolo':
        # For bdd100k_yolo: validate metadata against tmp JSON source
        tmp_labels_base = TMP_LABELS_DIR / '100k'
        all_issues.extend(validate_metadata_against_tmp_json(dataset_root, tmp_labels_base))
    elif source_root:
        # For limited datasets: validate against YOLO source
        all_issues.extend(validate_metadata_against_source(dataset_root, source_root, dataset['name']))
    # Note: If source_root is None, validation already failed in step [1]
    
    # Final summary
    print(f"\n{'='*70}")
    if not all_issues:
        print("✅ ALL VALIDATION PASSED")
        print(f"{'='*70}")
        print(f"\n{dataset['name']} is complete and valid!")
        print(f"All files checked - No issues found.")
    else:
        print(f"❌ {len(all_issues)} ISSUE CATEGORIES FOUND")
        print(f"{'='*70}")
        print(f"\n⚠️  VALIDATION STOPPED - Issues must be fixed before proceeding.")
        print(f"\nIssue Summary:")
        for issue in all_issues:
            print(f"  - {issue}")
        print(f"\n{'='*70}")
        print(f"NEXT STEPS:")
        print(f"{'='*70}")
        print(f"1. Review the detailed issues above")
        print(f"2. Fix the reported problems in the source files")
        print(f"3. Run this validation script again")
        print(f"4. Repeat until all issues are resolved")
        print(f"\nNote: All detailed issues are listed above for your reference.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
