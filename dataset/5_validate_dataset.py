"""
5. Comprehensive Dataset Validation.

Validates limited datasets against their source datasets with complete checks:
1. Source comparison - Object/file counts match source dataset
2. YOLO format validation - Structure, data.yaml, label format
3. JSON metadata validation - Metadata files correctness
4. Image-label integrity - All images have labels and vice versa
5. Attribute distribution - Weather/scene/time diversity maintained

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
    """Validate YOLO label file format."""
    issues = []
    invalid_files = []
    
    labels_dir = dataset_root / 'labels'
    if not labels_dir.exists():
        return [" labels/ not found"]
    
    for split_dir in labels_dir.iterdir():
        if not split_dir.is_dir():
            continue
        
        label_files = list(split_dir.glob('*.txt'))
        checked = min(100, len(label_files))
        
        for label_file in label_files[:checked]:
            try:
                with open(label_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) != 5:
                            invalid_files.append(f"{label_file.name}:{line_num}")
                            break
                        
                        class_id = int(parts[0])
                        if not (0 <= class_id < len(BDD100K_CLASSES)):
                            invalid_files.append(f"{label_file.name}:{line_num}")
                            break
                        
                        for coord in parts[1:]:
                            if not (0 <= float(coord) <= 1):
                                invalid_files.append(f"{label_file.name}:{line_num}")
                                break
            except Exception as e:
                invalid_files.append(f"{label_file.name}")
        
        status = "✓" if not invalid_files else "❌"
        print(f"    {status} {split_dir.name}: Checked {checked}/{len(label_files)} files")
    
    if invalid_files:
        issues.append(f"{len(invalid_files)} invalid label files")
        print(f"    ❌ {len(invalid_files)} invalid files found")
    
    return issues


def validate_image_label_matching(dataset_root):
    """Validate image-label correspondence."""
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


def validate_json_metadata(dataset_root):
    """Validate JSON metadata files."""
    issues = []
    json_dir = dataset_root / 'representative_json'
    
    if not json_dir.exists():
        issues.append("representative_json/ not found")
        print(f"    ⚠️  No JSON metadata (optional)")
        return issues
    
    # Check metadata files
    for split in ['train', 'val', 'test']:
        json_path = json_dir / f'{split}_metadata.json'
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                required = ['split', 'statistics', 'class_distribution']
                missing = [f for f in required if f not in data]
                
                if missing:
                    issues.append(f"{split}: Missing fields {missing}")
                    print(f"    ❌ {split}: Missing {missing}")
                else:
                    total = data['statistics'].get('total_objects', 0)
                    print(f"    ✓ {split}_metadata.json ({total:,} objects)")
            except Exception as e:
                issues.append(f"{split}: JSON error - {e}")
    
    # Check performance analysis
    for split in ['train', 'val']:
        perf_path = json_dir / f'{split}_performance_analysis.json'
        if perf_path.exists():
            try:
                with open(perf_path, 'r') as f:
                    data = json.load(f)
                print(f"    ✓ {split}_performance_analysis.json")
            except:
                issues.append(f"{split}_performance_analysis.json invalid")
    
    return issues


def validate_attribute_distribution(dataset_root, dataset_name):
    """Validate attribute distribution diversity."""
    base_dir = Path(__file__).parent.parent
    tmp_labels = base_dir / 'bdd100k_tmp_labels' / '100k'
    
    if not tmp_labels.exists():
        print(f"    ⚠️  Cannot validate attributes (tmp labels missing)")
        return []
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        labels_dir = dataset_root / 'labels' / split
        if not labels_dir.exists():
            continue
        
        basenames = {f.stem for f in labels_dir.glob('*.txt')}
        if not basenames:
            continue
        
        # Count attributes
        weather_counts = Counter()
        scene_counts = Counter()
        timeofday_counts = Counter()
        
        tmp_split = tmp_labels / split
        for basename in basenames:
            json_file = tmp_split / f"{basename}.json"
            if not json_file.exists():
                continue
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                attrs = data.get('attributes', {})
                weather_counts[attrs.get('weather', 'undefined')] += 1
                scene_counts[attrs.get('scene', 'undefined')] += 1
                timeofday_counts[attrs.get('timeofday', 'undefined')] += 1
            except:
                continue
        
        if weather_counts:
            print(f"    {split} diversity:")
            print(f"      Weather: {len(weather_counts)} types - {dict(weather_counts)}")
            print(f"      Scene: {len(scene_counts)} types - {dict(scene_counts)}")
            print(f"      Time: {len(timeofday_counts)} types - {dict(timeofday_counts)}")
            
            # Check minimum diversity
            if len(weather_counts) < 3:
                issues.append(f"{split}: Low weather diversity ({len(weather_counts)} types)")
            if len(scene_counts) < 3:
                issues.append(f"{split}: Low scene diversity ({len(scene_counts)} types)")
            if len(timeofday_counts) < 2:
                issues.append(f"{split}: Low time diversity ({len(timeofday_counts)} types)")
    
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
    print(f"COMPREHENSIVE VALIDATION: {dataset['name']}")
    print(f"{'='*70}")
    
    all_issues = []
    
    # [1] Source comparison (limited datasets only)
    if source_root and source_root.exists():
        print(f"\n[1/6] SOURCE COMPARISON")
        print(f"Source: {source_root.name}")
        
        for split in ['train', 'val', 'test']:
            dataset_labels = dataset_root / 'labels' / split
            source_labels = source_root / 'labels' / split
            
            if dataset_labels.exists() and source_labels.exists():
                all_issues.extend(validate_source_subset(
                    dataset['name'], dataset_labels, source_labels, split
                ))
    else:
        print(f"\n[1/6] SOURCE COMPARISON - Skipped (full dataset)")
    
    # [2] Directory structure
    print(f"\n[2/6] DIRECTORY STRUCTURE")
    all_issues.extend(validate_directory_structure(dataset_root))
    
    # [3] data.yaml validation
    print(f"\n[3/6] DATA.YAML VALIDATION")
    all_issues.extend(validate_data_yaml(dataset_root))
    
    # [4] Label format validation
    print(f"\n[4/6] LABEL FORMAT VALIDATION")
    all_issues.extend(validate_label_format(dataset_root))
    
    # [5] Image-label matching
    print(f"\n[5/6] IMAGE-LABEL CORRESPONDENCE")
    all_issues.extend(validate_image_label_matching(dataset_root))
    
    # [6] JSON metadata validation
    print(f"\n[6/6] JSON METADATA VALIDATION")
    all_issues.extend(validate_json_metadata(dataset_root))
    
    # Attribute distribution (bonus for limited datasets)
    if source_root:
        print(f"\n[BONUS] ATTRIBUTE DISTRIBUTION DIVERSITY")
        all_issues.extend(validate_attribute_distribution(dataset_root, dataset['name']))
    
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
