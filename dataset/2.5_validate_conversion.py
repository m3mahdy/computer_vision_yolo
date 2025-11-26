"""
2.5. Validate Conversion Output.

READ-ONLY comprehensive validation of script 2 outputs.
Does NOT modify any data - only reads and validates.

Validations performed:
1. Object counts match between JSON and YOLO labels (per class, per split)
2. File counts match between source and converted
3. Images directory exists and has correct count
4. Image-label correspondence (1:1 matching)
5. data.yaml structure and content validation (comprehensive)
6. Label file format validation (coordinates in 0-1 range)
7. Objects per image statistics
8. Empty files detection
9. Class distribution per split
10. Cross-split summary statistics

Usage:
    python dataset/2.5_validate_conversion.py
"""

import json
from pathlib import Path
from collections import Counter
import yaml

from bdd100k_config import BDD100K_CLASSES, CLASS_TO_IDX, YOLO_DATASET_ROOT


def count_objects_in_json(json_dir):
    """Count objects by class from BDD100K JSON files (READ-ONLY)."""
    if not json_dir.exists():
        return None
    
    class_counts = Counter()
    total_files = 0
    files_with_objects = 0
    empty_files = []
    
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
            else:
                empty_files.append(json_file.name)
        except:
            continue
    
    return {
        'class_counts': class_counts,
        'total_files': total_files,
        'files_with_objects': files_with_objects,
        'empty_files': empty_files
    }


def count_objects_in_yolo(labels_dir):
    """Count objects by class from YOLO label files (READ-ONLY)."""
    if not labels_dir.exists():
        return None
    
    class_counts = Counter()
    total_files = 0
    files_with_objects = 0
    empty_files = []
    objects_per_image = []
    invalid_coords = []
    
    for label_file in labels_dir.glob('*.txt'):
        total_files += 1
        file_object_count = 0
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                
                if len(parts) != 5:
                    invalid_coords.append(f"{label_file.name}:L{line_num}")
                    continue
                
                class_id = int(parts[0])
                if 0 <= class_id < len(BDD100K_CLASSES):
                    class_name = BDD100K_CLASSES[class_id]
                    class_counts[class_name] += 1
                    file_object_count += 1
                    
                    # Validate coordinates are in 0-1 range
                    for coord_idx, coord in enumerate(parts[1:], 1):
                        coord_val = float(coord)
                        if not (0 <= coord_val <= 1):
                            invalid_coords.append(f"{label_file.name}:L{line_num}:C{coord_idx}")
            
            if file_object_count > 0:
                files_with_objects += 1
                objects_per_image.append(file_object_count)
            else:
                empty_files.append(label_file.name)
        except:
            continue
    
    return {
        'class_counts': class_counts,
        'total_files': total_files,
        'files_with_objects': files_with_objects,
        'empty_files': empty_files,
        'objects_per_image': objects_per_image,
        'invalid_coords': invalid_coords
    }


def validate_split_object_counts(split_name, json_dir, yolo_labels_dir):
    """Validate object counts match between JSON and YOLO (READ-ONLY)."""
    print(f"\n{split_name.upper()}")
    print("-" * 50)
    
    issues = []
    
    # Count in JSON (READ-ONLY)
    json_stats = count_objects_in_json(json_dir)
    if not json_stats:
        issues.append(f"{split_name}: Source JSON not found")
        print(f"  ⚠️  Source JSON not found: {json_dir}")
        return issues
    
    # Count in YOLO (READ-ONLY)
    yolo_stats = count_objects_in_yolo(yolo_labels_dir)
    if not yolo_stats:
        issues.append(f"{split_name}: YOLO labels not found")
        print(f"  ❌ YOLO labels not found: {yolo_labels_dir}")
        return issues
    
    # Compare file counts
    print(f"\nFile Counts:")
    print(f"  JSON files: {json_stats['total_files']:,}")
    print(f"  YOLO files: {yolo_stats['total_files']:,}")
    print(f"  JSON with objects: {json_stats['files_with_objects']:,}")
    print(f"  YOLO with objects: {yolo_stats['files_with_objects']:,}")
    
    if json_stats['total_files'] != yolo_stats['total_files']:
        issues.append(f"{split_name}: File count mismatch")
        print(f"  ❌ Mismatch!")
    else:
        print(f"  ✓ Match")
    
    # Empty files
    if yolo_stats['empty_files']:
        print(f"  ⚠️  Empty YOLO files: {len(yolo_stats['empty_files'])}")
        if len(yolo_stats['empty_files']) <= 5:
            for empty in yolo_stats['empty_files'][:5]:
                print(f"    - {empty}")
    
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
    
    # Objects per image statistics
    if yolo_stats['objects_per_image']:
        obj_per_img = yolo_stats['objects_per_image']
        print(f"\nObjects per Image Statistics:")
        print(f"  Min: {min(obj_per_img)}")
        print(f"  Max: {max(obj_per_img)}")
        print(f"  Average: {sum(obj_per_img) / len(obj_per_img):.2f}")
        print(f"  Images with 1 object: {obj_per_img.count(1):,}")
        print(f"  Images with 2-5 objects: {sum(1 for x in obj_per_img if 2 <= x <= 5):,}")
        print(f"  Images with 6-10 objects: {sum(1 for x in obj_per_img if 6 <= x <= 10):,}")
        print(f"  Images with >10 objects: {sum(1 for x in obj_per_img if x > 10):,}")
    
    # Invalid coordinates
    if yolo_stats['invalid_coords']:
        issues.append(f"{split_name}: {len(yolo_stats['invalid_coords'])} invalid coordinates")
        print(f"\n  ❌ Invalid coordinates: {len(yolo_stats['invalid_coords'])}")
        for invalid in yolo_stats['invalid_coords'][:5]:
            print(f"    - {invalid}")
    
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
    """Validate images directory exists and has files (READ-ONLY)."""
    print(f"\n{split_name.upper()} - Images")
    print("-" * 50)
    
    issues = []
    
    if not images_dir.exists():
        issues.append(f"{split_name}: Images directory not found")
        print(f"  ❌ Not found: {images_dir}")
        return issues
    
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    print(f"  Image files: {len(image_files):,}")
    
    # Check file extensions distribution
    jpg_count = len(list(images_dir.glob('*.jpg')))
    png_count = len(list(images_dir.glob('*.png')))
    print(f"    JPG: {jpg_count:,}")
    print(f"    PNG: {png_count:,}")
    
    if len(image_files) == 0:
        issues.append(f"{split_name}: No images found")
        print(f"  ❌ No images")
    else:
        print(f"  ✓ Images present")
    
    return issues


def validate_image_label_correspondence(split_name, images_dir, labels_dir):
    """Validate each image has corresponding label and vice versa (READ-ONLY)."""
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
    print(f"  Matched pairs: {len(image_basenames & label_basenames):,}")
    
    if missing_labels:
        issues.append(f"{split_name}: {len(missing_labels)} images without labels")
        print(f"  ❌ {len(missing_labels)} images missing labels")
        if len(missing_labels) <= 5:
            for basename in list(missing_labels)[:5]:
                print(f"    - {basename}")
    
    if missing_images:
        issues.append(f"{split_name}: {len(missing_images)} labels without images")
        print(f"  ❌ {len(missing_images)} labels missing images")
        if len(missing_images) <= 5:
            for basename in list(missing_images)[:5]:
                print(f"    - {basename}")
    
    if not missing_labels and not missing_images:
        print(f"  ✓ Perfect correspondence")
    
    return issues


def validate_data_yaml(yaml_path, dataset_root):
    """Validate data.yaml file comprehensively (READ-ONLY)."""
    print(f"\nDATA.YAML Comprehensive Validation")
    print("-" * 50)
    
    issues = []
    
    if not yaml_path.exists():
        issues.append("data.yaml not found")
        print(f"  ❌ Not found: {yaml_path}")
        return issues
    
    try:
        # Check file is readable
        with open(yaml_path, 'r') as f:
            content = f.read()
            if not content.strip():
                issues.append("data.yaml: File is empty")
                print(f"  ❌ File is empty")
                return issues
        
        # Parse YAML
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            issues.append("data.yaml: Failed to parse YAML")
            print(f"  ❌ Failed to parse")
            return issues
        
        print(f"  ✓ File exists and is readable")
        print(f"  ✓ Valid YAML format")
        
        # 1. Validate 'path' field
        print(f"\n  [1] 'path' field:")
        if 'path' not in data:
            issues.append("data.yaml: Missing 'path' field")
            print(f"    ❌ Missing")
        else:
            path_value = data['path']
            print(f"    Value: {path_value}")
            
            if not isinstance(path_value, (str, Path)):
                issues.append("data.yaml: 'path' must be a string")
                print(f"    ❌ Not a string: {type(path_value)}")
            elif not str(path_value).strip():
                issues.append("data.yaml: 'path' is empty")
                print(f"    ❌ Empty")
            else:
                print(f"    ✓ Valid")
        
        # 2. Validate 'nc' (number of classes)
        print(f"\n  [2] 'nc' (number of classes):")
        if 'nc' not in data:
            issues.append("data.yaml: Missing 'nc' field")
            print(f"    ❌ Missing")
        else:
            nc_value = data['nc']
            print(f"    Value: {nc_value}")
            
            if not isinstance(nc_value, int):
                issues.append(f"data.yaml: 'nc' must be integer, got {type(nc_value)}")
                print(f"    ❌ Not integer")
            elif nc_value != len(BDD100K_CLASSES):
                issues.append(f"data.yaml: nc should be {len(BDD100K_CLASSES)}, got {nc_value}")
                print(f"    ❌ Expected {len(BDD100K_CLASSES)}")
            else:
                print(f"    ✓ Correct ({nc_value})")
        
        # 3. Validate 'names' (class names list)
        print(f"\n  [3] 'names' (class names):")
        if 'names' not in data:
            issues.append("data.yaml: Missing 'names' field")
            print(f"    ❌ Missing")
        else:
            names_value = data['names']
            
            if not isinstance(names_value, list):
                issues.append(f"data.yaml: 'names' must be list, got {type(names_value)}")
                print(f"    ❌ Not a list")
            else:
                print(f"    Length: {len(names_value)}")
                
                # Check length matches nc
                if 'nc' in data and len(names_value) != data['nc']:
                    issues.append(f"data.yaml: 'names' length ({len(names_value)}) != nc ({data['nc']})")
                    print(f"    ❌ Length mismatch with nc")
                
                # Check length matches BDD100K_CLASSES
                if len(names_value) != len(BDD100K_CLASSES):
                    issues.append(f"data.yaml: 'names' length should be {len(BDD100K_CLASSES)}")
                    print(f"    ❌ Expected {len(BDD100K_CLASSES)} classes")
                else:
                    print(f"    ✓ Correct length")
                
                # Validate each class name
                class_mismatches = []
                for idx, cls in enumerate(names_value):
                    if idx >= len(BDD100K_CLASSES):
                        break
                    
                    if not isinstance(cls, str):
                        class_mismatches.append(f"Index {idx}: not a string ({type(cls)})")
                    elif cls != BDD100K_CLASSES[idx]:
                        class_mismatches.append(f"Index {idx}: '{cls}' vs '{BDD100K_CLASSES[idx]}'")
                
                if class_mismatches:
                    issues.append(f"data.yaml: {len(class_mismatches)} class name mismatches")
                    print(f"    ❌ Mismatches found:")
                    for mismatch in class_mismatches[:5]:
                        print(f"      - {mismatch}")
                    if len(class_mismatches) > 5:
                        print(f"      ... and {len(class_mismatches) - 5} more")
                else:
                    print(f"    ✓ All names match BDD100K_CLASSES")
                
                # Check for duplicates
                if len(names_value) != len(set(names_value)):
                    duplicates = [name for name in names_value if names_value.count(name) > 1]
                    issues.append(f"data.yaml: Duplicate class names: {set(duplicates)}")
                    print(f"    ❌ Duplicates: {set(duplicates)}")
                else:
                    print(f"    ✓ No duplicates")
        
        # 4. Validate split definitions
        print(f"\n  [4] Split definitions:")
        splits_found = []
        expected_splits = ['train', 'val', 'test']
        
        for split in expected_splits:
            if split in data:
                splits_found.append(split)
                split_path = data[split]
                
                if not isinstance(split_path, str):
                    issues.append(f"data.yaml: '{split}' path must be string")
                    print(f"    ❌ {split}: not a string ({type(split_path)})")
                else:
                    print(f"    ✓ {split}: {split_path}")
                    
                    # Verify images directory exists
                    images_split_path = dataset_root / split_path
                    if not images_split_path.exists():
                        issues.append(f"data.yaml: '{split}' images path doesn't exist")
                        print(f"      ⚠️  Images not found: {images_split_path}")
                    else:
                        print(f"      ✓ Images exist: {images_split_path}")
                    
                    # Verify labels directory exists
                    labels_split_path = dataset_root / 'labels' / split
                    if not labels_split_path.exists():
                        issues.append(f"data.yaml: '{split}' labels path doesn't exist")
                        print(f"      ⚠️  Labels not found: {labels_split_path}")
                    else:
                        print(f"      ✓ Labels exist: {labels_split_path}")
            else:
                print(f"    ⚠️  {split}: not defined")
        
        if len(splits_found) != 3:
            issues.append(f"data.yaml: Only {len(splits_found)}/3 splits defined")
            print(f"    ⚠️  Only {len(splits_found)}/3 splits")
        else:
            print(f"    ✓ All 3 splits defined")
        
        # 5. Check for unexpected fields
        print(f"\n  [5] Extra fields check:")
        expected_fields = {'path', 'train', 'val', 'test', 'nc', 'names'}
        unexpected_fields = set(data.keys()) - expected_fields
        if unexpected_fields:
            print(f"    ⚠️  Unexpected: {unexpected_fields}")
        else:
            print(f"    ✓ No unexpected fields")
        
        # 6. File size check
        print(f"\n  [6] File integrity:")
        file_size = yaml_path.stat().st_size
        print(f"    Size: {file_size} bytes")
        if file_size > 10000:
            issues.append(f"data.yaml: File unusually large ({file_size} bytes)")
            print(f"    ⚠️  Seems large for data.yaml")
        elif file_size < 100:
            issues.append(f"data.yaml: File unusually small ({file_size} bytes)")
            print(f"    ⚠️  Seems too small")
        else:
            print(f"    ✓ Normal size")
        
    except yaml.YAMLError as e:
        issues.append(f"data.yaml: YAML parse error - {e}")
        print(f"  ❌ YAML parse error: {e}")
    except Exception as e:
        issues.append(f"data.yaml: Validation error - {e}")
        print(f"  ❌ Validation error: {e}")
    
    return issues


def validate_split_summary(dataset_root):
    """Generate cross-split summary statistics (READ-ONLY)."""
    print(f"\n{'='*70}")
    print("CROSS-SPLIT SUMMARY")
    print(f"{'='*70}")
    
    total_images = 0
    total_labels = 0
    total_objects = 0
    split_stats = {}
    
    for split in ['train', 'val', 'test']:
        images_dir = dataset_root / 'images' / split
        labels_dir = dataset_root / 'labels' / split
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
        
        img_count = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
        lbl_count = len(list(labels_dir.glob('*.txt')))
        
        # Count objects
        obj_count = 0
        for label_file in labels_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    obj_count += sum(1 for line in f if line.strip())
            except:
                continue
        
        split_stats[split] = {
            'images': img_count,
            'labels': lbl_count,
            'objects': obj_count
        }
        
        total_images += img_count
        total_labels += lbl_count
        total_objects += obj_count
    
    # Print summary
    print(f"\n{'Split':<10} {'Images':>10} {'Labels':>10} {'Objects':>10} {'Obj/Img':>10}")
    print("-" * 70)
    for split, stats in split_stats.items():
        avg_obj = stats['objects'] / stats['images'] if stats['images'] > 0 else 0
        print(f"{split:<10} {stats['images']:>10,} {stats['labels']:>10,} {stats['objects']:>10,} {avg_obj:>10.2f}")
    
    print("-" * 70)
    avg_obj_total = total_objects / total_images if total_images > 0 else 0
    print(f"{'TOTAL':<10} {total_images:>10,} {total_labels:>10,} {total_objects:>10,} {avg_obj_total:>10.2f}")
    
    return split_stats


def main():
    """Main validation function (READ-ONLY)."""
    base_dir = Path(__file__).parent.parent
    dataset_root = YOLO_DATASET_ROOT
    tmp_labels_dir = base_dir / 'bdd100k_tmp_labels' / '100k'
    
    print("="*70)
    print("VALIDATE SCRIPT 2 CONVERSION OUTPUT (READ-ONLY)")
    print("="*70)
    print(f"Dataset: {dataset_root}")
    print(f"Source: {tmp_labels_dir}")
    print("\nNOTE: This script only validates - no data will be modified")
    
    if not dataset_root.exists():
        print(f"\n❌ Dataset not found: {dataset_root}")
        print("Run script 2 first")
        return
    
    all_issues = []
    
    # Validate data.yaml
    all_issues.extend(validate_data_yaml(dataset_root / 'data.yaml', dataset_root))
    
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
    
    # Cross-split summary
    validate_split_summary(dataset_root)
    
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
