"""
BDD100K Dataset Validation Module.

This module handles validation operations including counting objects,
comparing statistics, and performing integrity checks.
"""

import json
from pathlib import Path
from tqdm import tqdm

from bdd100k_config import BDD100K_CLASSES


def count_objects_in_single_label(label_file_path):
    """
    Count objects by class from a single YOLO format label file.
    
    Args:
        label_file_path: Path to YOLO format .txt label file
        
    Returns:
        Dict mapping class_name to count: {class_name: count}
    """
    class_counts = {cls: 0 for cls in BDD100K_CLASSES}
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(BDD100K_CLASSES):
                            class_counts[BDD100K_CLASSES[class_id]] += 1
                    except (ValueError, IndexError):
                        continue
    except FileNotFoundError:
        pass
    return class_counts


def count_objects_in_labels(labels_dir, desc="Counting objects"):
    """
    Count objects by class from YOLO format label files in a directory.
    
    Args:
        labels_dir: Path to directory containing .txt label files
        desc: Description for progress bar
        
    Returns:
        Dict mapping class_name to count: {class_name: count}
    """
    object_counts = {cls: 0 for cls in BDD100K_CLASSES}
    txt_files = list(labels_dir.glob('*.txt'))
    
    for txt_file in tqdm(txt_files, desc=desc, unit='files', leave=False):
        file_counts = count_objects_in_single_label(txt_file)
        for cls in BDD100K_CLASSES:
            object_counts[cls] += file_counts[cls]
    
    return object_counts


def compare_dataset_statistics(tmp_labels_dir, yolo_labels_dir, split_name):
    """
    Compare object counts between original JSON files and generated YOLO labels.
    
    Args:
        tmp_labels_dir: Path to directory containing original JSON labels
        yolo_labels_dir: Path to directory containing generated YOLO labels
        split_name: Name of the split (train/val/test)
        
    Returns:
        Dict with comparison statistics including 'all_match', 'match_percentage', 'by_class'
        Returns None if JSON directory not found
    """
    from bdd100k_convert import count_objects_in_json_files
    
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


def perform_integrity_check(images_dir, labels_dir, split_name):
    """
    Verify all images have corresponding label files and vice versa.
    
    Args:
        images_dir: Path to directory containing image files
        labels_dir: Path to directory containing label files
        split_name: Name of the split (for display purposes)
        
    Returns:
        Tuple: (images_without_labels, labels_without_images, is_valid)
        - images_without_labels: Set of basenames for images missing labels
        - labels_without_images: Set of basenames for labels missing images
        - is_valid: Boolean indicating if dataset is valid (True if both sets are empty)
    """
    image_basenames = {f.stem for f in images_dir.glob('*.jpg')} | {f.stem for f in images_dir.glob('*.png')}
    label_basenames = {f.stem for f in labels_dir.glob('*.txt')}
    
    images_without_labels = image_basenames - label_basenames
    labels_without_images = label_basenames - image_basenames
    
    is_valid = len(images_without_labels) == 0 and len(labels_without_images) == 0
    
    return images_without_labels, labels_without_images, is_valid


def verify_dataset_integrity(dataset_root, splits):
    """
    Verify integrity of all specified splits in the dataset.
    
    Args:
        dataset_root: Path to root of YOLO dataset
        splits: List of split names to verify (e.g., ['train', 'val', 'test'])
        
    Returns:
        Dict mapping split_name to integrity check results
        Each result contains: 'is_valid', 'num_images', 'num_labels', 
                             'images_without_labels', 'labels_without_images'
    """
    print(f"\n{'='*70}")
    print(f"DATASET INTEGRITY CHECK")
    print(f"{'='*70}")
    
    results = {}
    all_valid = True
    
    for split in splits:
        images_dir = dataset_root / 'images' / split
        labels_dir = dataset_root / 'labels' / split
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"\n{split}: ⚠️  Missing (skipped)")
            continue
        
        images_without_labels, labels_without_images, is_valid = perform_integrity_check(
            images_dir, labels_dir, split
        )
        
        num_images = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
        num_labels = len(list(labels_dir.glob('*.txt')))
        
        results[split] = {
            'is_valid': is_valid,
            'num_images': num_images,
            'num_labels': num_labels,
            'images_without_labels': len(images_without_labels),
            'labels_without_images': len(labels_without_images)
        }
        
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"\n{split}: {status}")
        print(f"  Images: {num_images:,}")
        print(f"  Labels: {num_labels:,}")
        
        if not is_valid:
            all_valid = False
            if images_without_labels:
                print(f"  ⚠️  Images without labels: {len(images_without_labels)}")
            if labels_without_images:
                print(f"  ⚠️  Labels without images: {len(labels_without_images)}")
    
    print(f"\n{'='*70}")
    if all_valid:
        print("✅ ALL SPLITS VALID: Perfect image-label matching")
    else:
        print("❌ INTEGRITY ISSUES FOUND: Some mismatches detected")
    print(f"{'='*70}")
    
    return results
