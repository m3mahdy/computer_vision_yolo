"""
3. Create Metadata for YOLO Dataset.

Creates representative_json metadata files for bdd100k_yolo dataset.
This metadata enables fast representative sampling in script 4.

Reads from: bdd100k_tmp_labels (original BDD100K JSON files)
Writes to: bdd100k_yolo/representative_json/

Output files:
- representative_json/{split}_metadata.json (train/val/test)

Usage:
    python dataset/3_create_metadata_for_yolo.py
"""

import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from bdd100k_config import YOLO_DATASET_ROOT, BDD100K_CLASSES, CLASS_TO_IDX


def get_label_attributes(json_path):
    """Extract attributes from BDD100K JSON label file (READ-ONLY)."""
    try:
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        
        attributes = label_data.get('attributes', {})
        frames = label_data.get('frames', [])
        objects = frames[0].get('objects', []) if frames else label_data.get('objects', label_data.get('labels', []))
        
        # Extract categories from box2d objects only (detection, not segmentation)
        categories = []
        for obj in objects:
            if 'box2d' in obj:
                cat = obj.get('category', '')
                if cat in CLASS_TO_IDX:
                    categories.append(cat)
        
        # Count objects per class
        from collections import Counter
        class_counts = Counter(categories)
        
        return {
            'weather': attributes.get('weather', 'undefined'),
            'scene': attributes.get('scene', 'undefined'),
            'timeofday': attributes.get('timeofday', 'undefined'),
            'categories': list(set(categories)),  # Unique classes present
            'class_counts': dict(class_counts),  # Count per class
            'object_count': len(categories)
        }
    except Exception as e:
        return None
        return None


def create_metadata_for_split(split_name, tmp_labels_dir, yolo_dataset_root):
    """
    Create metadata JSON file for a specific split (READ-ONLY on source).
    
    Args:
        split_name: 'train', 'val', or 'test'
        tmp_labels_dir: Path to bdd100k_tmp_labels/100k/{split}
        yolo_dataset_root: Path to bdd100k_yolo
    """
    print(f"\n{'='*70}")
    print(f"CREATING METADATA: {split_name}")
    print(f"{'='*70}")
    
    # Verify source JSON files exist (READ-ONLY check)
    json_dir = tmp_labels_dir / split_name
    if not json_dir.exists():
        print(f"❌ Source JSON not found: {json_dir}")
        return False
    
    # Get list of JSON files (READ-ONLY)
    json_files = list(json_dir.glob('*.json'))
    if not json_files:
        print(f"❌ No JSON files found in: {json_dir}")
        return False
    
    print(f"Source: {json_dir}")
    print(f"Files: {len(json_files):,}")
    
    # Create metadata structure
    metadata = {
        'dataset': 'bdd100k_yolo',
        'split': split_name,
        'created': datetime.now().isoformat(),
        'total_files': len(json_files),
        'files': {}
    }
    
    # Process each JSON file (READ-ONLY)
    files_with_objects = 0
    total_objects = 0
    
    for json_file in tqdm(json_files, desc=f"  Processing {split_name}", unit='files'):
        basename = json_file.stem
        attrs = get_label_attributes(json_file)
        
        if attrs:
            metadata['files'][basename] = {
                'weather': attrs['weather'],
                'scene': attrs['scene'],
                'timeofday': attrs['timeofday'],
                'categories': attrs['categories'],
                'class_counts': attrs['class_counts'],
                'object_count': attrs['object_count']
            }
            
            if attrs['object_count'] > 0:
                files_with_objects += 1
                total_objects += attrs['object_count']
    
    # Add summary statistics
    metadata['summary'] = {
        'files_with_objects': files_with_objects,
        'total_objects': total_objects,
        'avg_objects_per_file': total_objects / files_with_objects if files_with_objects > 0 else 0
    }
    
    # Create output directory
    output_dir = yolo_dataset_root / 'representative_json'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write metadata JSON
    output_file = output_dir / f'{split_name}_metadata.json'
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Created: {output_file}")
    print(f"  Files processed: {len(json_files):,}")
    print(f"  Files with objects: {files_with_objects:,}")
    print(f"  Total objects: {total_objects:,}")
    print(f"  Avg objects/file: {metadata['summary']['avg_objects_per_file']:.2f}")
    
    return True


def main():
    """Main function."""
    base_dir = Path(__file__).parent.parent
    yolo_dataset_root = YOLO_DATASET_ROOT
    tmp_labels_dir = base_dir / 'bdd100k_tmp_labels' / '100k'
    
    print("="*70)
    print("CREATE METADATA FOR YOLO DATASET")
    print("="*70)
    print(f"Target: {yolo_dataset_root}")
    print(f"Source: {tmp_labels_dir}")
    
    # Validate YOLO dataset exists (READ-ONLY check)
    if not yolo_dataset_root.exists():
        print(f"\n❌ YOLO dataset not found: {yolo_dataset_root}")
        print("   Run script 2 first to create bdd100k_yolo dataset")
        return
    
    if not (yolo_dataset_root / 'data.yaml').exists():
        print(f"\n❌ data.yaml not found: {yolo_dataset_root / 'data.yaml'}")
        return
    
    # Validate tmp_labels exists (READ-ONLY check)
    if not tmp_labels_dir.exists():
        print(f"\n❌ Source labels not found: {tmp_labels_dir}")
        print("   Run script 1 first to download and extract BDD100K dataset")
        return
    
    print("✓ Validation passed")
    
    # Process each split
    splits_processed = []
    for split in ['train', 'val', 'test']:
        if create_metadata_for_split(split, tmp_labels_dir, yolo_dataset_root):
            splits_processed.append(split)
    
    # Final summary
    print(f"\n{'='*70}")
    if len(splits_processed) == 3:
        print("✅ ALL METADATA CREATED SUCCESSFULLY")
    else:
        print(f"⚠️  PARTIAL SUCCESS: {len(splits_processed)}/3 splits")
    print("="*70)
    print(f"Location: {yolo_dataset_root / 'representative_json'}")
    print("\nNext step: Run script 4 to create limited datasets")
    print("="*70)


if __name__ == '__main__':
    main()
