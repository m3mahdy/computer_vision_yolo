"""
2. Convert Labels to YOLO Format and Create data.yaml.

Converts BDD100K JSON labels to YOLO .txt format for all splits (train/val/test)
and creates the data.yaml configuration file.

Input:  bdd100k_tmp_labels/100k/{train,val,test}/*.json
Output: bdd100k_yolo/labels/{train,val,test}/*.txt
        bdd100k_yolo/data.yaml

Usage:
    python dataset/2_convert_labels_to_yolo.py
"""

import json
from pathlib import Path
from tqdm import tqdm

from bdd100k_config import BDD100K_CLASSES, CLASS_TO_IDX, YOLO_DATASET_ROOT


# BDD100K images are standard 1280x720
BDD100K_IMAGE_WIDTH = 1280
BDD100K_IMAGE_HEIGHT = 720


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert BDD100K bbox format to YOLO format WITHOUT validation or filtering.
    Converts all boxes as-is from source data - does not hide data quality issues.
    
    BDD100K format: {x1, y1, x2, y2} (absolute pixel coordinates)
    YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    
    Args:
        bbox: Dict with keys 'x1', 'y1', 'x2', 'y2' (BDD100K format)
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        List [x_center, y_center, width, height] in normalized coordinates (0-1)
    """
    x1, y1 = bbox['x1'], bbox['y1']
    x2, y2 = bbox['x2'], bbox['y2']
    
    # Calculate YOLO format values - convert exactly as-is
    x_center = ((x1 + x2) / 2.0) / img_width
    y_center = ((y1 + y2) / 2.0) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return [x_center, y_center, width, height]


def convert_json_to_yolo_file(json_path, output_txt_path):
    """
    Convert a single BDD100K JSON label file to YOLO .txt format file.
    
    IMPORTANT: Only processes box2d (detection bounding boxes).
    Ignores segmentation data (poly2d, area, lane, drivable area).
    
    Args:
        json_path: Path to BDD100K JSON label file
        output_txt_path: Path where YOLO .txt file will be created
        
    Returns:
        Tuple: (converted_count, skipped_count)
    """
    try:
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        
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
            
            # Skip if category not in detection classes (excludes segmentation: area/*, lane/*)
            if category not in CLASS_TO_IDX:
                continue
            
            # Only process box2d (detection boxes) - skip objects without bounding boxes
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
            yolo_line = f"{class_idx} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
            yolo_labels.append(yolo_line)
            converted_count += 1
        
        # Write to output file
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt_path, 'w') as f:
            f.write('\n'.join(yolo_labels))
            if yolo_labels:
                f.write('\n')
        
        return converted_count, skipped_count
    
    except Exception as e:
        print(f"Warning: Error processing {json_path}: {e}")
        # Create empty file on error
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        output_txt_path.touch()
        return 0, 0


def convert_labels_directory(json_dir, output_labels_dir, split_name=''):
    """
    Convert all JSON label files in a directory to YOLO .txt format.
    Creates corresponding .txt files in the output directory.
    
    Args:
        json_dir: Path to directory containing JSON label files
        output_labels_dir: Path to directory where .txt files will be created
        split_name: Optional name for progress bar (e.g., 'train', 'val', 'test')
        
    Returns:
        Dict with statistics: total_files, total_converted, total_skipped
    """
    json_files = list(json_dir.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return {'total_files': 0, 'total_converted': 0, 'total_skipped': 0}
    
    desc = f"Converting {split_name}" if split_name else "Converting labels"
    
    total_converted = 0
    total_skipped = 0
    
    for json_file in tqdm(json_files, desc=desc, unit='files'):
        output_txt = output_labels_dir / f"{json_file.stem}.txt"
        converted, skipped = convert_json_to_yolo_file(json_file, output_txt)
        total_converted += converted
        total_skipped += skipped
    
    return {
        'total_files': len(json_files),
        'total_converted': total_converted,
        'total_skipped': total_skipped
    }


def create_data_yaml(dataset_root):
    """
    Create data.yaml configuration file for YOLO training.
    
    Args:
        dataset_root: Path to dataset root directory
    """
    yaml_lines = [
        f"path: {dataset_root.absolute()}",
        "",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(BDD100K_CLASSES)}",
        "",
        "names:"
    ]
    
    for class_name in BDD100K_CLASSES:
        yaml_lines.append(f"- {class_name}")
    
    yaml_content = "\n".join(yaml_lines)
    yaml_path = dataset_root / 'data.yaml'
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Created: {yaml_path}")


def main():
    """Main function to convert all labels and create data.yaml."""
    print("="*70)
    print("CONVERT LABELS TO YOLO FORMAT & CREATE DATA.YAML")
    print("="*70)
    
    base_dir = Path(__file__).parent.parent
    tmp_labels_dir = base_dir / 'bdd100k_tmp_labels' / '100k'
    output_root = YOLO_DATASET_ROOT
    
    # Check if tmp labels exist
    if not tmp_labels_dir.exists():
        print(f"\n❌ Labels not found: {tmp_labels_dir}")
        print("Run 1_download_extract_main_dataset.py first")
        return
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Convert each split
    print()
    total_stats = {'total_files': 0, 'total_converted': 0, 'total_skipped': 0}
    
    for split in ['train', 'val', 'test']:
        json_dir = tmp_labels_dir / split
        output_dir = output_root / 'labels' / split
        
        if not json_dir.exists():
            print(f"⚠️  Skipping {split}: {json_dir} not found")
            continue
        
        stats = convert_labels_directory(json_dir, output_dir, split)
        
        total_stats['total_files'] += stats['total_files']
        total_stats['total_converted'] += stats['total_converted']
        total_stats['total_skipped'] += stats['total_skipped']
        
        print(f"  {split}: {stats['total_files']:,} files, "
              f"{stats['total_converted']:,} objects converted, "
              f"{stats['total_skipped']:,} skipped")
    
    # Create data.yaml
    create_data_yaml(output_root)
    
    # Copy images from tmp to final location
    print(f"\n{'='*70}")
    print("COPYING IMAGES")
    print(f"{'='*70}")
    
    tmp_images_dir = base_dir / 'bdd100k_tmp_images' / '100k'
    if not tmp_images_dir.exists():
        print(f"\n⚠️  Images not found: {tmp_images_dir}")
        print("Skipping image copy")
    else:
        import subprocess
        for split in ['train', 'val', 'test']:
            src = tmp_images_dir / split
            dest = output_root / 'images' / split
            
            if src.exists():
                dest.mkdir(parents=True, exist_ok=True)
                print(f"\nCopying {split}...")
                result = subprocess.run(
                    ['cp', '-r', f"{src}/", str(dest)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    image_count = len(list(dest.glob('*')))
                    print(f"  ✓ {split}: {image_count:,} images copied")
                else:
                    print(f"  ❌ {split}: Copy failed - {result.stderr}")
    
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE")
    print("="*70)
    print(f"\nTotal files: {total_stats['total_files']:,}")
    print(f"Total objects: {total_stats['total_converted']:,}")
    print(f"Total skipped: {total_stats['total_skipped']:,}")
    print(f"\nOutput: {output_root}")
    print("\nNext step: Run 2.5_validate_conversion.py to validate")
    print("          or run full processing script")


if __name__ == '__main__':
    main()
