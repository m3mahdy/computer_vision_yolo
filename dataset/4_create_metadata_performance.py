"""
4. Create Metadata and Performance Analysis.

Creates comprehensive metadata JSON files with attribute distribution analysis.
Analyzes object counts, file statistics, and BDD100K attributes (weather, scene, time).

Output files:
- representative_json/{split}_metadata.json
- representative_json/{split}_performance_analysis.json (train/val only)

Usage:
    python dataset/4_create_metadata_performance.py
"""

import json
from pathlib import Path
from collections import Counter
from datetime import datetime
from tqdm import tqdm

from bdd100k_config import BDD100K_CLASSES, CLASS_TO_IDX, REPRESENTATIVE_ATTRIBUTES


def get_label_attributes(json_path):
    """Extract attributes from BDD100K JSON label file."""
    try:
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        
        attributes = label_data['attributes']
        frames = label_data.get('frames', [])
        objects = frames[0].get('objects', []) if frames else label_data.get('objects', label_data.get('labels', []))
        categories = [obj.get('category', '') for obj in objects if 'box2d' in obj]
        
        return {
            'weather': attributes['weather'],
            'scene': attributes['scene'],
            'timeofday': attributes['timeofday'],
            'categories': [cat for cat in categories if cat in CLASS_TO_IDX],
            'num_objects': len(categories)
        }
    except:
        return None


def count_attribute_distribution(json_dir, filter_basenames=None):
    """Count attribute distribution across JSON files."""
    weather_counts = Counter()
    scene_counts = Counter()
    timeofday_counts = Counter()
    total_images = 0
    images_with_labels = 0
    
    json_files = list(json_dir.glob('*.json'))
    if filter_basenames:
        json_files = [f for f in json_files if f.stem in filter_basenames]
    
    for json_file in json_files:
        total_images += 1
        attrs = get_label_attributes(json_file)
        
        if attrs and attrs['categories']:
            images_with_labels += 1
            weather_counts[attrs['weather']] += 1
            scene_counts[attrs['scene']] += 1
            timeofday_counts[attrs['timeofday']] += 1
    
    return {
        'weather': dict(weather_counts),
        'scene': dict(scene_counts),
        'timeofday': dict(timeofday_counts),
        'total_images': total_images,
        'images_with_labels': images_with_labels
    }


def create_metadata_json(dataset_root, split_name, base_dir):
    """Create comprehensive metadata JSON with attribute analysis."""
    labels_dir = dataset_root / 'labels' / split_name
    images_dir = dataset_root / 'images' / split_name
    
    if not labels_dir.exists():
        print(f"  ⚠️  No labels found: {labels_dir}")
        return None
    
    label_files = list(labels_dir.glob('*.txt'))
    basenames = {f.stem for f in label_files}
    
    # Count objects per class
    class_counts = Counter()
    total_objects = 0
    images_with_objects = 0
    
    for label_file in tqdm(label_files, desc=f"  Analyzing {split_name}", leave=False):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if lines:
            images_with_objects += 1
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(BDD100K_CLASSES):
                        class_counts[BDD100K_CLASSES[class_id]] += 1
                        total_objects += 1
    
    # Attribute distribution
    tmp_labels_dir = base_dir / 'bdd100k_tmp_labels' / '100k' / split_name
    attr_dist = {}
    if tmp_labels_dir.exists():
        attr_dist = count_attribute_distribution(tmp_labels_dir, basenames)
    
    # Build metadata
    metadata = {
        'dataset': dataset_root.name,
        'split': split_name,
        'generated_at': datetime.now().isoformat(),
        'statistics': {
            'total_images': len(label_files),
            'images_with_objects': images_with_objects,
            'total_objects': total_objects,
            'avg_objects_per_image': round(total_objects / len(label_files), 2) if label_files else 0
        },
        'class_distribution': dict(class_counts),
        'attribute_distribution': attr_dist if attr_dist else None
    }
    
    return metadata


def create_performance_analysis_json(dataset_root, split_name):
    """Create performance analysis baseline JSON."""
    labels_dir = dataset_root / 'labels' / split_name
    
    if not labels_dir.exists():
        return None
    
    label_files = list(labels_dir.glob('*.txt'))
    
    # Count objects per class
    class_counts = Counter()
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(BDD100K_CLASSES):
                        class_counts[BDD100K_CLASSES[class_id]] += 1
    
    # Create baseline structure
    analysis = {
        'dataset': dataset_root.name,
        'split': split_name,
        'generated_at': datetime.now().isoformat(),
        'baseline': {
            'class_distribution': dict(class_counts),
            'total_objects': sum(class_counts.values()),
            'num_classes_present': len(class_counts)
        },
        'model_results': {},
        'notes': 'Baseline created. Run validation to populate model results.'
    }
    
    return analysis


def select_dataset():
    """Display menu and select dataset."""
    base_dir = Path(__file__).parent.parent
    
    # Find all datasets with data.yaml
    datasets = []
    for item in base_dir.iterdir():
        if item.is_dir() and (item / 'data.yaml').exists():
            datasets.append(item)
    
    if not datasets:
        print("\n❌ No datasets found with data.yaml")
        return None
    
    datasets.sort(key=lambda x: x.name)
    
    print("\n" + "="*70)
    print("SELECT DATASET FOR METADATA GENERATION")
    print("="*70)
    
    for idx, dataset in enumerate(datasets, 1):
        print(f"[{idx}] {dataset.name}")
    
    print("[0] Cancel")
    print("="*70)
    
    while True:
        choice = input(f"\nSelect (0-{len(datasets)}): ").strip()
        if choice == '0':
            return None
        try:
            choice_int = int(choice)
            if 1 <= choice_int <= len(datasets):
                return datasets[choice_int - 1]
        except ValueError:
            pass
        print("Invalid choice")


def main():
    """Main function."""
    base_dir = Path(__file__).parent.parent
    
    dataset_root = select_dataset()
    if not dataset_root:
        print("\nCancelled.")
        return
    
    print(f"\n{'='*70}")
    print(f"GENERATING METADATA: {dataset_root.name}")
    print(f"{'='*70}")
    
    # Create output directory
    output_dir = dataset_root / 'representative_json'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find available splits
    labels_root = dataset_root / 'labels'
    splits = [s.name for s in labels_root.iterdir() if s.is_dir() and list(s.glob('*.txt'))]
    
    if not splits:
        print(f"\n❌ No splits found in: {labels_root}")
        return
    
    print(f"\nFound splits: {', '.join(splits)}")
    
    # Generate metadata for each split
    for split in splits:
        print(f"\n{split.upper()}:")
        
        # Create metadata
        metadata = create_metadata_json(dataset_root, split, base_dir)
        if metadata:
            metadata_path = output_dir / f'{split}_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  ✓ {metadata_path.name}")
        
        # Create performance analysis (train/val only)
        if split in ['train', 'val']:
            analysis = create_performance_analysis_json(dataset_root, split)
            if analysis:
                analysis_path = output_dir / f'{split}_performance_analysis.json'
                with open(analysis_path, 'w') as f:
                    json.dump(analysis, f, indent=2)
                print(f"  ✓ {analysis_path.name}")
    
    print(f"\n{'='*70}")
    print(f"✅ METADATA GENERATED")
    print(f"{'='*70}")
    print(f"Location: {output_dir}")


if __name__ == '__main__':
    main()
