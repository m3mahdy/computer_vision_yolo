"""
3. Create Limited Datasets.

Creates limited datasets using sophisticated representative sampling.
Ensures diverse coverage across weather, scene, time, and class combinations.

Configurations:
1. bdd100k_yolo_limited - Balanced dataset (~25K train, 30-40% coverage)
2. bdd100k_yolo_tuning  - Tuning dataset (~14K train, 20% coverage)
3. bdd100k_yolo_tiny    - Tiny dataset (~500 train, fast testing)

Source hierarchy:
- Config 1 sources from: Full dataset (bdd100k_yolo)
- Config 2 sources from: Config 1 (bdd100k_yolo_limited) 
- Config 3 sources from: Config 1 (bdd100k_yolo_limited)

Usage:
    python dataset/3_create_limited_datasets.py
"""

import json
from pathlib import Path
import shutil
from tqdm import tqdm

from bdd100k_config import (LIMITED_DATASET_CONFIGS, YOLO_DATASET_ROOT, 
                             BDD100K_CLASSES, CLASS_TO_IDX, REPRESENTATIVE_ATTRIBUTES)


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


def select_representative_samples(split_labels_src, split_name, config, base_dir, constrain_to_basenames=None):
    """
    Select representative samples with sophisticated attribute-based sampling.
    Ensures diverse coverage across weather, scene, time, and class combinations.
    """
    samples_per_combo = config['samples_per_attribute_combo']
    min_per_class = config['min_samples_per_class']
    min_per_attr = config['min_samples_per_attribute_value']
    min_per_class_attr = config['min_samples_per_class_attribute_combo']
    
    print(f"\n  Selecting representative samples...")
    print(f"    - {samples_per_combo} per attribute combo")
    print(f"    - {min_per_class} per class")
    print(f"    - {min_per_attr} per attribute value")
    print(f"    - {min_per_class_attr} per class×attribute")
    
    # Get JSON files from tmp labels for attributes
    tmp_labels_dir = base_dir / 'bdd100k_tmp_labels' / '100k' / split_name
    if not tmp_labels_dir.exists():
        print(f"  ⚠️  No source JSON labels found: {tmp_labels_dir}")
        return set()
    
    json_files = list(tmp_labels_dir.glob('*.json'))
    if constrain_to_basenames:
        json_files = [f for f in json_files if f.stem in constrain_to_basenames]
    
    # Organize by attributes
    attribute_combo_groups = {}
    class_samples = {class_id: [] for class_id in range(len(BDD100K_CLASSES))}
    weather_samples = {w: [] for w in REPRESENTATIVE_ATTRIBUTES['weather']}
    scene_samples = {s: [] for s in REPRESENTATIVE_ATTRIBUTES['scene']}
    timeofday_samples = {t: [] for t in REPRESENTATIVE_ATTRIBUTES['timeofday']}
    class_attribute_samples = {}
    
    for json_file in tqdm(json_files, desc="  Analyzing", leave=False):
        attrs = get_label_attributes(json_file)
        if not attrs or not attrs['categories']:
            continue
        
        file_info = {'path': json_file, 'attrs': attrs}
        combo_key = (attrs['weather'], attrs['scene'], attrs['timeofday'])
        
        attribute_combo_groups.setdefault(combo_key, []).append(file_info)
        weather_samples[attrs['weather']].append(file_info)
        scene_samples[attrs['scene']].append(file_info)
        timeofday_samples[attrs['timeofday']].append(file_info)
        
        for cat in attrs['categories']:
            if cat in CLASS_TO_IDX:
                class_id = CLASS_TO_IDX[cat]
                class_samples[class_id].append(file_info)
                
                for attr_type, attr_value in [('weather', attrs['weather']), ('scene', attrs['scene']), ('timeofday', attrs['timeofday'])]:
                    combo = (class_id, attr_type, attr_value)
                    class_attribute_samples.setdefault(combo, []).append(file_info)
    
    selected_files = set()
    
    # Step 1: Select by attribute combinations
    for combo_key, files in attribute_combo_groups.items():
        sorted_files = sorted(files, key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']), reverse=True)
        num_to_select = min(samples_per_combo, len(sorted_files))
        selected_files.update(s['path'] for s in sorted_files[:num_to_select])
    
    # Step 2: Ensure min per class
    for class_id, samples in class_samples.items():
        if not samples:
            continue
        current_count = sum(1 for s in samples if s['path'] in selected_files)
        if current_count < min_per_class:
            sorted_samples = sorted([s for s in samples if s['path'] not in selected_files],
                                  key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']), reverse=True)
            needed = min_per_class - current_count
            selected_files.update(s['path'] for s in sorted_samples[:needed])
    
    # Step 3: Ensure min per attribute value
    for attr_dict in [weather_samples, scene_samples, timeofday_samples]:
        for attr_value, samples in attr_dict.items():
            if not samples:
                continue
            current_count = sum(1 for s in samples if s['path'] in selected_files)
            if current_count < min_per_attr:
                sorted_samples = sorted([s for s in samples if s['path'] not in selected_files],
                                      key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']), reverse=True)
                needed = min_per_attr - current_count
                selected_files.update(s['path'] for s in sorted_samples[:needed])
    
    # Step 4: Ensure min per class×attribute combo
    for (class_id, attr_type, attr_value), samples in class_attribute_samples.items():
        if not samples:
            continue
        current_count = sum(1 for s in samples if s['path'] in selected_files)
        if current_count < min_per_class_attr:
            sorted_samples = sorted([s for s in samples if s['path'] not in selected_files],
                                  key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']), reverse=True)
            needed = min_per_class_attr - current_count
            selected_files.update(s['path'] for s in sorted_samples[:needed])
    
    selected_basenames = {f.stem for f in selected_files}
    print(f"  ✓ Selected {len(selected_basenames)} representative samples")
    
    return selected_basenames


def select_config():
    """Display menu and select configuration."""
    print("\n" + "="*70)
    print("SELECT LIMITED DATASET CONFIGURATION")
    print("="*70)
    
    for config in LIMITED_DATASET_CONFIGS:
        print(f"\n[{config['id']}] {config['name']}")
        print(f"    {config['description']}")
        print(f"    Source: {config['source_dataset']}")
    
    print("\n[0] Cancel")
    print("="*70)
    
    while True:
        choice = input("\nSelect (0-3): ").strip()
        if choice == '0':
            return None
        try:
            choice_int = int(choice)
            for config in LIMITED_DATASET_CONFIGS:
                if config['id'] == choice_int:
                    return config
        except ValueError:
            pass
        print("Invalid choice")


def copy_dataset_files(source_root, output_root, splits, basenames_by_split):
    """Copy image and label files for specified basenames."""
    total_copied = 0
    
    for split in splits:
        basenames = basenames_by_split.get(split, set())
        if not basenames:
            continue
        
        source_images = source_root / 'images' / split
        source_labels = source_root / 'labels' / split
        output_images = output_root / 'images' / split
        output_labels = output_root / 'labels' / split
        
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        for basename in tqdm(basenames, desc=f"  Copying {split}", unit='files'):
            for ext in ['.jpg', '.png', '.jpeg']:
                img_file = source_images / f"{basename}{ext}"
                if img_file.exists():
                    shutil.copy2(img_file, output_images / img_file.name)
                    total_copied += 1
                    break
            
            label_file = source_labels / f"{basename}.txt"
            if label_file.exists():
                shutil.copy2(label_file, output_labels / label_file.name)
    
    return total_copied


def create_data_yaml(dataset_root, config):
    """Create data.yaml for dataset."""
    yaml_lines = [f"path: {dataset_root.absolute()}", ""]
    
    for split in ['train', 'val', 'test']:
        if split in config.get('splits', []):
            yaml_lines.append(f"{split}: images/{split}")
    
    yaml_lines.extend(["", f"nc: {len(BDD100K_CLASSES)}", "", "names:"])
    yaml_lines.extend(f"- {cls}" for cls in BDD100K_CLASSES)
    
    yaml_path = dataset_root / 'data.yaml'
    yaml_path.write_text("\n".join(yaml_lines))
    print(f"\n✓ Created: {yaml_path}")


def main():
    """Main function."""
    base_dir = Path(__file__).parent.parent
    
    config = select_config()
    if not config:
        print("\nCancelled.")
        return
    
    print(f"\n{'='*70}")
    print(f"CREATING: {config['name']}")
    print(f"{'='*70}")
    print(f"{config['description']}")
    
    # Determine source
    source_name = config.get('source_dataset', 'full')
    source_root = YOLO_DATASET_ROOT if source_name == 'full' else base_dir / source_name
    
    if not source_root.exists() or not (source_root / 'data.yaml').exists():
        print(f"\n❌ Source not found: {source_root}")
        return
    
    output_root = base_dir / config['name']
    basenames_by_split = {}
    
    # Process each split
    for split in config.get('splits', ['train', 'val', 'test']):
        source_labels_dir = source_root / 'labels' / split
        
        if not source_labels_dir.exists():
            continue
        
        # Check if should use full split
        use_full = (
            (split == 'test' and config.get('contain_full_test_split', False)) or
            (split == 'val' and config.get('contain_full_val_split', False))
        )
        
        if use_full:
            label_files = list(source_labels_dir.glob('*.txt'))
            basenames_by_split[split] = {f.stem for f in label_files}
            print(f"\n{split}: Using FULL split ({len(basenames_by_split[split]):,} files)")
        else:
            # Get constraint if hierarchical
            constrain_to = None
            if source_name != 'full':
                constrain_to = {f.stem for f in source_labels_dir.glob('*.txt')}
            
            # Select representative samples
            basenames_by_split[split] = select_representative_samples(
                source_labels_dir, split, config, base_dir, constrain_to
            )
    
    # Copy files
    print(f"\nCopying files...")
    total = copy_dataset_files(source_root, output_root, config['splits'], basenames_by_split)
    
    # Create data.yaml
    create_data_yaml(output_root, config)
    
    print(f"\n{'='*70}")
    print(f"✅ {config['name']} CREATED")
    print(f"{'='*70}")
    print(f"Total files: {total:,}")
    print(f"Location: {output_root}")


if __name__ == '__main__':
    main()
