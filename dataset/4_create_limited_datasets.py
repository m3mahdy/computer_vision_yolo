"""
4. Create Limited Datasets.

Creates limited datasets using sophisticated representative sampling.
Ensures diverse coverage across weather, scene, time, and class combinations.

Configurations:
1. bdd100k_yolo_limited - Balanced dataset (~25K train, 30-40% coverage)
2. bdd100k_yolo_tuning  - Tuning dataset (~14K train, 20% coverage)
3. bdd100k_yolo_tiny    - Tiny dataset (~500 train, fast testing)

Source hierarchy (READ-ONLY):
- Config 1 sources from: bdd100k_yolo (created by script 2)
- Config 2 sources from: bdd100k_yolo_limited (Config 1, created by this script)
- Config 3 sources from: bdd100k_yolo_limited (Config 1, created by this script)

Metadata (REQUIRED):
- Script 3 creates representative_json metadata for source datasets
- Metadata is mandatory - script will fail if not present
- Run script 3 on source dataset before creating limited datasets

Usage:
    python dataset/4_create_limited_datasets.py
"""

import json
from pathlib import Path
import shutil
from tqdm import tqdm

from bdd100k_config import (LIMITED_DATASET_CONFIGS, YOLO_DATASET_ROOT, 
                             BDD100K_CLASSES, CLASS_TO_IDX, REPRESENTATIVE_ATTRIBUTES)


def load_metadata_from_source(source_root, split_name):
    """Load metadata from source dataset's representative_json (READ-ONLY, REQUIRED)."""
    metadata_file = source_root / 'representative_json' / f'{split_name}_metadata.json'
    
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata required: {metadata_file}\n"
            f"Run script 3 on source dataset first to create metadata."
        )
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"    ✓ Loaded metadata: {len(metadata.get('files', {}))} files")
        return metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata from {metadata_file}: {e}")


def get_label_attributes_from_metadata(metadata, basename):
    """Extract attributes for a specific image from metadata (READ-ONLY)."""
    if not metadata or 'files' not in metadata:
        return None
    
    file_info = metadata['files'].get(basename)
    if not file_info:
        return None
    
    return {
        'weather': file_info.get('weather', 'undefined'),
        'scene': file_info.get('scene', 'undefined'),
        'timeofday': file_info.get('timeofday', 'undefined'),
        'categories': file_info.get('categories', []),
        'class_counts': file_info.get('class_counts', {}),
        'num_objects': file_info.get('object_count', 0)
    }


def select_representative_samples(source_root, split_name, config, base_dir, constrain_to_basenames=None):
    """
    Select representative samples with sophisticated attribute-based sampling.
    Ensures diverse coverage across weather, scene, time, and class combinations.
    READ-ONLY: Only reads from source, never modifies it.
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
    
    # Load metadata from source dataset (READ-ONLY, REQUIRED)
    metadata = load_metadata_from_source(source_root, split_name)
    
    # Get available basenames from source (READ-ONLY)
    source_labels_dir = source_root / 'labels' / split_name
    if not source_labels_dir.exists():
        print(f"  ⚠️  Source labels not found: {source_labels_dir}")
        return set()
    
    available_basenames = {f.stem for f in source_labels_dir.glob('*.txt')}
    
    # Constrain to specified basenames if provided (for hierarchical datasets)
    if constrain_to_basenames:
        available_basenames &= constrain_to_basenames
        print(f"    Constrained to: {len(available_basenames):,} files from source")
    
    print(f"    Analyzing: {len(available_basenames):,} files")
    
    # Organize by attributes
    attribute_combo_groups = {}
    class_samples = {class_id: [] for class_id in range(len(BDD100K_CLASSES))}
    weather_samples = {w: [] for w in REPRESENTATIVE_ATTRIBUTES['weather']}
    scene_samples = {s: [] for s in REPRESENTATIVE_ATTRIBUTES['scene']}
    timeofday_samples = {t: [] for t in REPRESENTATIVE_ATTRIBUTES['timeofday']}
    class_attribute_samples = {}
    
    # Analyze files using metadata from source (READ-ONLY)
    for basename in tqdm(available_basenames, desc="  Analyzing", leave=False):
        attrs = get_label_attributes_from_metadata(metadata, basename)
        
        if not attrs or not attrs['categories']:
            continue
        
        file_info = {'basename': basename, 'attrs': attrs}
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
    
    selected_basenames = set()
    
    # Step 1: Select by attribute combinations
    for combo_key, files in attribute_combo_groups.items():
        sorted_files = sorted(files, key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']), reverse=True)
        num_to_select = min(samples_per_combo, len(sorted_files))
        selected_basenames.update(s['basename'] for s in sorted_files[:num_to_select])
    
    # Step 2: Ensure min per class
    for class_id, samples in class_samples.items():
        if not samples:
            continue
        current_count = sum(1 for s in samples if s['basename'] in selected_basenames)
        if current_count < min_per_class:
            sorted_samples = sorted([s for s in samples if s['basename'] not in selected_basenames],
                                  key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']), reverse=True)
            needed = min_per_class - current_count
            selected_basenames.update(s['basename'] for s in sorted_samples[:needed])
    
    # Step 3: Ensure min per attribute value
    for attr_dict in [weather_samples, scene_samples, timeofday_samples]:
        for attr_value, samples in attr_dict.items():
            if not samples:
                continue
            current_count = sum(1 for s in samples if s['basename'] in selected_basenames)
            if current_count < min_per_attr:
                sorted_samples = sorted([s for s in samples if s['basename'] not in selected_basenames],
                                      key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']), reverse=True)
                needed = min_per_attr - current_count
                selected_basenames.update(s['basename'] for s in sorted_samples[:needed])
    
    # Step 4: Ensure min per class×attribute combo
    for (class_id, attr_type, attr_value), samples in class_attribute_samples.items():
        if not samples:
            continue
        current_count = sum(1 for s in samples if s['basename'] in selected_basenames)
        if current_count < min_per_class_attr:
            sorted_samples = sorted([s for s in samples if s['basename'] not in selected_basenames],
                                  key=lambda x: (len(set(x['attrs']['categories'])), x['attrs']['num_objects']), reverse=True)
            needed = min_per_class_attr - current_count
            selected_basenames.update(s['basename'] for s in sorted_samples[:needed])
    
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
    """Copy image and label files for specified basenames (READ-ONLY on source)."""
    total_copied = 0
    
    for split in splits:
        basenames = basenames_by_split.get(split, set())
        if not basenames:
            continue
        
        source_images = source_root / 'images' / split
        source_labels = source_root / 'labels' / split
        output_images = output_root / 'images' / split
        output_labels = output_root / 'labels' / split
        
        # Verify source exists (READ-ONLY check)
        if not source_images.exists() or not source_labels.exists():
            print(f"  ⚠️  Source {split} not found, skipping")
            continue
        
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


def display_selection_summary_and_confirm(output_root, config_name, basenames_by_split, source_metadata_by_split):
    """Display selection summary and get user confirmation before copying files."""
    print(f"\n{'='*70}")
    print("SELECTION SUMMARY")
    print(f"{'='*70}")
    
    total_images = 0
    total_objects = 0
    
    for split in sorted(basenames_by_split.keys()):
        basenames = basenames_by_split[split]
        source_metadata = source_metadata_by_split.get(split)
        
        split_objects = 0
        split_classes = {}
        
        if source_metadata and 'files' in source_metadata:
            for basename in basenames:
                file_info = source_metadata['files'].get(basename, {})
                obj_count = file_info.get('object_count', 0)
                split_objects += obj_count
                
                # Count per class
                class_counts = file_info.get('class_counts', {})
                for cls, count in class_counts.items():
                    split_classes[cls] = split_classes.get(cls, 0) + count
        
        print(f"\n{split.upper()}:")
        print(f"  Images: {len(basenames):,}")
        print(f"  Objects: {split_objects:,}")
        print(f"  Avg objects/image: {split_objects / len(basenames):.2f}" if basenames else "  Avg: 0")
        
        if split_classes:
            print(f"  Classes distribution:")
            for cls in sorted(split_classes.keys()):
                print(f"    - {cls}: {split_classes[cls]:,}")
        
        total_images += len(basenames)
        total_objects += split_objects
    
    print(f"\n{'-'*70}")
    print(f"TOTAL:")
    print(f"  Images: {total_images:,}")
    print(f"  Objects: {total_objects:,}")
    print(f"  Avg objects/image: {total_objects / total_images:.2f}" if total_images > 0 else "  Avg: 0")
    print(f"{'='*70}")
    
    response = input("\nProceed with copying files? (y/N): ").strip().lower()
    return response == 'y'


def create_metadata_json(output_root, config_name, split_name, basenames, source_metadata):
    """Create metadata JSON file for output dataset split."""
    from datetime import datetime
    
    # Create metadata structure
    metadata = {
        'dataset': config_name,
        'split': split_name,
        'created': datetime.now().isoformat(),
        'total_files': len(basenames),
        'files': {}
    }
    
    # Copy file info from source metadata
    files_with_objects = 0
    total_objects = 0
    
    for basename in basenames:
        if basename in source_metadata['files']:
            file_info = source_metadata['files'][basename]
            metadata['files'][basename] = file_info
            
            if file_info.get('object_count', 0) > 0:
                files_with_objects += 1
                total_objects += file_info.get('object_count', 0)
    
    # Add summary statistics
    metadata['summary'] = {
        'files_with_objects': files_with_objects,
        'total_objects': total_objects,
        'avg_objects_per_file': total_objects / files_with_objects if files_with_objects > 0 else 0
    }
    
    # Write metadata JSON
    output_dir = output_root / 'representative_json'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'{split_name}_metadata.json'
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Metadata: {output_file.name}")
    return metadata


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
    
    # Determine source (READ-ONLY)
    source_name = config.get('source_dataset', 'full')
    source_root = YOLO_DATASET_ROOT if source_name == 'full' else base_dir / source_name
    
    print(f"\nSource: {source_root}")
    
    # Validate source exists (READ-ONLY check)
    if not source_root.exists():
        print(f"\n❌ Source dataset not found: {source_root}")
        if source_name != 'full':
            print(f"   Create '{source_name}' first before creating this dataset")
        return
    
    if not (source_root / 'data.yaml').exists():
        print(f"\n❌ Source data.yaml not found: {source_root / 'data.yaml'}")
        return
    
    # Validate required splits exist in source (READ-ONLY check)
    for split in config.get('splits', []):
        if not (source_root / 'labels' / split).exists():
            print(f"\n❌ Source {split} split not found: {source_root / 'labels' / split}")
            return
    
    print(f"✓ Source validated")
    
    output_root = base_dir / config['name']
    if output_root.exists():
        print(f"\n⚠️  Output already exists: {output_root}")
        response = input("   Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
        shutil.rmtree(output_root)
    
    basenames_by_split = {}
    source_metadata_by_split = {}
    
    # STEP 1: Load metadata and select representative samples (READ-ONLY)
    print(f"\n{'='*70}")
    print("STEP 1: ANALYZING AND SELECTING SAMPLES")
    print(f"{'='*70}")
    
    for split in config.get('splits', ['train', 'val', 'test']):
        source_labels_dir = source_root / 'labels' / split
        
        if not source_labels_dir.exists():
            print(f"\n⚠️  {split} not found in source, skipping")
            continue
        
        # Load source metadata for this split (READ-ONLY)
        source_metadata = load_metadata_from_source(source_root, split)
        source_metadata_by_split[split] = source_metadata
        
        # Check if should use full split
        use_full = (
            (split == 'test' and config.get('contain_full_test_split', False)) or
            (split == 'val' and config.get('contain_full_val_split', False))
        )
        
        if use_full:
            # Read all basenames from source (READ-ONLY)
            label_files = list(source_labels_dir.glob('*.txt'))
            basenames_by_split[split] = {f.stem for f in label_files}
            print(f"\n{split}: Using FULL split ({len(basenames_by_split[split]):,} files from source)")
        else:
            # Get constraint if hierarchical (READ-ONLY)
            constrain_to = None
            if source_name != 'full':
                # Only select from files that exist in source
                constrain_to = {f.stem for f in source_labels_dir.glob('*.txt')}
                print(f"\n{split}: Hierarchical selection from {len(constrain_to):,} source files")
            else:
                print(f"\n{split}: Representative selection from full dataset")
            
            # Select representative samples (READ-ONLY on source)
            basenames_by_split[split] = select_representative_samples(
                source_root, split, config, base_dir, constrain_to
            )
    
    # STEP 2: Create metadata JSON files for output dataset
    print(f"\n{'='*70}")
    print("STEP 2: CREATING METADATA JSON FILES")
    print(f"{'='*70}")
    
    for split in config['splits']:
        if split in basenames_by_split and split in source_metadata_by_split:
            if source_metadata_by_split[split]:  # If metadata available
                create_metadata_json(
                    output_root, 
                    config['name'], 
                    split, 
                    basenames_by_split[split],
                    source_metadata_by_split[split]
                )
    
    # STEP 3: Display summary and get user approval
    if not display_selection_summary_and_confirm(output_root, config['name'], basenames_by_split, source_metadata_by_split):
        print("\n❌ Operation cancelled by user.")
        # Clean up created metadata files
        metadata_dir = output_root / 'representative_json'
        if metadata_dir.exists():
            shutil.rmtree(metadata_dir)
        return
    
    # STEP 4: Copy files after approval
    print(f"\n{'='*70}")
    print("STEP 3: COPYING FILES")
    print(f"{'='*70}")
    total = copy_dataset_files(source_root, output_root, config['splits'], basenames_by_split)
    
    # STEP 5: Create data.yaml
    create_data_yaml(output_root, config)
    
    print(f"\n{'='*70}")
    print(f"✅ {config['name']} CREATED")
    print(f"{'='*70}")
    print(f"Total files: {total:,}")
    print(f"Location: {output_root}")


if __name__ == '__main__':
    main()
