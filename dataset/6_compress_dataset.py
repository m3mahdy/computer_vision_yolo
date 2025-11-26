"""
6. Compress Dataset Files.

Compresses selected dataset into ZIP file for distribution.
Allows selecting which dataset to compress.

Usage:
    python dataset/6_compress_dataset.py
"""

from pathlib import Path

from bdd100k_config import LIMITED_DATASET_CONFIGS
from bdd100k_compress import compress_limited_dataset


def select_dataset():
    """Display menu and select dataset."""
    base_dir = Path(__file__).parent.parent
    
    # Available datasets
    datasets = []
    
    # Full dataset
    full_dataset = base_dir / 'bdd100k_yolo'
    if full_dataset.exists() and (full_dataset / 'data.yaml').exists():
        datasets.append({
            'id': 1,
            'name': 'bdd100k_yolo',
            'path': full_dataset,
            'description': 'Full BDD100K dataset (~70K train)'
        })
    
    # Limited datasets
    for idx, config in enumerate(LIMITED_DATASET_CONFIGS, start=2):
        dataset_path = base_dir / config['name']
        if dataset_path.exists() and (dataset_path / 'data.yaml').exists():
            datasets.append({
                'id': idx,
                'name': config['name'],
                'path': dataset_path,
                'description': config['description']
            })
    
    if not datasets:
        print("\n❌ ERROR: No datasets found")
        print("Create datasets first with scripts 2 and 3")
        return None
    
    print("\n" + "="*70)
    print("SELECT DATASET TO COMPRESS")
    print("="*70)
    
    for ds in datasets:
        print(f"\n[{ds['id']}] {ds['name']}")
        print(f"    {ds['description']}")
    
    print("\n[0] Cancel")
    print("="*70)
    
    while True:
        choice = input("\nSelect dataset (0-{}): ".format(len(datasets))).strip()
        if choice == '0':
            return None
        
        try:
            choice_int = int(choice)
            for ds in datasets:
                if ds['id'] == choice_int:
                    return ds
            print("Invalid choice. Try again.")
        except ValueError:
            print("Invalid input. Enter a number.")


def main():
    """Main function."""
    base_dir = Path(__file__).parent.parent
    
    # Select dataset
    dataset = select_dataset()
    if not dataset:
        print("\nCancelled.")
        return
    
    dataset_root = dataset['path']
    dataset_name = dataset['name']
    
    print(f"\n{'='*70}")
    print(f"COMPRESSING: {dataset_name}")
    print(f"{'='*70}")
    print(f"Location: {dataset_root}")
    
    # Create output directory for compressed files
    if dataset_name == 'bdd100k_yolo':
        output_dir = base_dir / 'bdd100k_zipped'
    else:
        output_dir = base_dir / 'bdd100k_limited_datasets_zipped'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compress dataset
    result = compress_limited_dataset(dataset_root, output_dir, dataset_name)
    
    if result:
        print(f"\n{'='*70}")
        print(f"✅ COMPRESSION COMPLETE")
        print(f"{'='*70}")
        print(f"File: {result['path']}")
        print(f"Size: {result['size_mb']:.1f} MB")
        print(f"Total files: {result['total_files']:,}")
    else:
        print("\n❌ Compression failed")


if __name__ == '__main__':
    main()
