"""
6. Compress Dataset Files.

Compresses selected dataset into ZIP file for distribution.
Allows selecting which dataset to compress.

Usage:
    python dataset/6_compress_dataset.py
"""

import zipfile
from pathlib import Path

from bdd100k_config import LIMITED_DATASET_CONFIGS, BDD100K_CLASSES


def compress_dataset(dataset_root, output_dir, dataset_name):
    """Compress a dataset into a ZIP file."""
    compressed_file = output_dir / f"{dataset_name}.zip"
    
    if compressed_file.exists():
        print(f"  Removing existing: {compressed_file.name}")
        compressed_file.unlink()
    
    print(f"  Compressing {dataset_name}...")
    
    total_files = 0
    
    with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for split_type in ['images', 'labels']:
            split_dir = dataset_root / split_type
            if split_dir.exists():
                for split in ['train', 'val', 'test']:
                    split_path = split_dir / split
                    if split_path.exists():
                        files = list(split_path.glob('*'))
                        for file_path in files:
                            if file_path.is_file():
                                arcname = Path(dataset_name) / split_type / split / file_path.name
                                zipf.write(file_path, arcname=arcname)
                                total_files += 1
        
        data_yaml = dataset_root / 'data.yaml'
        if data_yaml.exists():
            arcname = Path(dataset_name) / 'data.yaml'
            zipf.write(data_yaml, arcname=arcname)
            total_files += 1
        
        metadata_dir = dataset_root / 'representative_json'
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob('*.json'):
                arcname = Path(dataset_name) / 'representative_json' / metadata_file.name
                zipf.write(metadata_file, arcname=arcname)
                total_files += 1
    
    file_size_mb = compressed_file.stat().st_size / (1024 * 1024)
    
    print(f"    ✓ Compressed: {compressed_file.name} ({file_size_mb:.1f} MB)")
    
    return {
        'path': compressed_file,
        'size_mb': file_size_mb,
        'total_files': total_files
    }


def select_dataset():
    """Display menu and select dataset."""
    base_dir = Path(__file__).parent.parent
    
    datasets = []
    
    full_dataset = base_dir / 'bdd100k_yolo'
    if full_dataset.exists() and (full_dataset / 'data.yaml').exists():
        datasets.append({
            'id': 1,
            'name': 'bdd100k_yolo',
            'path': full_dataset,
            'description': 'Full BDD100K dataset (~70K train)'
        })
    
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
        print("\n❌ No datasets found")
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
        choice = input(f"\nSelect (0-{len(datasets)}): ").strip()
        if choice == '0':
            return None
        
        try:
            choice_int = int(choice)
            for ds in datasets:
                if ds['id'] == choice_int:
                    return ds
            print("Invalid choice")
        except ValueError:
            print("Invalid input")


def main():
    """Main function."""
    base_dir = Path(__file__).parent.parent
    
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
    
    if dataset_name == 'bdd100k_yolo':
        output_dir = base_dir / 'bdd100k_zipped'
    else:
        output_dir = base_dir / 'bdd100k_limited_datasets_zipped'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = compress_dataset(dataset_root, output_dir, dataset_name)
    
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
