"""
7. Compress Test Split Only (READ-ONLY).

Compresses only the test split from the first limited dataset in config.
Creates a standalone test dataset with data.yaml configured for test only.
Read-only operation - never modifies source data.

Source: First limited dataset from config (LIMITED_DATASET_CONFIGS[0])
Output: bdd100k_limited_datasets_zipped/{dataset_name}_test_split.zip

Usage:
    python dataset/7_compress_test_only.py
"""

import zipfile
from pathlib import Path
from tqdm import tqdm

from bdd100k_config import BDD100K_CLASSES, LIMITED_DATASET_CONFIGS


def compress_test_split(dataset_root, dataset_name, output_dir):
    """Compress only the test split for distribution (READ-ONLY)."""
    print("\n" + "="*70)
    print(f"COMPRESSING TEST SPLIT ONLY (READ-ONLY): {dataset_name}")
    print("="*70)
    
    test_images_dir = dataset_root / 'images' / 'test'
    test_labels_dir = dataset_root / 'labels' / 'test'
    
    if not test_images_dir.exists() or not test_labels_dir.exists():
        print(f"⚠️  Test split not found in {dataset_root}")
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    compressed_file = output_dir / f"{dataset_name}_test_split.zip"
    
    if compressed_file.exists():
        print(f"Removing existing: {compressed_file.name}")
        compressed_file.unlink()
    
    print(f"\nCompressing test split...")
    print(f"  Source: {dataset_root}")
    print(f"  Destination: {compressed_file}")
    print(f"  Note: Source data will not be modified.")
    
    archive_name = f"{dataset_name}_test"
    
    with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        test_images = list(test_images_dir.glob('*'))
        test_labels = list(test_labels_dir.glob('*.txt'))
        
        for img_file in tqdm(test_images, desc="  Compressing images", unit='files'):
            if img_file.is_file():
                arcname = Path(archive_name) / 'images' / 'test' / img_file.name
                zipf.write(img_file, arcname=arcname)
        
        for label_file in tqdm(test_labels, desc="  Compressing labels", unit='files'):
            arcname = Path(archive_name) / 'labels' / 'test' / label_file.name
            zipf.write(label_file, arcname=arcname)
        
        # Create test-only data.yaml
        yaml_lines = [
            "path: /computer_vision_yolo/bdd100k_yolo_limited",
            "",
            "test: images/test",
            "",
            f"nc: {len(BDD100K_CLASSES)}",
            "",
            "names:"
        ]
        for class_name in BDD100K_CLASSES:
            yaml_lines.append(f"- {class_name}")
        
        test_data_yaml = "\n".join(yaml_lines)
        zipf.writestr(f'{archive_name}/data.yaml', test_data_yaml)
        
        # Include test metadata if available
        for metadata_file in ['test_metadata.json', 'test_performance_analysis.json']:
            src_file = dataset_root / 'representative_json' / metadata_file
            if src_file.exists():
                arcname = Path(archive_name) / 'representative_json' / metadata_file
                zipf.write(src_file, arcname=arcname)
    
    file_size_mb = compressed_file.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ Test split compressed successfully!")
    print(f"  Location: {compressed_file}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Images: {len(test_images):,}")
    print(f"  Labels: {len(test_labels):,}")
    print(f"\nTo extract and use:")
    print(f"  unzip {compressed_file.name}")
    print(f"  cd {archive_name}")
    
    return {
        'path': compressed_file,
        'size_mb': file_size_mb,
        'num_images': len(test_images),
        'num_labels': len(test_labels)
    }


def main():
    """Main function (READ-ONLY operation)."""
    base_dir = Path(__file__).parent.parent
    
    # Get first limited dataset from config
    if not LIMITED_DATASET_CONFIGS:
        print("\n❌ No limited datasets in config")
        return
    
    config = LIMITED_DATASET_CONFIGS[0]
    dataset_name = config['name']
    dataset_root = base_dir / dataset_name
    
    print(f"\n{'='*70}")
    print(f"COMPRESS TEST SPLIT (READ-ONLY): {dataset_name}")
    print(f"{'='*70}")
    print(f"Source: {config['description']}")
    print(f"Path: {dataset_root}")
    print(f"{'='*70}")
    
    # Check dataset exists
    if not dataset_root.exists() or not (dataset_root / 'data.yaml').exists():
        print(f"\n❌ Dataset not found: {dataset_root}")
        print("Create limited datasets first with script 4")
        return
    
    # Check test split exists
    test_images_dir = dataset_root / 'images' / 'test'
    test_labels_dir = dataset_root / 'labels' / 'test'
    
    if not test_images_dir.exists() or not test_labels_dir.exists():
        print(f"\n❌ Test split not found in {dataset_name}")
        print(f"Expected:")
        print(f"  {test_images_dir}")
        print(f"  {test_labels_dir}")
        return
    
    # Output directory
    output_dir = base_dir / 'bdd100k_limited_datasets_zipped'
    
    result = compress_test_split(dataset_root, dataset_name, output_dir)
    
    if result:
        print(f"\n{'='*70}")
        print(f"✅ TEST SPLIT COMPRESSED")
        print(f"{'='*70}")
        print(f"File: {result['path']}")
        print(f"Size: {result['size_mb']:.1f} MB")
        print(f"Images: {result['num_images']:,}")
        print(f"Labels: {result['num_labels']:,}")
    else:
        print("\n❌ Compression failed")


if __name__ == '__main__':
    main()
