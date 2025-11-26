"""
7. Compress Test Split Only.

Compresses only the test split from the full dataset for quick distribution.
Creates a standalone test dataset with data.yaml configured for test only.

Source: bdd100k_yolo (full dataset)
Output: bdd100k_test_split_zipped/bdd100k_yolo_test_split.zip

Usage:
    python dataset/7_compress_test_only.py
"""

import zipfile
from pathlib import Path
from tqdm import tqdm

from bdd100k_config import YOLO_DATASET_ROOT, BDD100K_CLASSES


def compress_test_split(yolo_dataset_root, base_dir):
    """Compress only the test split for distribution."""
    print("\n" + "="*70)
    print("COMPRESSING TEST SPLIT ONLY")
    print("="*70)
    
    test_images_dir = yolo_dataset_root / 'images' / 'test'
    test_labels_dir = yolo_dataset_root / 'labels' / 'test'
    
    if not test_images_dir.exists() or not test_labels_dir.exists():
        print(f"⚠️  Test split not found in {yolo_dataset_root}")
        return None
    
    zipped_dir = base_dir / 'bdd100k_test_split_zipped'
    zipped_dir.mkdir(parents=True, exist_ok=True)
    compressed_file = zipped_dir / 'bdd100k_yolo_test_split.zip'
    
    if compressed_file.exists():
        print(f"Removing existing: {compressed_file.name}")
        compressed_file.unlink()
    
    print(f"\nCompressing test split...")
    print(f"  Source: {yolo_dataset_root}")
    print(f"  Destination: {compressed_file}")
    
    with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        test_images = list(test_images_dir.glob('*'))
        test_labels = list(test_labels_dir.glob('*.txt'))
        
        for img_file in tqdm(test_images, desc="  Compressing images", unit='files'):
            if img_file.is_file():
                arcname = Path('bdd100k_yolo_test') / 'images' / 'test' / img_file.name
                zipf.write(img_file, arcname=arcname)
        
        for label_file in tqdm(test_labels, desc="  Compressing labels", unit='files'):
            arcname = Path('bdd100k_yolo_test') / 'labels' / 'test' / label_file.name
            zipf.write(label_file, arcname=arcname)
        
        yaml_lines = [
            "path: .",
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
        zipf.writestr('bdd100k_yolo_test/data.yaml', test_data_yaml)
        
        for metadata_file in ['test_metadata.json', 'test_performance_analysis.json']:
            src_file = yolo_dataset_root / 'representative_json' / metadata_file
            if src_file.exists():
                arcname = Path('bdd100k_yolo_test') / 'representative_json' / metadata_file
                zipf.write(src_file, arcname=arcname)
    
    file_size_mb = compressed_file.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ Test split compressed successfully!")
    print(f"  Location: {compressed_file}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Images: {len(test_images):,}")
    print(f"  Labels: {len(test_labels):,}")
    print(f"\nTo extract and use:")
    print(f"  unzip {compressed_file.name}")
    print(f"  cd bdd100k_yolo_test")
    
    return {
        'path': compressed_file,
        'size_mb': file_size_mb,
        'num_images': len(test_images),
        'num_labels': len(test_labels)
    }


def main():
    """Main function."""
    base_dir = Path(__file__).parent.parent
    
    print(f"\n{'='*70}")
    print("COMPRESS TEST SPLIT ONLY")
    print(f"{'='*70}")
    print(f"Source: {YOLO_DATASET_ROOT}")
    print(f"Output: bdd100k_test_split_zipped/")
    
    if not YOLO_DATASET_ROOT.exists() or not (YOLO_DATASET_ROOT / 'data.yaml').exists():
        print(f"\n❌ Full dataset not found: {YOLO_DATASET_ROOT}")
        print("Create full dataset first with script 2")
        return
    
    test_images_dir = YOLO_DATASET_ROOT / 'images' / 'test'
    test_labels_dir = YOLO_DATASET_ROOT / 'labels' / 'test'
    
    if not test_images_dir.exists() or not test_labels_dir.exists():
        print(f"\n❌ Test split not found in full dataset")
        print(f"Expected:")
        print(f"  {test_images_dir}")
        print(f"  {test_labels_dir}")
        return
    
    result = compress_test_split(YOLO_DATASET_ROOT, base_dir)
    
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
