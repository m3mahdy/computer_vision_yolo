"""
BDD100K Dataset Compression Module.

This module handles compressing datasets into ZIP files for distribution.
"""

import zipfile
from pathlib import Path
from tqdm import tqdm

from bdd100k_config import BDD100K_CLASSES


def compress_test_split_only(yolo_dataset_root, base_dir):
    """
    Compress only the test split from the full dataset for quick distribution.
    Creates a standalone test dataset with data.yaml configured for test only.
    
    Args:
        yolo_dataset_root: Path to full YOLO dataset root
        base_dir: Base directory where compressed file will be saved
        
    Returns:
        Dict with compression info: 'path', 'size_mb', 'num_images', 'num_labels'
        Returns None if test split not found
    """
    print("\n" + "="*70)
    print("COMPRESSING TEST SPLIT ONLY")
    print("="*70)
    
    test_images_dir = yolo_dataset_root / 'images' / 'test'
    test_labels_dir = yolo_dataset_root / 'labels' / 'test'
    
    if not test_images_dir.exists() or not test_labels_dir.exists():
        print(f"⚠️  Test split not found in {yolo_dataset_root}")
        return None
    
    # Create output directory
    zipped_dir = base_dir / 'bdd100k_test_split_zipped'
    zipped_dir.mkdir(parents=True, exist_ok=True)
    compressed_file = zipped_dir / 'bdd100k_yolo_test_split.zip'
    
    # Remove existing compressed file
    if compressed_file.exists():
        print(f"Removing existing: {compressed_file.name}")
        compressed_file.unlink()
    
    print(f"\nCompressing test split...")
    print(f"  Source: {yolo_dataset_root}")
    print(f"  Destination: {compressed_file}")
    
    with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        # Count files first
        test_images = list(test_images_dir.glob('*'))
        test_labels = list(test_labels_dir.glob('*.txt'))
        
        # Add test images
        for img_file in tqdm(test_images, desc="  Compressing images", unit='files'):
            if img_file.is_file():
                arcname = Path('bdd100k_yolo_test') / 'images' / 'test' / img_file.name
                zipf.write(img_file, arcname=arcname)
        
        # Add test labels
        for label_file in tqdm(test_labels, desc="  Compressing labels", unit='files'):
            arcname = Path('bdd100k_yolo_test') / 'labels' / 'test' / label_file.name
            zipf.write(label_file, arcname=arcname)
        
        # Create and add test-only data.yaml
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
        
        # Copy test metadata and performance analysis files
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
    print(f"  # Run YOLO validation with data.yaml")
    
    return {
        'path': compressed_file,
        'size_mb': file_size_mb,
        'num_images': len(test_images),
        'num_labels': len(test_labels)
    }


def compress_limited_dataset(dataset_root, output_dir, dataset_name):
    """
    Compress a limited dataset into a ZIP file.
    
    Args:
        dataset_root: Path to dataset root directory
        output_dir: Directory where compressed file will be saved
        dataset_name: Name for the compressed file (without .zip extension)
        
    Returns:
        Dict with compression info: 'path', 'size_mb', 'total_files'
    """
    compressed_file = output_dir / f"{dataset_name}.zip"
    
    # Remove existing compressed file if present
    if compressed_file.exists():
        print(f"  Removing existing: {compressed_file.name}")
        compressed_file.unlink()
    
    print(f"  Compressing {dataset_name}...")
    
    total_files = 0
    
    with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        # Compress images and labels
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
        
        # Add data.yaml
        data_yaml = dataset_root / 'data.yaml'
        if data_yaml.exists():
            arcname = Path(dataset_name) / 'data.yaml'
            zipf.write(data_yaml, arcname=arcname)
            total_files += 1
        
        # Add metadata files
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
