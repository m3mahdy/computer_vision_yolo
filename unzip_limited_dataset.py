"""
Utility script to extract the compressed limited BDD100K YOLO dataset.

This script extracts bdd100k_yolo_limited.zip for quick experiments without
needing to process the full 100K dataset.

Usage:
    python unzip_limited_dataset.py
"""

import zipfile
from pathlib import Path
from tqdm import tqdm


def unzip_limited_dataset():
    """Extract the compressed limited dataset."""
    base_dir = Path(__file__).parent
    compressed_file = base_dir / 'bdd100k_yolo_limited.zip'
    extract_dir = base_dir / 'bdd100k_yolo_limited'
    
    # Validate compressed file exists
    if not compressed_file.exists():
        print(f"ERROR: Compressed file not found: {compressed_file}")
        print("\nPlease ensure bdd100k_yolo_limited.zip is in the current directory.")
        print("You can generate it by running: python process_bdd100k_to_yolo_dataset.py")
        return
    
    # Check if already extracted
    if extract_dir.exists() and (extract_dir / 'data.yaml').exists():
        response = input(f"\nDataset already extracted at: {extract_dir}\nOverwrite? (y/n): ")
        if response.lower() != 'y':
            print("Extraction cancelled.")
            return
        print("Overwriting existing dataset...")
    
    print(f"Extracting {compressed_file.name}...")
    print(f"Target directory: {extract_dir}")
    
    with zipfile.ZipFile(compressed_file, 'r') as zipf:
        # Get list of all files
        all_files = zipf.namelist()
        
        # Extract with progress bar
        for file in tqdm(all_files, desc="Extracting", unit='files'):
            zipf.extract(file, base_dir)
    
    # Verify extraction
    if extract_dir.exists() and (extract_dir / 'data.yaml').exists():
        print(f"\n✓ Successfully extracted to: {extract_dir}")
        
        # Count extracted files
        image_dirs = ['train/images', 'val/images', 'test/images']
        label_dirs = ['train/labels', 'val/labels', 'test/labels']
        
        total_images = sum(len(list((extract_dir / d).glob('*.jpg'))) 
                          for d in image_dirs if (extract_dir / d).exists())
        total_labels = sum(len(list((extract_dir / d).glob('*.txt'))) 
                          for d in label_dirs if (extract_dir / d).exists())
        
        print(f"  Images: {total_images:,}")
        print(f"  Labels: {total_labels:,}")
        print(f"\nYou can now use this dataset for quick experiments!")
    else:
        print("\nERROR: Extraction may have failed. Please verify the archive.")


if __name__ == '__main__':
    unzip_limited_dataset()
