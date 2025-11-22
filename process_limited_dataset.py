"""
Utility script to extract the compressed limited BDD100K YOLO dataset.

This script:
1. Checks if bdd100k_yolo_limited.zip exists in bdd100k_yolo_limited_zipped/
2. If not found, downloads it from Google Drive using gdown
3. Extracts the dataset for quick experiments without needing to process the full 100K dataset

Usage:
    python process_limited_dataset.py
"""

import zipfile
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

# Google Drive file ID for bdd100k_yolo_limited.zip
GDRIVE_FILE_ID = '1ISNZ07CZtuuzMMU4wxAVI5YW_XI9at7z'


def check_gdown_installed():
    """Check if gdown is installed, install if not."""
    try:
        import gdown
        return True
    except ImportError:
        print("\ngdown not found. Installing...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
            print("✓ gdown installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("ERROR: Failed to install gdown")
            print("Please install manually: pip install gdown")
            return False


def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive using gdown."""
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        print(f"\nDownloading from Google Drive...")
        print(f"File ID: {file_id}")
        print(f"Destination: {output_path}")
        
        gdown.download(url, str(output_path), quiet=False)
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Download complete: {size_mb:.1f} MB")
            return True
        else:
            print("ERROR: Download failed")
            return False
            
    except Exception as e:
        print(f"ERROR downloading file: {e}")
        return False


def process_limited_dataset():
    """Extract the compressed limited dataset."""
    base_dir = Path(__file__).parent
    zipped_dir = base_dir / 'bdd100k_yolo_limited_zipped'
    compressed_file = zipped_dir / 'bdd100k_yolo_limited.zip'
    extract_dir = base_dir / 'bdd100k_yolo_limited'
    
    # Create zipped directory if it doesn't exist
    zipped_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if compressed file exists, download if not
    if not compressed_file.exists():
        print(f"Compressed file not found: {compressed_file}")
        print("\nAttempting to download from Google Drive...")
        
        # Check and install gdown if needed
        if not check_gdown_installed():
            print("\nERROR: Cannot proceed without gdown")
            print("Please install gdown and try again: pip install gdown")
            return
        
        # Download the file
        if not download_from_gdrive(GDRIVE_FILE_ID, compressed_file):
            print("\nERROR: Download failed")
            print("You can generate it by running: python process_bdd100k_to_yolo_dataset.py")
            return
        
        print("✓ Download successful!")
    else:
        size_mb = compressed_file.stat().st_size / (1024 * 1024)
        print(f"Found compressed file: {compressed_file}")
        print(f"Size: {size_mb:.1f} MB")
    
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
    process_limited_dataset()
