"""
Unified BDD100K Dataset Download and Extract Script.

Downloads and extracts BDD100K datasets from Google Drive.
Supports multiple dataset variants:
1. Full dataset (100K images) - Source files
2. Limited dataset (~25K train images)
3. Tuning dataset (~14K train images)
4. Tiny dataset (~500 train images)

Usage:
    python dataset/bdd100k_download_extract.py
"""

import zipfile
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm


# Dataset configurations with Google Drive IDs
DATASETS = {
    '1': {
        'name': 'Full Dataset Source Files',
        'description': 'Original BDD100K images and labels (requires processing)',
        'files': [
            {
                'gdrive_id': '1yHEpeEdRDAz5yH4pbo4o1SvzKzGKRaLS',
                'filename': 'bdd100k_images_100k.zip',
                'size': '~5.28GB',
                'extract_to': 'bdd100k_tmp_images'
            },
            {
                'gdrive_id': '1Gh_5g-MAx1R5X3eNsTTdz_GPialECz0L',
                'filename': 'bdd100k_labels.zip',
                'size': '~180MB',
                'extract_to': 'bdd100k_tmp_labels'
            }
        ],
        'output_dir': 'bdd_100k_source'
    },
    '2': {
        'name': 'Limited Dataset (YOLO Ready)',
        'description': 'Balanced limited dataset - 30-40% coverage (~25K train)',
        'files': [
            {
                'gdrive_id': '1psk1Q9YUdV2e_xKzIV29hFfRjx4vGGmG',
                'filename': 'bdd100k_yolo_limited.zip',
                'size': '~3.3GB',
                'extract_to': '/computer_vision_yolo'  # Extracts to its own folder
            }
        ],
        'output_dir': 'bdd100k_limited_datasets_zipped'
    },
    '3': {
        'name': 'Tuning Dataset (YOLO Ready)',
        'description': 'Tuning dataset - 20% coverage (~14K train)',
        'files': [
            {
                'gdrive_id': '1QccaQ1tI_N3zXzp08Nemw13wl_02gUWG',
                'filename': 'bdd100k_yolo_tuning.zip',
                'size': '~1.58GB',
                'extract_to': '/computer_vision_yolo' 
            }
        ],
        'output_dir': 'bdd100k_limited_datasets_zipped'
    },
    '4': {
        'name': 'Tiny Dataset (YOLO Ready)',
        'description': 'Tiny dataset - ~500 train, ~1K total (fast testing)',
        'files': [
            {
                'gdrive_id': '1ftz8FuBT9yTf4ygSViSzjXJpuve-Zs55',
                'filename': 'bdd100k_yolo_tiny.zip',
                'size': '~100MB',
                'extract_to': '/computer_vision_yolo' 
            }
        ],
        'output_dir': 'bdd100k_limited_datasets_zipped'
    },
    '5': {
        'name': 'Test Split Only (YOLO Ready)',
        'description': 'Test split only for validation (20K images)',
        'files': [
            {
                'gdrive_id': '1OCrCDFHxQEKj3Y2Nm1PxWdCXMdEZp-YP',
                'filename': 'bdd100k_yolo_test_split.zip',
                'size': '~1.1GB',
                'extract_to': '/computer_vision_yolo' 
            }
        ],
        'output_dir': 'bdd100k_test_split_zipped'
    }
}


def check_gdown_installed():
    """Check if gdown is installed, install if not."""
    try:
        import gdown
        return True
    except ImportError:
        print("\n📦 gdown not found. Installing...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
            print("✓ gdown installed successfully\n")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install gdown")
            print("Please install manually: pip install gdown")
            return False


def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive using gdown."""
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        print(f"\n📥 Downloading from Google Drive...")
        print(f"   File ID: {file_id}")
        print(f"   Destination: {output_path.name}")
        
        gdown.download(url, str(output_path), quiet=False)
        
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Download complete: {size_mb:.1f} MB\n")
            return True
        else:
            print("❌ Download failed\n")
            return False
            
    except Exception as e:
        print(f"❌ Download error: {e}\n")
        return False


def extract_zip(zip_path, extract_to, base_dir):
    """Extract ZIP file with progress bar."""
    print(f"📦 Extracting {zip_path.name}...")
    
    if extract_to:
        target_dir = base_dir / extract_to
    else:
        target_dir = base_dir
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        members = zipf.namelist()
        
        for member in tqdm(members, desc="  Extracting", unit='files'):
            zipf.extract(member, target_dir)
    
    print(f"✓ Extracted to: {target_dir}\n")
    return target_dir


def show_menu():
    """Display dataset selection menu."""
    print("\n" + "="*70)
    print("BDD100K DATASET DOWNLOAD & EXTRACT")
    print("="*70)
    print("\nAvailable datasets:\n")
    
    for key, dataset in DATASETS.items():
        print(f"[{key}] {dataset['name']}")
        print(f"    {dataset['description']}")
        total_size = ', '.join([f['size'] for f in dataset['files']])
        print(f"    Size: {total_size}\n")
    
    print("[0] Exit")
    print("="*70)


def process_dataset(choice):
    """Download and extract selected dataset."""
    if choice not in DATASETS:
        print("Invalid choice")
        return
    
    dataset = DATASETS[choice]
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / dataset['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Processing: {dataset['name']}")
    print(f"{'='*70}")
    
    for file_info in dataset['files']:
        compressed_file = output_dir / file_info['filename']
        
        # Check if file already exists
        if compressed_file.exists():
            size_mb = compressed_file.stat().st_size / (1024 * 1024)
            print(f"\n✓ File already exists: {compressed_file.name} ({size_mb:.1f} MB)")
            
            response = input("  Re-download? (y/n): ").strip().lower()
            if response == 'y':
                compressed_file.unlink()
            else:
                print("  Using existing file")
        
        # Download if needed
        if not compressed_file.exists():
            if not download_from_gdrive(file_info['gdrive_id'], compressed_file):
                print(f"❌ Failed to download {file_info['filename']}")
                return
        
        # Extract
        extract_zip(compressed_file, file_info['extract_to'], base_dir)
    
    print(f"{'='*70}")
    print(f"✅ {dataset['name']} ready!")
    print(f"{'='*70}\n")


def main():
    """Main function."""
    # Check gdown
    if not check_gdown_installed():
        print("\n❌ Cannot proceed without gdown")
        print("Please install: pip install gdown")
        return
    
    while True:
        show_menu()
        choice = input("\nSelect dataset (0-5): ").strip()
        
        if choice == '0':
            print("\nExiting...")
            break
        
        if choice in DATASETS:
            process_dataset(choice)
            
            response = input("\nDownload another dataset? (y/n): ").strip().lower()
            if response != 'y':
                break
        else:
            print("\n❌ Invalid choice. Please try again.")


if __name__ == '__main__':
    main()
