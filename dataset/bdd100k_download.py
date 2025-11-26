"""
BDD100K Dataset Download Module.

This module handles downloading BDD100K dataset files from Google Drive mirrors
or prompting for manual downloads.
"""

import subprocess
import sys
from pathlib import Path

from bdd100k_config import BDD100K_URLS, BDD100K_GDOWN_IDS


def check_gdown_installed():
    """
    Check if gdown is installed, install if not.
    Returns True if gdown is available, False otherwise.
    """
    try:
        import gdown
        return True
    except ImportError:
        print("\n📦 Installing gdown for faster downloads...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            print("✓ gdown installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install gdown. Please install manually: pip install gdown")
            return False


def download_file(gdown_id, output_path):
    """
    Download a file from Google Drive using gdown.
    Much faster than direct download from BDD100K servers.
    
    Args:
        gdown_id: Google Drive file ID
        output_path: Path object where file should be saved
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={gdown_id}'
        gdown.download(url, str(output_path), quiet=False)
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def check_and_download_datasets(source_dir):
    """
    Check if dataset files exist, prompt to download if missing.
    
    Args:
        source_dir: Path object for directory where dataset files should be stored
        
    Returns:
        True if files exist or were downloaded successfully, False otherwise
    """
    images_zip = source_dir / "bdd100k_images_100k.zip"
    labels_zip = source_dir / "bdd100k_labels.zip"
    
    files_exist = images_zip.exists() and labels_zip.exists()
    
    if files_exist:
        print("\n✓ Dataset files found:")
        print(f"  Images: {images_zip} ({images_zip.stat().st_size / (1024**3):.2f} GB)")
        print(f"  Labels: {labels_zip} ({labels_zip.stat().st_size / (1024**2):.2f} MB)")
        return True
    
    # Prompt user to download
    print("\n" + "="*70)
    print("DATASET FILES NOT FOUND")
    print("="*70)
    print("\nMissing files:")
    if not images_zip.exists():
        print(f"  ❌ {images_zip.name} (~6GB)")
    if not labels_zip.exists():
        print(f"  ❌ {labels_zip.name} (~300MB)")
    
    print("\nOptions:")
    print("  1. Download automatically using gdown (faster, recommended)")
    print(f"  2. Download manually from: {BDD100K_URLS['website']}")
    print("  3. Exit and provide files later")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        # Check/install gdown
        if not check_gdown_installed():
            print("\nCannot proceed without gdown. Please install it or download manually.")
            return False
        
        print("\n📥 Downloading from Google Drive mirror (faster than official servers)...")
        print("   Original URLs available at: http://bdd-data.berkeley.edu/")
        source_dir.mkdir(parents=True, exist_ok=True)
        
        # Download using gdown
        success = True
        try:
            if not images_zip.exists():
                print(f"\n[1/2] Downloading images (~6GB, this may take a while)...")
                if not download_file(BDD100K_GDOWN_IDS['images'], images_zip):
                    success = False
            
            if not labels_zip.exists():
                print(f"\n[2/2] Downloading labels (~300MB)...")
                if not download_file(BDD100K_GDOWN_IDS['labels'], labels_zip):
                    success = False
            
            if success:
                print("\n✓ Downloads complete!")
                return True
            else:
                print("\n❌ Some downloads failed.")
                print(f"\nPlease download manually from: {BDD100K_URLS['website']}")
                return False
        
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            print(f"\nPlease download manually from: {BDD100K_URLS['website']}")
            return False
    
    elif choice == '2':
        print("\nPlease download files manually and place them in:")
        print(f"  {source_dir}/")
        print("\nThen run this script again.")
        return False
    
    else:
        print("\nExiting. Run script again when files are available.")
        return False
