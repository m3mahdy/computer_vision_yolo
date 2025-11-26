"""1. Download and extract main BDD100K dataset from Google Drive."""

import zipfile
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

GDRIVE_IDS = {
    'images': '1yHEpeEdRDAz5yH4pbo4o1SvzKzGKRaLS',
    'labels': '1Gh_5g-MAx1R5X3eNsTTdz_GPialECz0L'
}


def check_gdown_installed():
    try:
        import gdown
        return True
    except ImportError:
        print("\n📦 Installing gdown...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
            print("✓ Installed\n")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed. Install manually: pip install gdown")
            return False


def download_from_gdrive(file_id, output_path):
    try:
        import gdown
        print(f"\n📥 Downloading {output_path.name}...")
        gdown.download(f'https://drive.google.com/uc?id={file_id}', str(output_path), quiet=False)
        if output_path.exists():
            print(f"✓ Complete ({output_path.stat().st_size / (1024 * 1024):.1f} MB)\n")
            return True
        print("❌ Failed\n")
        return False
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


def extract_zip(zip_path, extract_to):
    print(f"📦 Extracting {zip_path.name}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for member in tqdm(zipf.namelist(), desc="  Extracting", unit='files'):
            zipf.extract(member, extract_to)
    print("✓ Done\n")


def main():
    print("="*70)
    print("DOWNLOAD & EXTRACT MAIN BDD100K DATASET")
    print("="*70)
    
    if not check_gdown_installed():
        return
    
    base_dir = Path(__file__).parent.parent
    source_dir = base_dir / 'bdd_100k_source'
    source_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (name, file_id) in enumerate([('IMAGES', GDRIVE_IDS['images']), ('LABELS', GDRIVE_IDS['labels'])], 1):
        print(f"\n[{idx}/2] {name}")
        zip_file = source_dir / f'bdd100k_{name.lower()}_100k.zip' if name == 'IMAGES' else source_dir / 'bdd100k_labels.zip'
        extract_dir = base_dir / f'bdd100k_tmp_{name.lower()}'
        
        if not zip_file.exists():
            if not download_from_gdrive(file_id, zip_file):
                print(f"❌ Failed to download {name.lower()}")
                return
        else:
            print(f"✓ File exists: {zip_file.name}")
        
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"✓ Already extracted: {extract_dir}")
        else:
            extract_zip(zip_file, extract_dir)
    
    print("\n" + "="*70)
    print("✅ READY - Next: Run 2_convert_labels_to_yolo.py")
    print("="*70)


if __name__ == '__main__':
    main()
