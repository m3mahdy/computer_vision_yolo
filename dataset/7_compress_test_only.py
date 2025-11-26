"""
7. Compress Test Split Only.

Compresses only the test split from the full dataset for quick distribution.
Creates a standalone test dataset with data.yaml configured for test only.

Source: bdd100k_yolo (full dataset)
Output: bdd100k_test_split_zipped/bdd100k_yolo_test_split.zip

Usage:
    python dataset/7_compress_test_only.py
"""

from pathlib import Path

from bdd100k_config import YOLO_DATASET_ROOT
from bdd100k_compress import compress_test_split_only


def main():
    """Main function."""
    base_dir = Path(__file__).parent.parent
    
    print(f"\n{'='*70}")
    print("COMPRESS TEST SPLIT ONLY")
    print(f"{'='*70}")
    print(f"Source: {YOLO_DATASET_ROOT}")
    print(f"Output: bdd100k_test_split_zipped/")
    
    # Check if full dataset exists
    if not YOLO_DATASET_ROOT.exists() or not (YOLO_DATASET_ROOT / 'data.yaml').exists():
        print(f"\n❌ ERROR: Full dataset not found: {YOLO_DATASET_ROOT}")
        print("Create full dataset first with script 2_convert_labels_to_yolo.py")
        return
    
    # Check if test split exists
    test_images_dir = YOLO_DATASET_ROOT / 'images' / 'test'
    test_labels_dir = YOLO_DATASET_ROOT / 'labels' / 'test'
    
    if not test_images_dir.exists() or not test_labels_dir.exists():
        print(f"\n❌ ERROR: Test split not found in full dataset")
        print(f"Expected:")
        print(f"  {test_images_dir}")
        print(f"  {test_labels_dir}")
        return
    
    # Compress
    result = compress_test_split_only(YOLO_DATASET_ROOT, base_dir)
    
    if result:
        print(f"\n{'='*70}")
        print(f"✅ TEST SPLIT COMPRESSED")
        print(f"{'='*70}")
        print(f"File: {result['path']}")
        print(f"Size: {result['size_mb']:.1f} MB")
        print(f"Images: {result['num_images']:,}")
        print(f"Labels: {result['num_labels']:,}")
        print(f"\nTo use:")
        print(f"  unzip bdd100k_yolo_test_split.zip")
        print(f"  cd bdd100k_yolo_test")
        print(f"  # Ready for YOLO validation/testing")
    else:
        print("\n❌ Compression failed")


if __name__ == '__main__':
    main()
