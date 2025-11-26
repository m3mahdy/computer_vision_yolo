#!/usr/bin/env python3
"""
Analyze BDD100K categories to distinguish detection vs segmentation classes.
"""

import json
from pathlib import Path
from collections import Counter


def analyze_categories(base_dir: Path):
    """Analyze all categories and their frequencies."""
    labels_dir = base_dir / 'bdd100k_tmp_labels' / '100k'
    
    # Collect all categories with counts
    category_counter = Counter()
    scene_counter = Counter()
    
    total_files = 0
    max_files = 10000  # Analyze more files
    
    for split in ['train', 'val', 'test']:
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue
        
        for json_file in split_dir.glob('*.json'):
            if total_files >= max_files:
                break
                
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Get scene
                attrs = data.get('attributes', {})
                if 'scene' in attrs:
                    scene_counter[attrs['scene']] += 1
                
                # Handle different BDD100K label formats
                frames = data.get('frames', [])
                if frames:
                    objects = frames[0].get('objects', [])
                else:
                    objects = data.get('objects', data.get('labels', []))
                
                for obj in objects:
                    category = obj.get('category', '')
                    if category:
                        category_counter[category] += 1
                
                total_files += 1
                
            except Exception as e:
                continue
        
        if total_files >= max_files:
            break
    
    print(f"Analyzed {total_files} label files\n")
    
    print("=" * 80)
    print("ALL CATEGORIES (sorted by frequency)")
    print("=" * 80)
    for category, count in category_counter.most_common():
        print(f"{count:8d}  {category}")
    
    print("\n" + "=" * 80)
    print("DETECTION CLASSES (object detection - no prefix)")
    print("=" * 80)
    detection_classes = []
    for category, count in category_counter.most_common():
        if '/' not in category:  # Detection classes have no prefix
            detection_classes.append(category)
            print(f"{count:8d}  {category}")
    
    print("\n" + "=" * 80)
    print("SEGMENTATION/LANE CLASSES (area/*, lane/*)")
    print("=" * 80)
    for category, count in category_counter.most_common():
        if '/' in category:  # Segmentation/lane classes have prefix
            print(f"{count:8d}  {category}")
    
    print("\n" + "=" * 80)
    print("SCENE VALUES (sorted by frequency)")
    print("=" * 80)
    for scene, count in scene_counter.most_common():
        print(f"{count:8d}  {scene}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique categories: {len(category_counter)}")
    print(f"Detection classes (no prefix): {len(detection_classes)}")
    print(f"Segmentation/lane classes (with prefix): {len(category_counter) - len(detection_classes)}")
    print(f"\nDetection classes list:")
    print(detection_classes)
    print(f"\nScene values found:")
    print(sorted(scene_counter.keys()))


if __name__ == '__main__':
    base_dir = Path.cwd()
    analyze_categories(base_dir)
