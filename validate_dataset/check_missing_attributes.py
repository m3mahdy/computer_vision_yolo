#!/usr/bin/env python3
"""
Check if any BDD100K label files are missing attribute fields
"""

import json
from pathlib import Path


def check_missing_attributes(base_dir: Path):
    """Check for missing attributes in BDD100K labels."""
    
    labels_dir = base_dir / 'bdd100k_tmp_labels' / '100k'
    
    print("=" * 80)
    print("CHECKING FOR MISSING ATTRIBUTES IN BDD100K LABELS")
    print("=" * 80)
    
    for split in ['train', 'val', 'test']:
        split_dir = labels_dir / split
        
        if not split_dir.exists():
            continue
        
        print(f"\n{split.upper()} Split:")
        json_files = list(split_dir.glob('*.json'))
        print(f"  Total files: {len(json_files):,}")
        
        missing_attributes = []
        missing_weather = []
        missing_scene = []
        missing_timeofday = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check if attributes field exists
                if 'attributes' not in data:
                    missing_attributes.append(json_file.name)
                    continue
                
                attrs = data['attributes']
                
                # Check individual fields
                if 'weather' not in attrs:
                    missing_weather.append(json_file.name)
                
                if 'scene' not in attrs:
                    missing_scene.append(json_file.name)
                
                if 'timeofday' not in attrs:
                    missing_timeofday.append(json_file.name)
                    
            except Exception as e:
                print(f"  Error reading {json_file.name}: {e}")
        
        # Report results
        if missing_attributes:
            print(f"  ❌ Files missing 'attributes' field: {len(missing_attributes)}")
            if len(missing_attributes) <= 5:
                for f in missing_attributes:
                    print(f"     - {f}")
        else:
            print(f"  ✓ All files have 'attributes' field")
        
        if missing_weather:
            print(f"  ❌ Files missing 'weather': {len(missing_weather)}")
            if len(missing_weather) <= 5:
                for f in missing_weather:
                    print(f"     - {f}")
        else:
            print(f"  ✓ All files have 'weather' attribute")
        
        if missing_scene:
            print(f"  ❌ Files missing 'scene': {len(missing_scene)}")
            if len(missing_scene) <= 5:
                for f in missing_scene:
                    print(f"     - {f}")
        else:
            print(f"  ✓ All files have 'scene' attribute")
        
        if missing_timeofday:
            print(f"  ❌ Files missing 'timeofday': {len(missing_timeofday)}")
            if len(missing_timeofday) <= 5:
                for f in missing_timeofday:
                    print(f"     - {f}")
        else:
            print(f"  ✓ All files have 'timeofday' attribute")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if not missing_attributes and not missing_weather and not missing_scene and not missing_timeofday:
        print("\n✓ ALL files contain complete attribute information!")
        print("  - Every file has 'attributes' field")
        print("  - Every file has weather, scene, and timeofday")
        print("\n⚠️  DEFAULT VALUES ARE NOT NEEDED!")
        print("  The code can safely assume all attributes exist.")
    else:
        print("\n❌ Some files are missing attributes")
        print("  Default values ARE needed for robustness")


if __name__ == '__main__':
    base_dir = Path.cwd()
    check_missing_attributes(base_dir)
