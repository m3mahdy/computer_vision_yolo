#!/usr/bin/env python3
"""
Analyze BDD100K Labels to Extract Actual Attribute Values

This script scans the actual BDD100K JSON label files to determine:
1. What attribute fields exist
2. What values each attribute can have
3. Distribution of these values across splits
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import sys


def analyze_bdd100k_labels(base_dir: Path):
    """Analyze BDD100K label files to extract all attribute values."""
    
    labels_dir = base_dir / 'bdd100k_tmp_labels' / '100k'
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("BDD100K ATTRIBUTE ANALYSIS")
    print("=" * 80)
    print(f"Scanning: {labels_dir}\n")
    
    # Track attributes across all splits
    all_attributes = defaultdict(lambda: defaultdict(Counter))
    split_stats = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = labels_dir / split
        
        if not split_dir.exists():
            print(f"⚠️  Split '{split}' not found, skipping...")
            continue
        
        print(f"Analyzing {split.upper()} split...")
        
        json_files = list(split_dir.glob('*.json'))
        print(f"  Found {len(json_files):,} JSON files")
        
        # Counters for this split
        weather_counter = Counter()
        scene_counter = Counter()
        timeofday_counter = Counter()
        
        # Track what fields exist
        sample_attributes = {}
        files_with_attributes = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check for attributes field
                if 'attributes' in data:
                    files_with_attributes += 1
                    attrs = data['attributes']
                    
                    # Store sample for inspection
                    if not sample_attributes:
                        sample_attributes = attrs
                    
                    # Count attribute values
                    if 'weather' in attrs:
                        weather_counter[attrs['weather']] += 1
                    
                    if 'scene' in attrs:
                        scene_counter[attrs['scene']] += 1
                    
                    if 'timeofday' in attrs:
                        timeofday_counter[attrs['timeofday']] += 1
                        
            except Exception as e:
                print(f"  ⚠️  Error reading {json_file.name}: {e}")
        
        # Store results for this split
        split_stats[split] = {
            'total_files': len(json_files),
            'files_with_attributes': files_with_attributes,
            'weather': dict(weather_counter),
            'scene': dict(scene_counter),
            'timeofday': dict(timeofday_counter),
            'sample_attributes': sample_attributes
        }
        
        # Aggregate to all splits
        for weather, count in weather_counter.items():
            all_attributes['weather'][split][weather] = count
        for scene, count in scene_counter.items():
            all_attributes['scene'][split][scene] = count
        for timeofday, count in timeofday_counter.items():
            all_attributes['timeofday'][split][timeofday] = count
        
        print(f"  ✓ {files_with_attributes:,} files have attributes\n")
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS: ATTRIBUTE VALUES FOUND")
    print("=" * 80)
    
    for split in ['train', 'val', 'test']:
        if split not in split_stats:
            continue
            
        stats = split_stats[split]
        print(f"\n{split.upper()} Split:")
        print(f"  Total files: {stats['total_files']:,}")
        print(f"  Files with attributes: {stats['files_with_attributes']:,}")
        
        if stats['sample_attributes']:
            print(f"\n  Sample attributes structure:")
            for key, value in stats['sample_attributes'].items():
                print(f"    - {key}: {value} (type: {type(value).__name__})")
    
    # Display attribute value distributions
    print("\n" + "=" * 80)
    print("WEATHER ATTRIBUTE VALUES")
    print("=" * 80)
    
    all_weather_values = set()
    for split_data in all_attributes['weather'].values():
        all_weather_values.update(split_data.keys())
    
    if all_weather_values:
        print(f"\nFound {len(all_weather_values)} unique weather values:")
        print(f"  {sorted(all_weather_values)}")
        
        print(f"\n{'Value':<20} | {'Train':>12} {'Val':>12} {'Test':>12} | {'Total':>12}")
        print("-" * 80)
        
        for weather in sorted(all_weather_values):
            train_count = all_attributes['weather']['train'].get(weather, 0)
            val_count = all_attributes['weather']['val'].get(weather, 0)
            test_count = all_attributes['weather']['test'].get(weather, 0)
            total = train_count + val_count + test_count
            print(f"{weather:<20} | {train_count:>12,} {val_count:>12,} {test_count:>12,} | {total:>12,}")
    else:
        print("  No weather values found!")
    
    print("\n" + "=" * 80)
    print("SCENE ATTRIBUTE VALUES")
    print("=" * 80)
    
    all_scene_values = set()
    for split_data in all_attributes['scene'].values():
        all_scene_values.update(split_data.keys())
    
    if all_scene_values:
        print(f"\nFound {len(all_scene_values)} unique scene values:")
        print(f"  {sorted(all_scene_values)}")
        
        print(f"\n{'Value':<20} | {'Train':>12} {'Val':>12} {'Test':>12} | {'Total':>12}")
        print("-" * 80)
        
        for scene in sorted(all_scene_values):
            train_count = all_attributes['scene']['train'].get(scene, 0)
            val_count = all_attributes['scene']['val'].get(scene, 0)
            test_count = all_attributes['scene']['test'].get(scene, 0)
            total = train_count + val_count + test_count
            print(f"{scene:<20} | {train_count:>12,} {val_count:>12,} {test_count:>12,} | {total:>12,}")
    else:
        print("  No scene values found!")
    
    print("\n" + "=" * 80)
    print("TIME OF DAY ATTRIBUTE VALUES")
    print("=" * 80)
    
    all_timeofday_values = set()
    for split_data in all_attributes['timeofday'].values():
        all_timeofday_values.update(split_data.keys())
    
    if all_timeofday_values:
        print(f"\nFound {len(all_timeofday_values)} unique timeofday values:")
        print(f"  {sorted(all_timeofday_values)}")
        
        print(f"\n{'Value':<20} | {'Train':>12} {'Val':>12} {'Test':>12} | {'Total':>12}")
        print("-" * 80)
        
        for timeofday in sorted(all_timeofday_values):
            train_count = all_attributes['timeofday']['train'].get(timeofday, 0)
            val_count = all_attributes['timeofday']['val'].get(timeofday, 0)
            test_count = all_attributes['timeofday']['test'].get(timeofday, 0)
            total = train_count + val_count + test_count
            print(f"{timeofday:<20} | {train_count:>12,} {val_count:>12,} {test_count:>12,} | {total:>12,}")
    else:
        print("  No timeofday values found!")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nUnique attribute values found:")
    print(f"  Weather: {len(all_weather_values)} values")
    print(f"  Scene: {len(all_scene_values)} values")
    print(f"  Time of Day: {len(all_timeofday_values)} values")
    
    # Check for 'unknown' or None values
    print(f"\nChecking for 'unknown' or null values:")
    for attr_name, attr_data in all_attributes.items():
        unknown_count = 0
        none_count = 0
        for split_data in attr_data.values():
            unknown_count += split_data.get('unknown', 0)
            none_count += split_data.get(None, 0) + split_data.get('', 0)
        
        if unknown_count > 0 or none_count > 0:
            print(f"  {attr_name}:")
            if unknown_count > 0:
                print(f"    - 'unknown': {unknown_count:,} occurrences")
            if none_count > 0:
                print(f"    - null/empty: {none_count:,} occurrences")
    
    print("\n" + "=" * 80)
    print("✓ Analysis complete")
    print("=" * 80)
    
    return all_attributes, split_stats


if __name__ == '__main__':
    base_dir = Path.cwd()
    
    print(f"Base directory: {base_dir}\n")
    
    all_attributes, split_stats = analyze_bdd100k_labels(base_dir)
    
    print("\n" + "=" * 80)
    print("RECOMMENDED ATTRIBUTE LOOKUPS FOR CODE")
    print("=" * 80)
    
    print("\nBased on the analysis, update your code with these values:\n")
    
    all_weather = set()
    all_scene = set()
    all_timeofday = set()
    
    for split_data in all_attributes['weather'].values():
        all_weather.update(split_data.keys())
    for split_data in all_attributes['scene'].values():
        all_scene.update(split_data.keys())
    for split_data in all_attributes['timeofday'].values():
        all_timeofday.update(split_data.keys())
    
    print("WEATHER_VALUES = [")
    for w in sorted(all_weather):
        print(f"    '{w}',")
    print("]\n")
    
    print("SCENE_VALUES = [")
    for s in sorted(all_scene):
        print(f"    '{s}',")
    print("]\n")
    
    print("TIMEOFDAY_VALUES = [")
    for t in sorted(all_timeofday):
        print(f"    '{t}',")
    print("]")
