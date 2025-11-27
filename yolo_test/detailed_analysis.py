"""
Detailed per-image analysis module for YOLO model testing.

This module handles:
1. Per-image prediction collection using model.predict()
2. Matching predictions to ground truth using IoU
3. Attribute-based failure analysis (weather, scene, time, size, etc.)
4. Generation of detailed analysis charts and CSV reports
"""

import time
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from tqdm import tqdm
import yaml


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Confidence threshold for prediction collection (standard for evaluation/mAP)
PREDICTION_CONF_THRESHOLD = 0.001

# Maximum detections per image
MAX_DETECTIONS_PER_IMAGE = 300

# Object count bucket ranges
OBJECT_COUNT_BUCKETS = ["1-5", "6-10", "11-20", "21-50", "50+"]
OBJECT_COUNT_THRESHOLDS = [5, 10, 20, 50]  # Boundaries for bucketing

# Object size bucket definitions (based on bbox area as % of image area)
SIZE_BUCKET_SMALL_THRESHOLD = 0.01   # < 1% of image = small
SIZE_BUCKET_MEDIUM_THRESHOLD = 0.05  # 1-5% of image = medium, >5% = large
SIZE_BUCKETS = ["small", "medium", "large"]

# Progress logging interval (log every N images during collection)
PROGRESS_LOG_INTERVAL_DIVISOR = 10  # Log 10 times throughout processing

# Output subfolder name for performance analysis results
PERFORMANCE_ANALYSIS_FOLDER = "performance_analysis"

# =============================================================================


def collect_per_image_predictions(
    model: Any,
    data_yaml_path: Path,
    used_split: str,
    device: str,
    iou_threshold: float,
    labels_dir: Path,
) -> Tuple[List[Dict], float]:
    """
    Collect per-image predictions for detailed analysis.
    
    Args:
        model: YOLO model instance
        data_yaml_path: Path to data.yaml configuration
        used_split: Dataset split ('train', 'val', 'test')
        device: Device to run on ('cuda', 'cpu', etc.)
        iou_threshold: IoU threshold for NMS
        labels_dir: Directory containing label files
    
    Returns:
        Tuple of (per_image_records, total_time)
        - per_image_records: List of dicts with predictions and ground truth per image
        - total_time: Total time taken for collection
    """
    print("\n" + "=" * 80)
    print("COLLECTING PER-IMAGE PREDICTIONS FOR DETAILED ANALYSIS")
    print("=" * 80)
    
    # Find images directory
    images_dir = None
    try:
        with open(data_yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
        images_dir = Path(data_cfg.get('path', '.')) / 'images' / used_split
    except Exception:
        images_dir = None

    # Collect image files
    if images_dir and images_dir.exists():
        image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    else:
        image_files = []
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir}. Check data_yaml_path and split.")
    
    print(f"✓ Found {len(image_files)} images to evaluate")
    print(f"✓ Model device: {device}")
    print(f"✓ IoU threshold: {iou_threshold}")
    print(f"✓ Collecting per-image details for attribute-based analysis...")

    predict_start_time = time.time()

    # Initialize accumulators
    per_image_records = []
    total_infer_time = 0.0

    num_classes = len(model.names)
    print(f"✓ Tracking {num_classes} classes")
    print(f"  Model classes: {list(model.names.values())}")
    
    # Log every N images to show progress
    log_interval = max(1, len(image_files) // PROGRESS_LOG_INTERVAL_DIVISOR)
    
    print(f"\n{'='*80}")
    print("PROCESSING IMAGES")
    print(f"{'='*80}")

    for img_idx, img_path in enumerate(tqdm(image_files, desc="Collecting image details", unit="img"), start=1):
        t0 = time.time()
        # IMPORTANT: Use same parameters as official validation for consistency
        # conf threshold is from PREDICTION_CONF_THRESHOLD constant (standard for evaluation/mAP)
        # This ensures we capture all predictions just like model.val() does
        results = model.predict(
            str(img_path), 
            device=device, 
            verbose=False,
            conf=PREDICTION_CONF_THRESHOLD,  # Use constant for consistency
            iou=iou_threshold,  # MUST match validation IoU threshold for NMS
            max_det=MAX_DETECTIONS_PER_IMAGE,  # Use constant
            save=False,
            show=False,
        )
        infer_time = time.time() - t0
        total_infer_time += infer_time

        # Take first result
        res = results[0]
        
        # Log progress periodically
        if img_idx % log_interval == 0:
            avg_time = total_infer_time / img_idx
            remaining = (len(image_files) - img_idx) * avg_time
            print(f"  Progress: {img_idx}/{len(image_files)} images | "
                  f"Avg: {avg_time*1000:.1f}ms/img | "
                  f"ETA: {remaining:.1f}s")

        # Extract predictions
        if not hasattr(res, 'boxes') or res.boxes is None:
            preds = []
        else:
            preds = []
            for b in res.boxes:
                try:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    conf = float(b.conf[0].cpu().numpy())
                    cls = int(b.cls[0].cpu().numpy())
                except Exception:
                    # Fallback for non-tensor boxes
                    xyxy = b.xyxy[0].tolist()
                    conf = float(b.conf[0])
                    cls = int(b.cls[0])
                preds.append({'xyxy': xyxy, 'conf': conf, 'cls': cls})

        # Load ground truth boxes from labels directory
        label_path = labels_dir / f"{img_path.stem}.txt"
        gts = []
        if label_path.exists():
            h_w = PILImage.open(img_path).size  # (w,h)
            w_img, h_img = h_w
            with open(label_path, 'r') as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center, y_center, bw, bh = map(float, parts[1:5])
                        x1 = (x_center - bw / 2) * w_img
                        y1 = (y_center - bh / 2) * h_img
                        x2 = (x_center + bw / 2) * w_img
                        y2 = (y_center + bh / 2) * h_img
                        gts.append({'xyxy': [x1, y1, x2, y2], 'cls': cls})

        # Store per-image record for analysis
        per_image_records.append({
            'image': str(img_path),
            'n_preds': len(preds),
            'n_gts': len(gts),
            'preds': preds,
            'gts': gts,
            'infer_time': infer_time,
        })

    predict_end_time = time.time()
    predict_total_time = predict_end_time - predict_start_time

    print(f"\n{'='*80}")
    print("PER-IMAGE COLLECTION COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Total images processed: {len(image_files)}")
    print(f"✓ Collection time: {predict_total_time:.2f}s")
    print(f"✓ Average time per image: {predict_total_time/len(image_files):.3f}s")
    print(f"{'='*80}\n")

    return per_image_records, predict_total_time


def analyze_failures_by_attributes(
    per_image_records: List[Dict],
    image_attributes: Dict[str, Dict],
    class_names: Dict[int, str],
    class_name_to_id: Dict[str, int],
    iou_threshold: float,
    test_run_dir: Path,
    train_metadata_data: Dict = None,
) -> Tuple[Dict, Dict, List[Path]]:
    """
    Analyze model performance by image attributes.
    
    Args:
        per_image_records: List of per-image prediction/GT records
        image_attributes: Dict mapping image basename to attributes (weather, scene, etc.)
        class_names: Dict mapping class ID to class name
        class_name_to_id: Dict mapping class name to class ID
        iou_threshold: IoU threshold for matching
        test_run_dir: Directory to save analysis results
        train_metadata_data: Optional training metadata for exposure analysis
    
    Returns:
        Tuple of (summary_dict, dataframes_dict, chart_paths)
        - summary_dict: Overall statistics and counts
        - dataframes_dict: Dict of pandas DataFrames with detailed breakdowns
        - chart_paths: List of paths to generated chart images
    """
    print("\n" + "=" * 80)
    print("ANALYZING FAILURES BY ATTRIBUTES")
    print("=" * 80)
    
    # Create performance analysis subfolder
    analysis_dir = test_run_dir / PERFORMANCE_ANALYSIS_FOLDER
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Analysis results will be saved to: {analysis_dir}")
    
    # Load training metadata if provided
    train_class_counts = {}
    if train_metadata_data:
        train_class_counts = train_metadata_data.get("class_counts", {})
        print(f"✓ Training metadata loaded with {len(train_class_counts)} classes")

    # Helper IoU function
    def iou_xyxy(a, b):
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        xi1 = max(xa1, xb1)
        yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2)
        yi2 = min(ya2, yb2)
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h
        area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    # Initialize accumulators
    per_image_summary_rows = []
    
    # Aggregate by attributes
    attr_aggregates = {
        "weather": {},
        "scene": {},
        "timeofday": {}
    }
    
    # Per-class accuracy within each attribute
    class_attr_stats = {
        "weather": {},
        "scene": {},
        "timeofday": {}
    }
    
    # Object count buckets (using constants)
    count_buckets = OBJECT_COUNT_BUCKETS
    count_bucket_stats = {bucket: {"expected": 0, "matched": 0, "images": 0} for bucket in count_buckets}
    
    # Object size buckets (using constants)
    size_buckets = SIZE_BUCKETS
    size_bucket_stats = {bucket: {"expected": 0, "matched": 0} for bucket in size_buckets}
    
    # Per-class stats for each size bucket
    class_size_stats = {}
    for cid in class_names.keys():
        class_size_stats[cid] = {bucket: {"expected": 0, "matched": 0} for bucket in size_buckets}
    
    def get_count_bucket(count: int) -> str:
        """Categorize by object count using constant thresholds"""
        thresholds = OBJECT_COUNT_THRESHOLDS
        if count <= thresholds[0]:
            return OBJECT_COUNT_BUCKETS[0]
        elif count <= thresholds[1]:
            return OBJECT_COUNT_BUCKETS[1]
        elif count <= thresholds[2]:
            return OBJECT_COUNT_BUCKETS[2]
        elif count <= thresholds[3]:
            return OBJECT_COUNT_BUCKETS[3]
        else:
            return OBJECT_COUNT_BUCKETS[4]
    
    def get_size_bucket(area_ratio: float) -> str:
        """Categorize object by bbox area ratio using constant thresholds"""
        if area_ratio < SIZE_BUCKET_SMALL_THRESHOLD:
            return "small"
        elif area_ratio < SIZE_BUCKET_MEDIUM_THRESHOLD:
            return "medium"
        else:
            return "large"

    # Process each image
    print(f"Processing {len(per_image_records)} images for attribute analysis...")
    for rec in tqdm(per_image_records, desc="Analyzing", unit="img"):
        img_path = Path(rec["image"])
        basename = img_path.stem
        attrs = image_attributes.get(basename, {}) if image_attributes else {}

        weather = attrs.get("weather", "unknown")
        scene = attrs.get("scene", "unknown")
        timeofday = attrs.get("timeofday", "unknown")
        
        # New metadata structure uses 'class_counts' and 'object_count'
        expected_objects_map = attrs.get("class_counts", {}) if isinstance(attrs.get("class_counts", {}), dict) else {}
        expected_total = int(attrs.get("object_count", rec.get("n_gts", 0)))

        # Match predictions to ground truth
        preds = rec.get("preds", [])
        gts = rec.get("gts", [])

        matched_gt = set()
        matched_per_class = {}
        
        # Get image dimensions for size calculation
        try:
            img = PILImage.open(img_path)
            img_width, img_height = img.size
            img_area = img_width * img_height
        except Exception:
            img_area = 1280 * 720  # fallback
        
        # Match predictions to ground truth and track sizes
        for pi, p in enumerate(preds):
            best_iou = 0.0
            best_gi = None
            for gi, g in enumerate(gts):
                if gi in matched_gt:
                    continue
                if p["cls"] != g["cls"]:
                    continue
                iou_val = iou_xyxy(p["xyxy"], g["xyxy"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gi = gi
            if best_gi is not None and best_iou >= iou_threshold:
                matched_gt.add(best_gi)
                cls_id = p["cls"]
                matched_per_class[cls_id] = matched_per_class.get(cls_id, 0) + 1
                
                # Calculate size for matched object
                gt_box = gts[best_gi]["xyxy"]
                box_area = max(0, gt_box[2] - gt_box[0]) * max(0, gt_box[3] - gt_box[1])
                area_ratio = box_area / img_area if img_area > 0 else 0
                size_bucket = get_size_bucket(area_ratio)
                size_bucket_stats[size_bucket]["matched"] += 1
                class_size_stats[cls_id][size_bucket]["matched"] += 1
        
        # Track all ground truth objects by size (including unmatched)
        for gi, g in enumerate(gts):
            gt_box = g["xyxy"]
            box_area = max(0, gt_box[2] - gt_box[0]) * max(0, gt_box[3] - gt_box[1])
            area_ratio = box_area / img_area if img_area > 0 else 0
            size_bucket = get_size_bucket(area_ratio)
            size_bucket_stats[size_bucket]["expected"] += 1
            cls_id = g["cls"]
            class_size_stats[cls_id][size_bucket]["expected"] += 1

        matched_count = len(matched_gt)
        accuracy = (matched_count / expected_total) if expected_total > 0 else None

        # Store per-image data
        per_image_summary_rows.append({
            "image": basename,
            "weather": weather,
            "scene": scene,
            "timeofday": timeofday,
            "expected_total": expected_total,
            "matched": matched_count,
            "missed": expected_total - matched_count,
            "accuracy": accuracy,
        })

        # Aggregate by attributes
        for attr_key, attr_val in [("weather", weather), ("scene", scene), ("timeofday", timeofday)]:
            if attr_val not in attr_aggregates[attr_key]:
                attr_aggregates[attr_key][attr_val] = {"expected": 0, "matched": 0, "images": 0}
            attr_aggregates[attr_key][attr_val]["expected"] += expected_total
            attr_aggregates[attr_key][attr_val]["matched"] += matched_count
            attr_aggregates[attr_key][attr_val]["images"] += 1
            
            # Per-class stats within this attribute
            if attr_val not in class_attr_stats[attr_key]:
                class_attr_stats[attr_key][attr_val] = {}
            
            for cname, cnt in expected_objects_map.items():
                cid = class_name_to_id.get(cname)
                if cid is None:
                    continue
                if cid not in class_attr_stats[attr_key][attr_val]:
                    class_attr_stats[attr_key][attr_val][cid] = {"expected": 0, "matched": 0}
                class_attr_stats[attr_key][attr_val][cid]["expected"] += int(cnt)
                class_attr_stats[attr_key][attr_val][cid]["matched"] += matched_per_class.get(cid, 0)

        # Aggregate by object count bucket
        bucket = get_count_bucket(expected_total)
        count_bucket_stats[bucket]["expected"] += expected_total
        count_bucket_stats[bucket]["matched"] += matched_count
        count_bucket_stats[bucket]["images"] += 1

    # Create DataFrames
    df_images = pd.DataFrame(per_image_summary_rows)
    
    # Attribute aggregates
    attr_summary_dfs = {}
    for attr_key, attr_vals in attr_aggregates.items():
        rows = []
        for val, stats in attr_vals.items():
            accuracy = (stats["matched"] / stats["expected"]) if stats["expected"] > 0 else None
            rows.append({
                attr_key: val,
                "images": stats["images"],
                "expected": stats["expected"],
                "matched": stats["matched"],
                "missed": stats["expected"] - stats["matched"],
                "accuracy": accuracy
            })
        df_temp = pd.DataFrame(rows) if rows else pd.DataFrame()
        # Filter out rows with None accuracy before sorting
        if not df_temp.empty:
            df_temp = df_temp[df_temp["accuracy"].notna()]
        attr_summary_dfs[attr_key] = df_temp.sort_values("accuracy") if not df_temp.empty else pd.DataFrame()

    # Object count bucket summary
    count_rows = []
    for bucket, stats in count_bucket_stats.items():
        accuracy = (stats["matched"] / stats["expected"]) if stats["expected"] > 0 else None
        count_rows.append({
            "object_count_range": bucket,
            "images": stats["images"],
            "expected": stats["expected"],
            "matched": stats["matched"],
            "missed": stats["expected"] - stats["matched"],
            "accuracy": accuracy
        })
    df_count_buckets = pd.DataFrame(count_rows)
    
    # Object size bucket summary (overall)
    size_rows = []
    for bucket in size_buckets:
        stats = size_bucket_stats[bucket]
        accuracy = (stats["matched"] / stats["expected"]) if stats["expected"] > 0 else None
        size_rows.append({
            "size_bucket": bucket,
            "expected": stats["expected"],
            "matched": stats["matched"],
            "missed": stats["expected"] - stats["matched"],
            "accuracy": accuracy
        })
    df_size_buckets = pd.DataFrame(size_rows)
    
    # Per-class accuracy by size
    class_size_rows = []
    for cid, cname in class_names.items():
        for bucket in size_buckets:
            stats = class_size_stats[cid][bucket]
            if stats["expected"] > 0:
                accuracy = stats["matched"] / stats["expected"]
                class_size_rows.append({
                    "class_id": cid,
                    "class_name": cname,
                    "size_bucket": bucket,
                    "expected": stats["expected"],
                    "matched": stats["matched"],
                    "accuracy": accuracy
                })
    df_class_size = pd.DataFrame(class_size_rows)
    
    # Per-class accuracy by weather/scene/timeofday
    class_weather_rows = []
    for weather_val, class_stats in class_attr_stats["weather"].items():
        for cid, stats in class_stats.items():
            if stats["expected"] > 0:
                class_weather_rows.append({
                    "weather": weather_val,
                    "class_id": cid,
                    "class_name": class_names[cid],
                    "expected": stats["expected"],
                    "matched": stats["matched"],
                    "accuracy": stats["matched"] / stats["expected"]
                })
    df_class_weather = pd.DataFrame(class_weather_rows)
    
    class_scene_rows = []
    for scene_val, class_stats in class_attr_stats["scene"].items():
        for cid, stats in class_stats.items():
            if stats["expected"] > 0:
                class_scene_rows.append({
                    "scene": scene_val,
                    "class_id": cid,
                    "class_name": class_names[cid],
                    "expected": stats["expected"],
                    "matched": stats["matched"],
                    "accuracy": stats["matched"] / stats["expected"]
                })
    df_class_scene = pd.DataFrame(class_scene_rows)
    
    class_time_rows = []
    for time_val, class_stats in class_attr_stats["timeofday"].items():
        for cid, stats in class_stats.items():
            if stats["expected"] > 0:
                class_time_rows.append({
                    "timeofday": time_val,
                    "class_id": cid,
                    "class_name": class_names[cid],
                    "expected": stats["expected"],
                    "matched": stats["matched"],
                    "accuracy": stats["matched"] / stats["expected"]
                })
    df_class_time = pd.DataFrame(class_time_rows)

    # Compile all dataframes
    dataframes_dict = {
        "per_image": df_images,
        "weather": attr_summary_dfs.get("weather", pd.DataFrame()),
        "scene": attr_summary_dfs.get("scene", pd.DataFrame()),
        "timeofday": attr_summary_dfs.get("timeofday", pd.DataFrame()),
        "object_count": df_count_buckets,
        "object_size": df_size_buckets,
        "class_size": df_class_size,
        "class_weather": df_class_weather,
        "class_scene": df_class_scene,
        "class_time": df_class_time,
    }

    # Create summary statistics
    total_images = len(per_image_summary_rows)
    total_expected = sum(row["expected_total"] for row in per_image_summary_rows)
    total_matched = sum(row["matched"] for row in per_image_summary_rows)
    overall_accuracy = (total_matched / total_expected) if total_expected > 0 else 0.0

    summary_dict = {
        "total_images": total_images,
        "total_expected_objects": total_expected,
        "total_matched_objects": total_matched,
        "total_missed_objects": total_expected - total_matched,
        "overall_accuracy": overall_accuracy,
        "weather_breakdown": len(attr_aggregates["weather"]),
        "scene_breakdown": len(attr_aggregates["scene"]),
        "timeofday_breakdown": len(attr_aggregates["timeofday"]),
    }

    print(f"\n✓ Analysis complete:")
    print(f"  - Total images: {total_images}")
    print(f"  - Total objects: {total_expected}")
    print(f"  - Matched objects: {total_matched}")
    print(f"  - Overall accuracy: {overall_accuracy:.2%}")
    
    # Save all DataFrames to CSV in the analysis directory
    print(f"\n✓ Saving analysis results to CSV files...")
    df_images.to_csv(analysis_dir / "per_image_accuracy.csv", index=False)
    
    for attr_key in ["weather", "scene", "timeofday"]:
        df = dataframes_dict.get(attr_key, pd.DataFrame())
        if not df.empty:
            df.to_csv(analysis_dir / f"accuracy_by_{attr_key}.csv", index=False)
    
    df_count_buckets.to_csv(analysis_dir / "accuracy_by_object_count.csv", index=False)
    df_size_buckets.to_csv(analysis_dir / "accuracy_by_size.csv", index=False)
    
    if not df_class_size.empty:
        df_class_size.to_csv(analysis_dir / "accuracy_by_class_and_size.csv", index=False)
    if not df_class_weather.empty:
        df_class_weather.to_csv(analysis_dir / "accuracy_by_class_and_weather.csv", index=False)
    if not df_class_scene.empty:
        df_class_scene.to_csv(analysis_dir / "accuracy_by_class_and_scene.csv", index=False)
    if not df_class_time.empty:
        df_class_time.to_csv(analysis_dir / "accuracy_by_class_and_timeofday.csv", index=False)
    
    print(f"✓ Saved {len([f for f in analysis_dir.glob('*.csv')])} CSV files to {analysis_dir}")
    
    # Note: Chart generation will be handled by the main script
    # as it requires matplotlib imports and plotting functions
    chart_paths = []

    return summary_dict, dataframes_dict, chart_paths
