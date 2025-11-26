import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, List

import cv2
import numpy as np
import torch
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from tqdm import tqdm

import shutil
import time
import glob

from ultralytics import YOLO
import wandb

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from PIL import Image as PILImage


def setup_environment(use_wandb: bool) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    if use_wandb:
        print("✓ W&B logging enabled")
    else:
        print("✓ W&B logging disabled")
    return device


def load_data_config(data_yaml_path: Path, yolo_dataset_root: Path) -> Dict[str, Any]:
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_yaml_path}\n\n"
            f"Please run the dataset preparation script first:\n"
            f"  python3 process_bdd100k_to_yolo_dataset.py\n"
        )

    with open(data_yaml_path, "r") as f:
        data_config = yaml.safe_load(f)

    data_config["path"] = str(yolo_dataset_root)

    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    return data_config


def generate_class_colors(class_names: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
    rng = np.random.default_rng(42)
    colors_map: Dict[int, Tuple[int, int, int]] = {}
    for class_id in class_names.keys():
        color = tuple(int(channel) for channel in rng.integers(40, 255, size=3))
        colors_map[class_id] = color
    return colors_map


def build_attribute_text(attributes: Dict[str, Any]) -> str:
    """Build a one-line attribute summary like in the notebook.

    Expected keys in attributes JSON (from representative metadata):
    - weather
    - scene
    - timeofday
    """
    if not attributes:
        return "Attributes: N/A"

    weather = attributes.get("weather", "unknown")
    scene = attributes.get("scene", "unknown")
    time_of_day = attributes.get("timeofday", "unknown")

    return f"Attributes: weather={weather}, scene={scene}, time={time_of_day}"


def draw_ground_truth(
    img_path: Path,
    label_path: Path,
    class_names: Dict[int, str],
    colors: Dict[int, Tuple[int, int, int]],
) -> Tuple[np.ndarray, int]:
    """Draw ground-truth boxes using deterministic colors."""
    import cv2
    
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h, w = img_bgr.shape[:2]
    object_count = 0

    if label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    object_count += 1
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    color = tuple(int(c) for c in colors.get(class_id, (255, 255, 255)))
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
                    label = class_names.get(class_id, f"class_{class_id}")
                    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(
                        img_bgr,
                        (x1, max(0, y1 - label_h - baseline - 6)),
                        (x1 + label_w + 8, y1),
                        color,
                        -1,
                    )
                    text_color = (0, 0, 0) if sum(color) > 500 else (255, 255, 255)
                    cv2.putText(img_bgr, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), object_count


def draw_predictions_with_consistent_colors(
    result: Any,
    colors: Dict[int, Tuple[int, int, int]],
    class_names: Dict[int, str],
) -> np.ndarray:
    """Draw model predictions using same palette as ground truth."""
    import cv2
    
    img_bgr = result.orig_img.copy()
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        color = tuple(int(c) for c in colors.get(class_id, (255, 255, 255)))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
        label = f"{class_names.get(class_id, f'class_{class_id}')} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            img_bgr,
            (x1, max(0, y1 - label_h - baseline - 6)),
            (x1 + label_w + 8, y1),
            color,
            -1,
        )
        text_color = (0, 0, 0) if sum(color) > 500 else (255, 255, 255)
        cv2.putText(img_bgr, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def generate_sample_comparisons(
    model: YOLO,
    valid_images: List[Path],
    labels_dir: Path,
    class_names: Dict[int, str],
    test_run_dir: Path,
    num_samples: int = 6,
    device: str = "cpu",
    image_attributes: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Generate high-resolution comparison images.

    Returns a list of dictionaries containing:
    - comparison_image_path: Path to the generated comparison image
    - original_image_path: Path to the original source image
    - attributes: Dictionary with weather, scene, timeofday
    - gt_count: Number of ground truth objects
    - pred_count: Number of predicted objects
    """
    import random
    import cv2
    from tqdm import tqdm

    comparisons_dir = test_run_dir / "sample_comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    colors = generate_class_colors(class_names)
    num_comparisons = min(num_samples, len(valid_images))

    if num_comparisons == 0:
        print("\u26a0\ufe0f  No labeled images available for comparison generation.")
        return []

    sample_images = (
        random.sample(valid_images, num_comparisons)
        if len(valid_images) > num_comparisons
        else valid_images
    )
    print(f"\nGenerating {len(sample_images)} high-resolution comparison figures with attributes...")

    comparison_data: List[Dict[str, Any]] = []

    for idx, img_path in enumerate(tqdm(sample_images, desc="Generating comparisons"), 1):
        label_path = labels_dir / f"{img_path.stem}.txt"

        # Run inference
        result = model(str(img_path), verbose=False, device=device)[0]

        # Draw ground truth and predictions
        gt_img, gt_count = draw_ground_truth(img_path, label_path, class_names, colors)
        pred_img = draw_predictions_with_consistent_colors(result, colors, class_names)

        pred_count = len(result.boxes)

        # Create side-by-side comparison with higher resolution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14), dpi=300)

        ax1.imshow(gt_img)
        ax1.set_title(
            f"Ground Truth ({gt_count} objects)",
            fontweight="bold",
            fontsize=16,
        )
        
        ax1.axis('off')

        ax2.imshow(pred_img)
        ax2.set_title(
            f"Prediction ({pred_count} objects)",
            fontweight="bold",
            fontsize=16,
        )

        ax2.axis('off')
        
        # fig.suptitle(
        #     f"Comparison #{idx}: {img_path.name}",
        #     fontsize=18,
        #     fontweight="bold",
        # )
        plt.tight_layout()

        comparison_path = comparisons_dir / f"comparison_{idx:02d}_{img_path.stem}.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Extract attributes from metadata if available
        attributes = {}
        if image_attributes:
            img_basename = img_path.stem
            img_meta = image_attributes.get(img_basename, {})
            attributes = {
                "weather": img_meta.get("weather", "unknown"),
                "scene": img_meta.get("scene", "unknown"),
                "timeofday": img_meta.get("timeofday", "unknown"),
            }

        comparison_data.append({
            "comparison_image_path": comparison_path,
            "original_image_path": img_path,
            "attributes": attributes,
            "gt_count": gt_count,
            "pred_count": pred_count,
        })

    print(f"\u2713 Generated {len(comparison_data)} comparison images")
    print(f"  Saved to: {comparisons_dir}")

    return comparison_data


def visualize_predictions(
    model: YOLO,
    image_paths: List[Path],
    class_names: Dict[int, str],
    conf_threshold: float = 0.25,
    figsize: Tuple[int, int] = (20, 10),
) -> plt.Figure:
    """
    Visualize predictions on sample images.
    
    Args:
        model: Loaded YOLO model
        image_paths: List of image paths to visualize
        class_names: Dictionary mapping class IDs to names
        conf_threshold: Confidence threshold for predictions
        figsize: Figure size
        
    Returns:
        Matplotlib figure with predictions
    """
    num_images = len(image_paths)
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten() if num_images > 1 else axes
    
    colors = generate_class_colors(class_names)
    
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                color = colors.get(cls, (255, 0, 0))
                cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                label = f"{class_names.get(cls, f'class_{cls}')}: {conf:.2f}"
                cv2.putText(img_rgb, label, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        ax = axes[idx] if num_images > 1 else axes
        ax.imshow(img_rgb)
        ax.set_title(f"{img_path.name}", fontsize=10)
        ax.axis('off')
    
    for idx in range(num_images, len(axes) if isinstance(axes, np.ndarray) else 1):
        if num_images > 1:
            axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def load_model(model_name: str, models_dir: Path) -> Tuple[YOLO, Dict[str, float]]:
    
    
    model_path = models_dir / f"{model_name}.pt"
    
    
    if not model_path.exists():
        print(f'Model not found at {model_path}')
        print(f'Downloading {model_name} ...')
        
        try:
            # Download model - ensure .pt extension for ultralytics
            # Ultralytics expects model names with .pt extension for download
            if not model_name.endswith('.pt'):
                model_name_for_download = model_name + '.pt'
            else:
                model_name_for_download = model_name
                
            print(f'  Requesting model: {model_name_for_download}')
            model = YOLO(model_name_for_download)
            
            # Create models directory
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model to our directory using export/save
            try:
                # Try to save using the model's save method
                if hasattr(model, 'save'):
                    model.save(str(model_path))
                    print(f'✓ Model downloaded and saved to {model_path}')
                    print(f'  Size: {model_path.stat().st_size / (1024*1024):.1f} MB')
                else:
                    # Fallback: copy from cache
                    cache_patterns = [
                        str(Path.home() / '.cache' / 'ultralytics' / '**' / f'{model_name}.pt'),
                        str(Path.home() / '.config' / 'Ultralytics' / '**' / f'{model_name}.pt'),
                    ]
                    
                    model_found = False
                    for pattern in cache_patterns:
                        cache_paths = glob.glob(pattern, recursive=True)
                        if cache_paths:
                            shutil.copy(cache_paths[0], model_path)
                            print(f'✓ Model downloaded and saved to {model_path}')
                            print(f'  Size: {model_path.stat().st_size / (1024*1024):.1f} MB')
                            model_found = True
                            break
                    
                    if not model_found:
                        print(f'✓ Model loaded from ultralytics cache')
                        print(f'  Note: Model is in cache, not copied to {model_path}')
                        print(f'  This is normal and the model will work correctly')
            except Exception as save_error:
                print(f'⚠️  Could not save model to custom location: {save_error}')
                print(f'✓ Model loaded successfully from ultralytics cache')
                
        except Exception as e:
            print(f'\n❌ Error downloading model: {e}')
            raise
    else:
        model = YOLO(str(model_path))
        print(f'✓ Model loaded from {model_path}')

    
    
    

    model = YOLO(str(model_path))
    info_values = model.info()
    keys = ["layers", "params", "size(MB)", "FLOPs(G)"]
    model_info: Dict[str, float] = {}
    for key, value in zip(keys, info_values):
        model_info[key] = value
        
        
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    model_info["size(MB)"] = model_size_mb


    print("\n📊 Model Information:")
    print(f"  Model: {model_name}")
    print(f"  Classes in model: {len(model.names)}")
    print(f"  Task: {model.task}")
    print(f"  Parameters: {model_info.get('params', 0) / 1e6:.1f}M")
    print(f"  Model Size: {model_info.get('size(MB)', 0):.1f} MB")
    print(f"  FLOPs (640x640): {model_info.get('FLOPs(G)', 0):.2f} GFLOPs")
    print(f"  Model Size: {model_info['size(MB)']:.1f} MB")

    return model, model_info


def load_dataset(used_dataset_root: Path, used_split: str, data_config: Dict[str, Any]) -> Dict[str, Any]:
    images_dir = used_dataset_root / "images" / used_split
    labels_dir = used_dataset_root / "labels" / used_split

    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    label_files = sorted(
        [labels_dir / f"{img.stem}.txt" for img in image_files if (labels_dir / f"{img.stem}.txt").exists()]
    )
    valid_images = [img for img in image_files if (labels_dir / f"{img.stem}.txt").exists()]

    print("✓ Dataset loaded")
    print(f"  Total images: {len(image_files)}")
    print(f"  Images with labels: {len(valid_images)}")
    print(f"  Label files: {len(label_files)}")

    metadata_dir = used_dataset_root / "representative_json"
    performance_file = metadata_dir / f"{used_split}_performance_analysis.json"

    if performance_file.exists():
        with open(performance_file, "r") as f:
            performance_data = json.load(f)
        print(f"\n✓ Performance metadata loaded: {performance_file.name}")
        print(f"  Images with attributes: {performance_data['total_images']}")

        # Map using basename so we can look up attributes by image stem
        # (e.g., image file d0518d52-0188b977.jpg -> basename key "d0518d52-0188b977").
        image_attributes = {img["basename"]: img for img in performance_data["images"]}
    else:
        print(f"\n⚠️ Performance metadata not found: {performance_file}")
        performance_data = None
        image_attributes = {}

    num_classes = data_config["nc"]
    class_names = {i: name for i, name in enumerate(data_config["names"])}
    class_name_to_id = {name: i for i, name in enumerate(data_config["names"])}

    return {
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "image_files": image_files,
        "valid_images": valid_images,
        "metadata_dir": metadata_dir,
        "performance_data": performance_data,
        "image_attributes": image_attributes,
        "num_classes": num_classes,
        "class_names": class_names,
        "class_name_to_id": class_name_to_id,
    }


def run_yolo_validation(
    model: YOLO,
    data_yaml_path: Path,
    used_split: str,
    device: str,
    iou_threshold: float,
    test_run_dir: Path,
    batch_size: int = 16,
) -> Tuple[Any, float]:
    """
    Run per-image prediction using `model.predict` and compute simple detection metrics
    (TP/FP/FN per class) by IoU matching. This replaces the coarse `model.val` call
    so we can attach per-image metadata and produce richer failure analysis.

    Returns a lightweight object similar to ultralytics' validation result containing
    basic timing and box-level aggregated metrics, plus the total_time.
    """
    print("\nRunning per-image YOLO evaluation (model.predict)...")

    images_dir = None
    # try to infer images dir from data_yaml
    try:
        with open(data_yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
        images_dir = Path(data_cfg.get('path', '.')) / 'images' / used_split
    except Exception:
        images_dir = None

    # collect image files
    if images_dir and images_dir.exists():
        image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    else:
        # fallback: try to find images under current working dir
        image_files = []

    start_time = time.time()

    # Helper functions
    def iou_xyxy(box_a, box_b):
        # boxes are [x1,y1,x2,y2]
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
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
    per_image_records = []
    num_images = len(image_files)
    total_infer_time = 0.0

    # prepare confusion matrix counts
    num_classes = len(model.names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    class_tp = {i: 0 for i in range(num_classes)}
    class_fp = {i: 0 for i in range(num_classes)}
    class_fn = {i: 0 for i in range(num_classes)}

    for img_path in tqdm(image_files, desc="Evaluating images", unit="img"):
        t0 = time.time()
        results = model.predict(str(img_path), device=device, verbose=False)
        infer_time = time.time() - t0
        total_infer_time += infer_time

        # take first result
        res = results[0]

        # predicted boxes: list of [x1,y1,x2,y2], cls, conf
        preds = []
        for b in res.boxes:
            try:
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                cls = int(b.cls[0].cpu().numpy())
            except Exception:
                # fallback for non-tensor boxes
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                cls = int(b.cls[0])
            preds.append({'xyxy': xyxy, 'conf': conf, 'cls': cls})

        # load GT boxes
        label_path = img_path.with_suffix('.txt')
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

        # match preds to gts by IoU and class
        matched_gt = set()
        matched_pred = set()

        for pi, p in enumerate(preds):
            best_iou = 0.0
            best_gi = None
            for gi, g in enumerate(gts):
                if gi in matched_gt:
                    continue
                if p['cls'] != g['cls']:
                    continue
                iou_val = iou_xyxy(p['xyxy'], g['xyxy'])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gi = gi

            if best_gi is not None and best_iou >= iou_threshold:
                # true positive
                class_tp[p['cls']] += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                # mark confusion: correctly predicted class
                confusion[gts[best_gi]['cls'], p['cls']] += 1
            else:
                # false positive (either no gt or class mismatch)
                class_fp[p['cls']] += 1
                # if there is a closest gt but different class, count confusion
                best_iou_any = 0.0
                best_gi_any = None
                for gi, g in enumerate(gts):
                    if gi in matched_gt:
                        continue
                    iou_val = iou_xyxy(p['xyxy'], g['xyxy'])
                    if iou_val > best_iou_any:
                        best_iou_any = iou_val
                        best_gi_any = gi
                if best_gi_any is not None and best_iou_any >= iou_threshold:
                    # predicted wrong class for an existing GT
                    confusion[gts[best_gi_any]['cls'], p['cls']] += 1

        # any unmatched gts are false negatives
        for gi, g in enumerate(gts):
            if gi not in matched_gt:
                class_fn[g['cls']] += 1

        per_image_records.append({
            'image': str(img_path),
            'n_preds': len(preds),
            'n_gts': len(gts),
            'preds': preds,
            'gts': gts,
            'infer_time': infer_time,
        })

    end_time = time.time()
    total_time = end_time - start_time

    # compute per-class metrics
    yolo_class_metrics = {}
    p_list = []
    r_list = []
    for cls_idx in range(num_classes):
        tp = class_tp.get(cls_idx, 0)
        fp = class_fp.get(cls_idx, 0)
        fn = class_fn.get(cls_idx, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ap50 = precision  # approximate AP@0.5 with precision (not exact)
        yolo_class_metrics[cls_idx] = {'precision': precision, 'recall': recall, 'ap50': ap50}
        p_list.append(precision)
        r_list.append(recall)

    # aggregate metrics
    mean_precision = float(np.mean(p_list)) if p_list else 0.0
    mean_recall = float(np.mean(r_list)) if r_list else 0.0
    map50 = float(np.mean([v['ap50'] for v in yolo_class_metrics.values()])) if yolo_class_metrics else 0.0

    # prepare lightweight result object
    class BoxStats:
        pass

    box = BoxStats()
    box.mp = mean_precision
    box.mr = mean_recall
    box.map50 = map50
    box.map = map50
    box.ap_class_index = list(yolo_class_metrics.keys())
    box.p = [yolo_class_metrics[i]['precision'] for i in sorted(yolo_class_metrics.keys())]
    box.r = [yolo_class_metrics[i]['recall'] for i in sorted(yolo_class_metrics.keys())]
    box.ap50 = [yolo_class_metrics[i]['ap50'] for i in sorted(yolo_class_metrics.keys())]
    box.ap = box.ap50

    speed = {'preprocess': 0.0, 'inference': total_infer_time / max(1, num_images) * 1000.0, 'loss': 0.0, 'postprocess': 0.0}

    class ValRes:
        pass

    results_obj = ValRes()
    results_obj.speed = speed
    results_obj.box = box
    results_obj.fitness = (box.mp + box.mr + box.map50) / 3.0
    results_obj.confusion_matrix = type('C', (), {'matrix': confusion})
    # attach per-image records for deeper analysis
    results_obj.per_image = per_image_records

    print(f"\n✓ Per-image evaluation completed: {len(per_image_records)} images, total_time={total_time:.2f}s")

    return results_obj, total_time


def extract_core_metrics(
    validation_results: Any,
    images_dir: Path,
    num_classes: int,
    class_names: Dict[int, str],
    model_info: Dict[str, float],
    total_time: float,
) -> Dict[str, Any]:
    num_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))

    # Prefer execution time reported by YOLO when available to avoid
    # re-calculating timing from external code.

    
    preprocess = float(validation_results.speed.get("preprocess", 0.0)) 
    inference = float(validation_results.speed.get("inference", 0.0)) 
    loss = float(validation_results.speed.get("loss", 0.0)) 
    postprocess = float(validation_results.speed.get("postprocess", 0.0)) 

    avg_inference_time = inference+postprocess+preprocess
    
    fps = 1000.0 / avg_inference_time

    yolo_metrics = {
        "precision": float(validation_results.box.mp),
        "recall": float(validation_results.box.mr),
        "map50": float(validation_results.box.map50),
        "map50_95": float(validation_results.box.map),
        "fitness": float(validation_results.fitness),
    }

    yolo_class_metrics: Dict[str, Dict[str, float]] = {}
    class_tp: Dict[int, int] = {}
    class_fp: Dict[int, int] = {}
    class_fn: Dict[int, int] = {}

    if hasattr(validation_results.box, "ap_class_index") and len(validation_results.box.ap_class_index) > 0:
        for i, class_idx in enumerate(validation_results.box.ap_class_index):
            idx = int(class_idx)
            name = class_names.get(idx, f"class_{idx}")
            precision = float(validation_results.box.p[i]) if i < len(validation_results.box.p) else 0.0
            recall = float(validation_results.box.r[i]) if i < len(validation_results.box.r) else 0.0
            ap50 = float(validation_results.box.ap50[i]) if i < len(validation_results.box.ap50) else 0.0
            ap50_95 = float(validation_results.box.ap[i]) if i < len(validation_results.box.ap) else 0.0
            yolo_class_metrics[name] = {
                "precision": precision,
                "recall": recall,
                "ap50": ap50,
                "ap50_95": ap50_95,
            }
            class_tp[idx] = 0
            class_fp[idx] = 0
            class_fn[idx] = 0

    # Build confusion-matrix-derived TP/FP/FN using the YOLO confusion matrix
    # when available, but store both the original matrix and aggregated counts
    # so that downstream visualizations use this script's definition.
    if hasattr(validation_results, "confusion_matrix") and hasattr(validation_results.confusion_matrix, "matrix"):
        confusion_matrix_raw = validation_results.confusion_matrix.matrix
    else:
        confusion_matrix_raw = np.zeros((num_classes, num_classes), dtype=int)

    confusion_matrix = np.array(confusion_matrix_raw, copy=True)

    for i in range(num_classes):
        tp_val = 0
        fp_val = 0
        fn_val = 0
        if i < confusion_matrix.shape[0] and i < confusion_matrix.shape[1]:
            tp_val = int(confusion_matrix[i, i])
        if i < confusion_matrix.shape[1]:
            fp_val = int(confusion_matrix[:, i].sum() - confusion_matrix[i, i])
        if i < confusion_matrix.shape[0]:
            fn_val = int(confusion_matrix[i, :].sum() - confusion_matrix[i, i])
        class_tp[i] = tp_val
        class_fp[i] = fp_val
        class_fn[i] = fn_val

    print("\n" + "=" * 80)
    print("OFFICIAL YOLO VALIDATION RESULTS")
    print("=" * 80)
    print(f"Precision (mean): {yolo_metrics['precision']:.4f}")
    print(f"Recall (mean):    {yolo_metrics['recall']:.4f}")
    print(f"mAP@0.5:          {yolo_metrics['map50']:.4f}")
    print(f"mAP@0.5:0.95:     {yolo_metrics['map50_95']:.4f}")
    print(f"Fitness:          {yolo_metrics['fitness']:.4f}")
    print("\n⚡ Performance Metrics:")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Average Inference Time: {avg_inference_time * 1000:.2f} ms per image")
    print(f"  FPS (Frames Per Second): {fps:.2f}")
    print("=" * 80)

    metrics = {
        "num_images": num_images,
        "avg_inference_time": avg_inference_time,
        "fps": fps,
        "yolo_metrics": yolo_metrics,
        "yolo_class_metrics": yolo_class_metrics,
        "class_tp": class_tp,
        "class_fp": class_fp,
        "class_fn": class_fn,
        "confusion_matrix": confusion_matrix,
    }
    return metrics


def build_per_class_dataframe(
    num_classes: int,
    class_names: Dict[int, str],
    class_tp: Dict[int, int],
    class_fp: Dict[int, int],
    class_fn: Dict[int, int],
    yolo_class_metrics: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, float, float, float, int, int, int]:
    metrics_data: List[Dict[str, Any]] = []
    for class_id in sorted(class_names.keys()):
        tp_val = class_tp.get(class_id, 0)
        fp_val = class_fp.get(class_id, 0)
        fn_val = class_fn.get(class_id, 0)
        precision = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
        recall = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        map50 = yolo_class_metrics.get(class_names[class_id], {}).get("ap50", 0.0)
        metrics_data.append(
            {
                "Class": class_names[class_id],
                "TP": tp_val,
                "FP": fp_val,
                "FN": fn_val,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "mAP@0.5": map50,
            }
        )

    df_metrics = pd.DataFrame(metrics_data)

    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0.0
    )

    return df_metrics, overall_precision, overall_recall, overall_f1, total_tp, total_fp, total_fn


def plot_core_and_map_metrics(
    df_metrics: pd.DataFrame,
    total_tp: int,
    total_fp: int,
    total_fn: int,
    overall_precision: float,
    overall_recall: float,
    overall_f1: float,
    yolo_metrics: Dict[str, float],
    test_run_dir: Path,
) -> Dict[str, Path]:
    """Generate individual core and mAP visualizations for maximum clarity.

    Returns a dict of figure names to image paths so the PDF builder can
    insert each diagram on its own, one by one.
    """
    sns.set_style("whitegrid")

    fig_paths: Dict[str, Path] = {}

    # Precision by class
    precision_sorted = df_metrics.sort_values("Precision")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    bars = ax.barh(precision_sorted["Class"], precision_sorted["Precision"], color="#5BC0EB")
    ax.set_title("Precision by Class", fontweight="bold", fontsize=22)
    ax.set_xlabel("Precision", fontweight="bold", fontsize=18)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)
    # Add value labels
    for idx, (bar, value) in enumerate(zip(bars, precision_sorted["Precision"])):
        ax.text(value + 0.02, idx, f"{value:.3f}", va="center", fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig_paths["precision_by_class"] = test_run_dir / "precision_by_class.png"
    plt.savefig(fig_paths["precision_by_class"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Recall by class
    recall_sorted = df_metrics.sort_values("Recall")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    bars = ax.barh(recall_sorted["Class"], recall_sorted["Recall"], color="#F25F5C")
    ax.set_title("Recall by Class", fontweight="bold", fontsize=22)
    ax.set_xlabel("Recall", fontweight="bold", fontsize=18)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)
    # Add value labels
    for idx, (bar, value) in enumerate(zip(bars, recall_sorted["Recall"])):
        ax.text(value + 0.02, idx, f"{value:.3f}", va="center", fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig_paths["recall_by_class"] = test_run_dir / "recall_by_class.png"
    plt.savefig(fig_paths["recall_by_class"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    # F1-Score by class
    f1_sorted = df_metrics.sort_values("F1-Score")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    bars = ax.barh(f1_sorted["Class"], f1_sorted["F1-Score"], color="#9BC53D")
    ax.set_title("F1-Score by Class", fontweight="bold", fontsize=22)
    ax.set_xlabel("F1-Score", fontweight="bold", fontsize=18)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)
    # Add value labels
    for idx, (bar, value) in enumerate(zip(bars, f1_sorted["F1-Score"])):
        ax.text(value + 0.02, idx, f"{value:.3f}", va="center", fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig_paths["f1_by_class"] = test_run_dir / "f1_by_class.png"
    plt.savefig(fig_paths["f1_by_class"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Overall detection outcomes
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax.bar(["TP", "FP", "FN"], [total_tp, total_fp, total_fn], color=["#177E89", "#ED6A5A", "#F4A259"])
    ax.set_title("Overall Detection Outcomes", fontweight="bold", fontsize=22)
    ax.set_ylabel("Count", fontweight="bold", fontsize=18)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(total_tp, total_fp, total_fn) * 0.01,
            f"{int(height)}",
            ha="center",
            fontweight="bold",
            fontsize=14,
        )
    plt.tight_layout()
    fig_paths["detection_outcomes"] = test_run_dir / "detection_outcomes.png"
    plt.savefig(fig_paths["detection_outcomes"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    # mAP@0.5 by class
    map_sorted = df_metrics.sort_values("mAP@0.5")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    bars = ax.barh(map_sorted["Class"], map_sorted["mAP@0.5"], color="#B388EB")
    ax.set_title("mAP@0.5 by Class", fontweight="bold", fontsize=22)
    ax.set_xlabel("mAP@0.5", fontweight="bold", fontsize=18)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)
    # Add value labels
    for idx, (bar, value) in enumerate(zip(bars, map_sorted["mAP@0.5"])):
        ax.text(value + 0.02, idx, f"{value:.3f}", va="center", fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig_paths["map50_by_class"] = test_run_dir / "map50_by_class.png"
    plt.savefig(fig_paths["map50_by_class"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Overall metrics bar chart
    overall_plot_values = {
        "Precision": overall_precision,
        "Recall": overall_recall,
        "F1-Score": overall_f1,
        "mAP@0.5": yolo_metrics["map50"],
        "mAP@0.5:0.95": yolo_metrics["map50_95"],
    }
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax.bar(overall_plot_values.keys(), overall_plot_values.values(), color="#FFA630")
    ax.set_ylim(0, 1)
    ax.set_title("Overall Metrics", fontweight="bold", fontsize=22)
    ax.set_ylabel("Score", fontweight="bold", fontsize=18)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)
    for idx, (bar, value) in enumerate(zip(bars, overall_plot_values.values())):
        ax.text(
            idx,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
            fontsize=14,
        )
    plt.tight_layout()
    fig_paths["overall_metrics"] = test_run_dir / "overall_metrics.png"
    plt.savefig(fig_paths["overall_metrics"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fig_paths


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    num_classes: int,
    class_names: Dict[int, str],
    model_name: str,
    test_run_dir: Path,
) -> Path:
    """Plot confusion matrix using the exact style from yolo_test notebook.

    Rows (i): true classes
    Columns (j): predicted classes
    confusion_matrix[i, j]: count of true class i predicted as class j
    """
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Draw each cell manually with solid colors
    for i in range(num_classes):
        for j in range(num_classes):
            value = confusion_matrix[i, j]
            
            # Determine cell color
            if value == 0:
                # White for empty cells
                cell_color = 'white'
            elif i == j:
                cell_color = '#00A676'  # Correct predictions (green)
            else:
                cell_color = '#D7263D'  # Misclassifications (red)
            
            rect = Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=cell_color,
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(rect)
            
            # Add text annotations with smaller font
            if value > 0:
                text_color = 'white' if i == j else '#F7F7F7'
                ax.text(
                    j, i, str(int(value)),
                    ha='center', va='center',
                    color=text_color,
                    fontsize=9,
                    fontweight='bold'
                )

    # Set axis limits and properties
    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_ylim(num_classes - 0.5, -0.5)
    ax.set_aspect('equal')

    # Set ticks and labels with smaller font
    class_labels = [class_names[i] for i in range(num_classes)]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_labels, fontsize=8, fontweight='bold', rotation=45, ha='right')
    ax.set_yticklabels(class_labels, fontsize=8, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontweight='bold', fontsize=11)
    ax.set_ylabel('True Class', fontweight='bold', fontsize=11)
    ax.set_title(f'Confusion Matrix ({model_name} validation)', fontweight='bold', fontsize=13)
    ax.grid(False)

    # Center the confusion matrix in the figure
    plt.tight_layout()

    confusion_matrix_path = test_run_dir / "confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print("(Green = Correct Predictions, Red = Incorrect Predictions, White = No Predictions)")
    
    return confusion_matrix_path


def generate_failure_analysis(
    per_image_records: List[Dict[str, Any]],
    image_attributes: Dict[str, Any],
    class_names: Dict[int, str],
    test_run_dir: Path,
    dataset_root: Path,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Comprehensive analysis of prediction accuracy in relation to:
    - Image attributes (weather, scene, timeofday)
    - Object counts (total_objects, objects_per_class)
    
    Generates:
    - CSV files with detailed per-image and aggregated data
    - Charts showing accuracy by attribute values
    - Charts showing accuracy by object count ranges
    - Per-class accuracy breakdowns within each attribute
    """
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE FAILURE ANALYSIS")
    print("Analyzing relationship between attributes and prediction accuracy...")
    print("=" * 80)

    # build reverse map from class name -> id
    class_name_to_id = {name: cid for cid, name in class_names.items()}
    
    # Initialize train-test comparison DataFrame (will be populated later)
    df_train_test = pd.DataFrame()
    
    # Load training split metadata to analyze training exposure
    train_class_counts = {}
    train_total_images = 0
    train_metadata_path = dataset_root / "representative_json" / "train_performance_analysis.json"
    
    if train_metadata_path.exists():
        print("\nLoading training split metadata for exposure analysis...")
        try:
            with open(train_metadata_path, "r") as f:
                train_data = json.load(f)
            train_total_images = train_data.get("total_images", 0)
            
            # Extract per-class counts from training data
            for img_data in train_data.get("images", []):
                objects_per_class = img_data.get("objects_per_class", {})
                for class_name, count in objects_per_class.items():
                    cid = class_name_to_id.get(class_name)
                    if cid is not None:
                        train_class_counts[cid] = train_class_counts.get(cid, 0) + int(count)
            
            print(f"✓ Training metadata loaded: {train_total_images} images, {sum(train_class_counts.values())} objects")
        except Exception as e:
            print(f"⚠️  Warning: Could not load training metadata: {e}")
            train_class_counts = {}
    else:
        print(f"⚠️  Training metadata not found: {train_metadata_path}")
        train_class_counts = {}

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

    # Accumulators
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
    
    # Object count buckets
    count_buckets = ["1-5", "6-10", "11-20", "21-50", "50+"]
    count_bucket_stats = {bucket: {"expected": 0, "matched": 0, "images": 0} for bucket in count_buckets}
    
    # Object size buckets (based on bbox area ratio)
    size_buckets = ["small", "medium", "large"]
    size_bucket_stats = {bucket: {"expected": 0, "matched": 0} for bucket in size_buckets}
    
    # Per-class stats for each size bucket
    class_size_stats = {}
    for cid in class_names.keys():
        class_size_stats[cid] = {bucket: {"expected": 0, "matched": 0} for bucket in size_buckets}
    
    def get_count_bucket(count: int) -> str:
        if count <= 5:
            return "1-5"
        elif count <= 10:
            return "6-10"
        elif count <= 20:
            return "11-20"
        elif count <= 50:
            return "21-50"
        else:
            return "50+"
    
    def get_size_bucket(area_ratio: float) -> str:
        """Categorize object by bbox area ratio to image size"""
        if area_ratio < 0.01:  # < 1% of image
            return "small"
        elif area_ratio < 0.05:  # 1-5% of image
            return "medium"
        else:  # > 5% of image
            return "large"

    # Process each image
    for rec in per_image_records:
        img_path = Path(rec["image"])
        basename = img_path.stem
        attrs = image_attributes.get(basename, {}) if image_attributes else {}

        weather = attrs.get("weather", "unknown")
        scene = attrs.get("scene", "unknown")
        timeofday = attrs.get("timeofday", "unknown")
        
        expected_objects_map = attrs.get("objects_per_class", {}) if isinstance(attrs.get("objects_per_class", {}), dict) else {}
        expected_total = int(attrs.get("total_objects", rec.get("n_gts", 0)))

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

    # Save CSVs
    df_images.to_csv(test_run_dir / "per_image_accuracy.csv", index=False)
    for attr_key, df in attr_summary_dfs.items():
        if not df.empty:
            df.to_csv(test_run_dir / f"accuracy_by_{attr_key}.csv", index=False)
    df_count_buckets.to_csv(test_run_dir / "accuracy_by_object_count.csv", index=False)
    df_size_buckets.to_csv(test_run_dir / "accuracy_by_size.csv", index=False)
    if not df_class_size.empty:
        df_class_size.to_csv(test_run_dir / "accuracy_by_class_and_size.csv", index=False)
    if not df_class_weather.empty:
        df_class_weather.to_csv(test_run_dir / "accuracy_by_class_and_weather.csv", index=False)
    if not df_class_scene.empty:
        df_class_scene.to_csv(test_run_dir / "accuracy_by_class_and_scene.csv", index=False)
    if not df_class_time.empty:
        df_class_time.to_csv(test_run_dir / "accuracy_by_class_and_timeofday.csv", index=False)
    if not df_train_test.empty:
        df_train_test.to_csv(test_run_dir / "train_test_class_comparison.csv", index=False)

    # Generate charts
    print("\nGenerating accuracy analysis charts...")
    charts = {}
    
    sns.set_style("whitegrid")
    
    # Chart 1: Accuracy by Weather
    if not attr_summary_dfs["weather"].empty:
        df_weather = attr_summary_dfs["weather"].sort_values("accuracy")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        bars = ax.barh(df_weather["weather"], df_weather["accuracy"], color="#5BC0EB")
        ax.set_xlabel("Accuracy", fontweight="bold", fontsize=14)
        ax.set_ylabel("Weather", fontweight="bold", fontsize=14)
        ax.set_title("Prediction Accuracy by Weather Condition", fontweight="bold", fontsize=16)
        ax.set_xlim(0, 1)
        for i, (idx, row) in enumerate(df_weather.iterrows()):
            ax.text(row["accuracy"] + 0.02, i, f'{row["accuracy"]:.2%} ({row["images"]} imgs)', 
                   va='center', fontsize=10)
        plt.tight_layout()
        chart_path = test_run_dir / "accuracy_by_weather.png"
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        charts["accuracy_by_weather"] = str(chart_path)

    # Chart 2: Accuracy by Scene
    if not attr_summary_dfs["scene"].empty:
        df_scene = attr_summary_dfs["scene"].sort_values("accuracy")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        bars = ax.barh(df_scene["scene"], df_scene["accuracy"], color="#F25F5C")
        ax.set_xlabel("Accuracy", fontweight="bold", fontsize=14)
        ax.set_ylabel("Scene", fontweight="bold", fontsize=14)
        ax.set_title("Prediction Accuracy by Scene Type", fontweight="bold", fontsize=16)
        ax.set_xlim(0, 1)
        for i, (idx, row) in enumerate(df_scene.iterrows()):
            ax.text(row["accuracy"] + 0.02, i, f'{row["accuracy"]:.2%} ({row["images"]} imgs)', 
                   va='center', fontsize=10)
        plt.tight_layout()
        chart_path = test_run_dir / "accuracy_by_scene.png"
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        charts["accuracy_by_scene"] = str(chart_path)

    # Chart 3: Accuracy by Time of Day
    if not attr_summary_dfs["timeofday"].empty:
        df_time = attr_summary_dfs["timeofday"].sort_values("accuracy")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        bars = ax.barh(df_time["timeofday"], df_time["accuracy"], color="#9BC53D")
        ax.set_xlabel("Accuracy", fontweight="bold", fontsize=14)
        ax.set_ylabel("Time of Day", fontweight="bold", fontsize=14)
        ax.set_title("Prediction Accuracy by Time of Day", fontweight="bold", fontsize=16)
        ax.set_xlim(0, 1)
        for i, (idx, row) in enumerate(df_time.iterrows()):
            ax.text(row["accuracy"] + 0.02, i, f'{row["accuracy"]:.2%} ({row["images"]} imgs)', 
                   va='center', fontsize=10)
        plt.tight_layout()
        chart_path = test_run_dir / "accuracy_by_timeofday.png"
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        charts["accuracy_by_timeofday"] = str(chart_path)

    # Chart 4: Accuracy by Object Count Range
    if not df_count_buckets.empty:
        # Filter out buckets with no data
        df_count_valid = df_count_buckets[df_count_buckets["accuracy"].notna()].copy()
        if not df_count_valid.empty:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            bars = ax.bar(df_count_valid["object_count_range"], df_count_valid["accuracy"], color="#FFA630")
            ax.set_xlabel("Objects per Image", fontweight="bold", fontsize=14)
            ax.set_ylabel("Accuracy", fontweight="bold", fontsize=14)
            ax.set_title("Prediction Accuracy by Object Count", fontweight="bold", fontsize=16)
            ax.set_ylim(0, 1)
            for i, row in df_count_valid.iterrows():
                ax.text(i, row["accuracy"] + 0.02, f'{row["accuracy"]:.2%}\n({row["images"]} imgs)', 
                       ha='center', fontsize=9)
            plt.tight_layout()
            chart_path = test_run_dir / "accuracy_by_object_count.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_by_object_count"] = str(chart_path)
    
    # Chart 5: Accuracy by Object Size (Scale/Distance)
    if not df_size_buckets.empty:
        # Filter out buckets with no data (accuracy is None)
        df_size_valid = df_size_buckets[df_size_buckets["accuracy"].notna()].copy()
        if not df_size_valid.empty:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            bars = ax.bar(df_size_valid["size_bucket"], df_size_valid["accuracy"], 
                         color=["#E63946", "#F4A261", "#2A9D8F"][:len(df_size_valid)])
            ax.set_xlabel("Object Size (bbox area relative to image)", fontweight="bold", fontsize=14)
            ax.set_ylabel("Accuracy", fontweight="bold", fontsize=14)
            ax.set_title("Prediction Accuracy by Object Size\n(small: <1%, medium: 1-5%, large: >5% of image)", 
                        fontweight="bold", fontsize=16)
            ax.set_ylim(0, 1)
            for i, row in df_size_valid.iterrows():
                ax.text(i, row["accuracy"] + 0.02, 
                       f'{row["accuracy"]:.2%}\n({row["expected"]} objs)', 
                       ha='center', fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            chart_path = test_run_dir / "accuracy_by_size.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_by_size"] = str(chart_path)
    
    # Chart 6: Per-Class Accuracy by Size (Heatmap)
    if not df_class_size.empty:
        pivot_data = df_class_size.pivot(index="class_name", columns="size_bucket", values="accuracy")
        pivot_data = pivot_data[["small", "medium", "large"]]  # Order columns
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
        sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap="RdYlGn", 
                   vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, ax=ax)
        ax.set_title("Per-Class Accuracy by Object Size", fontweight="bold", fontsize=16)
        ax.set_xlabel("Object Size", fontweight="bold", fontsize=14)
        ax.set_ylabel("Class", fontweight="bold", fontsize=14)
        plt.tight_layout()
        chart_path = test_run_dir / "accuracy_by_class_and_size.png"
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        charts["accuracy_by_class_and_size"] = str(chart_path)
    
    # Chart 7: Per-Class Accuracy by Weather (Top classes only for readability)
    if not df_class_weather.empty:
        # Get top 5 classes by total expected count
        top_classes = df_class_weather.groupby("class_name")["expected"].sum().nlargest(5).index.tolist()
        df_top = df_class_weather[df_class_weather["class_name"].isin(top_classes)]
        
        if not df_top.empty:
            pivot_data = df_top.pivot(index="class_name", columns="weather", values="accuracy")
            fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
            sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap="RdYlGn", 
                       vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, ax=ax)
            ax.set_title("Per-Class Accuracy by Weather (Top 5 Classes)", fontweight="bold", fontsize=16)
            ax.set_xlabel("Weather Condition", fontweight="bold", fontsize=14)
            ax.set_ylabel("Class", fontweight="bold", fontsize=14)
            plt.tight_layout()
            chart_path = test_run_dir / "accuracy_by_class_and_weather.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_by_class_and_weather"] = str(chart_path)
    
    # Chart 8: Per-Class Accuracy by Scene (Top classes only)
    if not df_class_scene.empty:
        top_classes = df_class_scene.groupby("class_name")["expected"].sum().nlargest(5).index.tolist()
        df_top = df_class_scene[df_class_scene["class_name"].isin(top_classes)]
        
        if not df_top.empty:
            pivot_data = df_top.pivot(index="class_name", columns="scene", values="accuracy")
            fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
            sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap="RdYlGn", 
                       vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, ax=ax)
            ax.set_title("Per-Class Accuracy by Scene (Top 5 Classes)", fontweight="bold", fontsize=16)
            ax.set_xlabel("Scene Type", fontweight="bold", fontsize=14)
            ax.set_ylabel("Class", fontweight="bold", fontsize=14)
            plt.tight_layout()
            chart_path = test_run_dir / "accuracy_by_class_and_scene.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_by_class_and_scene"] = str(chart_path)
    
    # Chart 9: Per-Class Accuracy by Time of Day (Top classes only)
    if not df_class_time.empty:
        top_classes = df_class_time.groupby("class_name")["expected"].sum().nlargest(5).index.tolist()
        df_top = df_class_time[df_class_time["class_name"].isin(top_classes)]
        
        if not df_top.empty:
            pivot_data = df_top.pivot(index="class_name", columns="timeofday", values="accuracy")
            fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
            sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap="RdYlGn", 
                       vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, ax=ax)
            ax.set_title("Per-Class Accuracy by Time of Day (Top 5 Classes)", fontweight="bold", fontsize=16)
            ax.set_xlabel("Time of Day", fontweight="bold", fontsize=14)
            ax.set_ylabel("Class", fontweight="bold", fontsize=14)
            plt.tight_layout()
            chart_path = test_run_dir / "accuracy_by_class_and_timeofday.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_by_class_and_timeofday"] = str(chart_path)
    
    # Chart 10: Train vs Test Class Distribution
    if not df_train_test.empty and train_class_counts:
        df_sorted = df_train_test.sort_values("train_count", ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
        x = np.arange(len(df_sorted))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_sorted["train_count"], width, label="Train Split", color="#3498db")
        bars2 = ax.bar(x + width/2, df_sorted["test_count"], width, label="Test Split", color="#e74c3c")
        
        ax.set_xlabel("Class", fontweight="bold", fontsize=14)
        ax.set_ylabel("Object Count", fontweight="bold", fontsize=14)
        ax.set_title("Class Distribution: Training vs Testing Split", fontweight="bold", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted["class_name"], rotation=45, ha="right")
        ax.legend(fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        chart_path = test_run_dir / "train_test_distribution.png"
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        charts["train_test_distribution"] = str(chart_path)
    
    # Chart 11: Test Accuracy vs Training Exposure (Scatter)
    if not df_train_test.empty and train_class_counts:
        df_valid = df_train_test[df_train_test["test_accuracy"].notna()].copy()
        
        if not df_valid.empty:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
            
            # Scatter plot
            scatter = ax.scatter(df_valid["train_count"], df_valid["test_accuracy"], 
                               s=df_valid["test_count"]*3, alpha=0.6, c=df_valid["test_accuracy"],
                               cmap="RdYlGn", vmin=0, vmax=1, edgecolors="black", linewidth=1)
            
            # Add class labels
            for _, row in df_valid.iterrows():
                ax.annotate(row["class_name"], 
                          (row["train_count"], row["test_accuracy"]),
                          xytext=(5, 5), textcoords="offset points", 
                          fontsize=9, alpha=0.8)
            
            # Add trend line if enough data points
            if len(df_valid) > 2:
                z = np.polyfit(df_valid["train_count"], df_valid["test_accuracy"], 1)
                p = np.poly1d(z)
                x_line = np.linspace(df_valid["train_count"].min(), df_valid["train_count"].max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label="Trend Line")
            
            ax.set_xlabel("Training Exposure (Object Count in Train Split)", fontweight="bold", fontsize=14)
            ax.set_ylabel("Test Accuracy", fontweight="bold", fontsize=14)
            ax.set_title("Test Accuracy vs Training Exposure\n(bubble size = test object count)", 
                        fontweight="bold", fontsize=16)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Test Accuracy", fontweight="bold", fontsize=12)
            
            if len(df_valid) > 2:
                ax.legend(fontsize=12)
            
            plt.tight_layout()
            chart_path = test_run_dir / "accuracy_vs_training_exposure.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_vs_training_exposure"] = str(chart_path)
    
    # Chart 12: Train/Test Ratio vs Accuracy
    if not df_train_test.empty and train_class_counts:
        df_valid = df_train_test[(df_train_test["test_accuracy"].notna()) & 
                                (df_train_test["train_test_ratio"].notna())].copy()
        
        if not df_valid.empty:
            df_sorted = df_valid.sort_values("test_accuracy")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
            
            # Left: Accuracy by class with train/test ratio as color
            bars = ax1.barh(df_sorted["class_name"], df_sorted["test_accuracy"])
            
            # Color bars by train/test ratio
            norm = plt.Normalize(vmin=df_sorted["train_test_ratio"].min(), 
                               vmax=df_sorted["train_test_ratio"].max())
            cmap = plt.cm.coolwarm
            for bar, ratio in zip(bars, df_sorted["train_test_ratio"]):
                bar.set_color(cmap(norm(ratio)))
            
            ax1.set_xlabel("Test Accuracy", fontweight="bold", fontsize=14)
            ax1.set_ylabel("Class", fontweight="bold", fontsize=14)
            ax1.set_title("Test Accuracy by Class\n(colored by train/test ratio)", 
                         fontweight="bold", fontsize=14)
            ax1.set_xlim(0, 1)
            ax1.grid(axis="x", alpha=0.3)
            
            # Colorbar for left plot
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar1 = plt.colorbar(sm, ax=ax1)
            cbar1.set_label("Train/Test Ratio", fontweight="bold", fontsize=12)
            
            # Right: Direct scatter of ratio vs accuracy
            scatter = ax2.scatter(df_valid["train_test_ratio"], df_valid["test_accuracy"],
                                s=100, alpha=0.6, c=df_valid["test_accuracy"],
                                cmap="RdYlGn", vmin=0, vmax=1, edgecolors="black", linewidth=1)
            
            for _, row in df_valid.iterrows():
                ax2.annotate(row["class_name"], 
                           (row["train_test_ratio"], row["test_accuracy"]),
                           xytext=(5, 5), textcoords="offset points", 
                           fontsize=8, alpha=0.8)
            
            ax2.set_xlabel("Train/Test Object Ratio", fontweight="bold", fontsize=14)
            ax2.set_ylabel("Test Accuracy", fontweight="bold", fontsize=14)
            ax2.set_title("Accuracy vs Train/Test Ratio", fontweight="bold", fontsize=14)
            ax2.set_ylim(0, 1.05)
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            chart_path = test_run_dir / "accuracy_vs_train_test_ratio.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_vs_train_test_ratio"] = str(chart_path)

    # Overall statistics
    overall_expected = df_images["expected_total"].sum()
    overall_matched = df_images["matched"].sum()
    overall_accuracy = (overall_matched / overall_expected) if overall_expected > 0 else None
    
    # Per-class test accuracy summary (for train-test comparison)
    test_class_stats = {}
    for cid in class_names.keys():
        total_expected = 0
        total_matched = 0
        for rec in per_image_records:
            gts = rec.get("gts", [])
            for g in gts:
                if g["cls"] == cid:
                    total_expected += 1
            # Count matched for this class
            preds = rec.get("preds", [])
            matched_gt_local = set()
            for p in preds:
                if p["cls"] != cid:
                    continue
                for gi, g in enumerate(gts):
                    if gi in matched_gt_local or g["cls"] != cid:
                        continue
                    if iou_xyxy(p["xyxy"], g["xyxy"]) >= iou_threshold:
                        matched_gt_local.add(gi)
                        total_matched += 1
                        break
        test_class_stats[cid] = {
            "expected": total_expected,
            "matched": total_matched,
            "accuracy": (total_matched / total_expected) if total_expected > 0 else None
        }
    
    # Build train-test comparison dataframe
    train_test_rows = []
    for cid, cname in class_names.items():
        train_count = train_class_counts.get(cid, 0)
        test_stats = test_class_stats.get(cid, {})
        test_expected = test_stats.get("expected", 0)
        test_accuracy = test_stats.get("accuracy")
        
        if test_expected > 0:  # Only include classes present in test set
            train_test_rows.append({
                "class_id": cid,
                "class_name": cname,
                "train_count": train_count,
                "test_count": test_expected,
                "test_accuracy": test_accuracy,
                "train_test_ratio": (train_count / test_expected) if test_expected > 0 else None
            })
    
    df_train_test = pd.DataFrame(train_test_rows)

    # Summary output
    summary = {
        "overall": {
            "total_images": len(df_images),
            "total_expected_objects": int(overall_expected),
            "total_matched_objects": int(overall_matched),
            "overall_accuracy": overall_accuracy,
        },
        "train_metadata": {
            "train_images": train_total_images,
            "train_total_objects": sum(train_class_counts.values()) if train_class_counts else 0,
        },
        "by_weather": attr_summary_dfs["weather"].to_dict(orient="records") if not attr_summary_dfs["weather"].empty else [],
        "by_scene": attr_summary_dfs["scene"].to_dict(orient="records") if not attr_summary_dfs["scene"].empty else [],
        "by_timeofday": attr_summary_dfs["timeofday"].to_dict(orient="records") if not attr_summary_dfs["timeofday"].empty else [],
        "by_object_count": df_count_buckets.to_dict(orient="records"),
        "by_size": df_size_buckets.to_dict(orient="records"),
        "by_class_and_size": df_class_size.to_dict(orient="records") if not df_class_size.empty else [],
        "by_class_and_weather": df_class_weather.to_dict(orient="records") if not df_class_weather.empty else [],
        "by_class_and_scene": df_class_scene.to_dict(orient="records") if not df_class_scene.empty else [],
        "by_class_and_timeofday": df_class_time.to_dict(orient="records") if not df_class_time.empty else [],
        "train_test_comparison": df_train_test.to_dict(orient="records") if not df_train_test.empty else [],
        "charts": charts,
    }

    with open(test_run_dir / "failure_analysis_comprehensive.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Overall Accuracy: {overall_accuracy:.2%}" if overall_accuracy else "Overall Accuracy: N/A")
    print(f"Total Images: {len(df_images)}")
    print(f"Expected Objects: {overall_expected}")
    print(f"Matched Objects: {overall_matched}")
    
    if not attr_summary_dfs["weather"].empty:
        print("\nWeakest Weather Conditions:")
        for _, row in attr_summary_dfs["weather"].head(3).iterrows():
            if pd.notna(row['accuracy']):
                print(f"  - {row['weather']}: {row['accuracy']:.2%} ({row['images']} images)")
    
    if not attr_summary_dfs["scene"].empty:
        print("\nWeakest Scenes:")
        for _, row in attr_summary_dfs["scene"].head(3).iterrows():
            if pd.notna(row['accuracy']):
                print(f"  - {row['scene']}: {row['accuracy']:.2%} ({row['images']} images)")
    
    if not attr_summary_dfs["timeofday"].empty:
        print("\nWeakest Times of Day:")
        for _, row in attr_summary_dfs["timeofday"].head(3).iterrows():
            if pd.notna(row['accuracy']):
                print(f"  - {row['timeofday']}: {row['accuracy']:.2%} ({row['images']} images)")
    
    if not df_size_buckets.empty:
        print("\nAccuracy by Object Size (Scale/Distance):")
        for _, row in df_size_buckets.iterrows():
            if pd.notna(row['accuracy']):
                print(f"  - {row['size_bucket']}: {row['accuracy']:.2%} ({row['expected']} objects)")
    
    if not df_train_test.empty and train_class_counts:
        print("\nTrain-Test Comparison (Classes with Lowest Accuracy):")
        df_lowest = df_train_test[df_train_test["test_accuracy"].notna()].nsmallest(3, "test_accuracy")
        for _, row in df_lowest.iterrows():
            print(f"  - {row['class_name']}: {row['test_accuracy']:.2%} accuracy | "
                  f"Train: {row['train_count']} objs, Test: {row['test_count']} objs | "
                  f"Ratio: {row['train_test_ratio']:.1f}x")
    
    print("\n✓ Comprehensive failure analysis complete")
    print(f"  - Charts saved: {len(charts)}")
    print(f"  - CSV files saved: {9 + len(attr_summary_dfs)}")
    print(f"  - JSON summary: {test_run_dir / 'failure_analysis_comprehensive.json'}")
    print("=" * 80)

    return summary

def generate_pdf_and_json_report(
    model_name: str,
    run_name: str,
    wb_run_name: str,
    used_dataset: str,
    used_split: str,
    iou_threshold: float,
    test_run_dir: Path,
    model_info: Dict[str, float],
    metrics: Dict[str, Any],
    df_metrics: pd.DataFrame,
    confusion_matrix: np.ndarray,
    class_names: Dict[int, str],
    total_time: float,
    comparison_data: List[Dict[str, Any]] | None = None,
    failure_analysis_summary: Dict[str, Any] | None = None,
) -> None:
    pdf_report_path = test_run_dir / "report.pdf"
    json_report_path = test_run_dir / "metrics_data.json"

    doc = SimpleDocTemplate(
        str(pdf_report_path),
        pagesize=A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    story: List[Any] = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#34495e"),
        spaceAfter=12,
        spaceBefore=20,
    )

    story.append(Paragraph("YOLO Model Analysis Report", title_style))
    story.append(Spacer(1, 12))

    info_data = [
        ["Model:", model_name],
        ["Model Size:", f"{model_info.get('size(MB)', 0.0):.1f} MB"],
        ["Parameters:", f"{model_info.get('params', 0) / 1e6:.1f} M"],
        ["FLOPs (640x640):", f"{model_info.get('FLOPs(G)', 0.0):.2f} GFLOPs"],
        ["Run Name:", run_name],
        ["W&B Run Name:", wb_run_name],
        ["Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Dataset:", f"{used_dataset} - {used_split} split"],
        ["Images Processed:", str(metrics["num_images"])],
         ["Total Execution Time", f"{total_time:.2f}s"],
        ["IoU Threshold:", str(iou_threshold)],
    ]

    info_table = Table(info_data, colWidths=[2.2 * inch, 3.8 * inch])
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ecf0f1")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 1, colors.white),
            ]
        )
    )
    story.append(info_table)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Inference Performance", heading_style))
    perf_data = [
        ["Metric", "Value"],
        ["Average Inference Time", f"{metrics['avg_inference_time']:.2f} ms per image"],
        ["FPS (Frames Per Second)", f"{metrics['fps']:.2f}"],
    ]
    perf_table = Table(perf_data, colWidths=[3 * inch, 3 * inch])
    perf_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#27ae60")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#d5f4e6")),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(perf_table)
    note="Note: Inference time includes preprocessing, model inference, and postprocessing."
    story.append(Paragraph(note, styles["Normal"]))
    story.append(Spacer(1, 20))
        
    story.append(Paragraph("Overall Accuracy Metrics", heading_style))
    acc = metrics["overall"]
    yolo_m = metrics["yolo_metrics"]
    acc_data = [
        ["Metric", "Value"],
        ["Precision", f"{acc['precision']:.4f}"],
        ["Recall", f"{acc['recall']:.4f}"],
        ["F1-Score", f"{acc['f1']:.4f}"],
        ["mAP@0.5", f"{yolo_m['map50']:.4f}"],
        ["mAP@0.5:0.95", f"{yolo_m['map50_95']:.4f}"],
        ["True Positives", str(acc["tp"])],
        ["False Positives", str(acc["fp"])],
        ["False Negatives", str(acc["fn"])],
    ]
    acc_table = Table(acc_data, colWidths=[3 * inch, 3 * inch])
    acc_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(acc_table)
    
    story.append(PageBreak())
    story.append(Paragraph("Confusion Matrix", heading_style))

    confusion_matrix_img_path = test_run_dir / "confusion_matrix.png"
    if confusion_matrix_img_path.exists():
        with PILImage.open(confusion_matrix_img_path) as img:
            w, h = img.size
            ratio = h / w
            pdf_w = 6.5 * inch
            pdf_h = pdf_w * ratio
            story.append(Image(str(confusion_matrix_img_path), width=pdf_w, height=pdf_h))

    story.append(Paragraph(f"Correct Predictions (Diagonal Sum): {int(np.trace(confusion_matrix))}", styles["Normal"]))
    story.append(Paragraph(f"Total Matched Predictions: {int(confusion_matrix.sum())}", styles["Normal"]))


    story.append(PageBreak())
    story.append(Paragraph("Performance Visualizations", heading_style))

    # Add diagrams in pairs per page (two per page) for better layout
    core_and_map_figures = [
        "precision_by_class",
        "recall_by_class",
        "f1_by_class",
        "detection_outcomes",
        "map50_by_class",
        "overall_metrics",
    ]

    for idx, fig_key in enumerate(core_and_map_figures, 1):
        fig_path = test_run_dir / f"{fig_key}.png"
        if fig_path.exists():
            with PILImage.open(fig_path) as img:
                w, h = img.size
                ratio = h / w
                pdf_w = 5.0 * inch
                pdf_h = pdf_w * ratio
                story.append(Image(str(fig_path), width=pdf_w, height=pdf_h))
                story.append(Spacer(1, 8))

        # After every two figures, move to a new page (except after the last)
        if idx % 2 == 0 and idx < len(core_and_map_figures):
            story.append(PageBreak())

    story.append(PageBreak())
    story.append(Paragraph("Per-Class Performance", heading_style))

    table_data = [["Class", "TP", "FP", "FN", "Precision", "Recall", "F1-Score", "mAP@0.5"]]
    yolo_class_metrics = metrics["yolo_class_metrics"]
    for _, row in df_metrics.iterrows():
        class_name = row["Class"]
        map50_val = yolo_class_metrics.get(class_name, {}).get("ap50", 0.0)
        table_data.append(
            [
                str(row["Class"]),
                str(row["TP"]),
                str(row["FP"]),
                str(row["FN"]),
                f"{row['Precision']:.4f}",
                f"{row['Recall']:.4f}",
                f"{row['F1-Score']:.4f}",
                f"{map50_val:.4f}",
            ]
        )

    per_class_table = Table(
        table_data,
        colWidths=[1.0 * inch, 0.5 * inch, 0.5 * inch, 0.5 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch],
    )
    per_class_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(per_class_table)

    story.append(Spacer(1, 12))

    # Add Failure Analysis Section
    if failure_analysis_summary and failure_analysis_summary.get("charts"):
        story.append(PageBreak())
        story.append(Paragraph("Comprehensive Failure Analysis", heading_style))
        story.append(Spacer(1, 12))
        
        # Overall summary
        overall = failure_analysis_summary.get("overall", {})
        if overall:
            analysis_data = [
                ["Metric", "Value"],
                ["Total Images Analyzed", str(overall.get("total_images", 0))],
                ["Total Expected Objects", str(overall.get("total_expected_objects", 0))],
                ["Total Matched Objects", str(overall.get("total_matched_objects", 0))],
                ["Overall Accuracy", f"{overall.get('overall_accuracy', 0):.2%}" if overall.get('overall_accuracy') else "N/A"],
            ]
            analysis_table = Table(analysis_data, colWidths=[3 * inch, 3 * inch])
            analysis_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e67e22")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("FONTSIZE", (0, 1), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#fdebd0")),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ])
            )
            story.append(analysis_table)
            story.append(Spacer(1, 20))
        
        charts = failure_analysis_summary.get("charts", {})
        
        # Attribute-based analysis charts
        story.append(Paragraph("Accuracy by Environmental Attributes", heading_style))
        story.append(Spacer(1, 8))
        
        for chart_key in ["accuracy_by_weather", "accuracy_by_scene", "accuracy_by_timeofday"]:
            if chart_key in charts:
                chart_path = Path(charts[chart_key])
                if chart_path.exists():
                    with PILImage.open(chart_path) as img:
                        w, h = img.size
                        ratio = h / w
                        pdf_w = 5.5 * inch
                        pdf_h = pdf_w * ratio
                        story.append(Image(str(chart_path), width=pdf_w, height=pdf_h))
                        story.append(Spacer(1, 10))
        
        story.append(PageBreak())
        story.append(Paragraph("Accuracy by Object Characteristics", heading_style))
        story.append(Spacer(1, 8))
        
        # Object count and size charts
        for chart_key in ["accuracy_by_object_count", "accuracy_by_size"]:
            if chart_key in charts:
                chart_path = Path(charts[chart_key])
                if chart_path.exists():
                    with PILImage.open(chart_path) as img:
                        w, h = img.size
                        ratio = h / w
                        pdf_w = 5.5 * inch
                        pdf_h = pdf_w * ratio
                        story.append(Image(str(chart_path), width=pdf_w, height=pdf_h))
                        story.append(Spacer(1, 10))
        
        # Train-Test Comparison Section
        if "train_test_distribution" in charts or "accuracy_vs_training_exposure" in charts:
            story.append(PageBreak())
            story.append(Paragraph("Training Exposure vs Test Performance", heading_style))
            story.append(Spacer(1, 8))
            
            train_info = failure_analysis_summary.get("train_metadata", {})
            if train_info:
                train_summary_data = [
                    ["Metric", "Train Split", "Test Split"],
                    ["Total Images", str(train_info.get("train_images", 0)), 
                     str(failure_analysis_summary["overall"]["total_images"])],
                    ["Total Objects", str(train_info.get("train_total_objects", 0)),
                     str(failure_analysis_summary["overall"]["total_expected_objects"])],
                ]
                train_table = Table(train_summary_data, colWidths=[2.5 * inch, 1.75 * inch, 1.75 * inch])
                train_table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8e44ad")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 12),
                        ("FONTSIZE", (0, 1), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#e8daef")),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ])
                )
                story.append(train_table)
                story.append(Spacer(1, 16))
            
            # Train-test distribution chart
            if "train_test_distribution" in charts:
                chart_path = Path(charts["train_test_distribution"])
                if chart_path.exists():
                    with PILImage.open(chart_path) as img:
                        w, h = img.size
                        ratio = h / w
                        pdf_w = 6.5 * inch
                        pdf_h = pdf_w * ratio
                        story.append(Image(str(chart_path), width=pdf_w, height=pdf_h))
                        story.append(Spacer(1, 12))
            
            # Accuracy vs training exposure
            if "accuracy_vs_training_exposure" in charts:
                chart_path = Path(charts["accuracy_vs_training_exposure"])
                if chart_path.exists():
                    with PILImage.open(chart_path) as img:
                        w, h = img.size
                        ratio = h / w
                        pdf_w = 6.5 * inch
                        pdf_h = pdf_w * ratio
                        story.append(Image(str(chart_path), width=pdf_w, height=pdf_h))
                        story.append(Spacer(1, 12))
            
            # Accuracy vs train/test ratio
            if "accuracy_vs_train_test_ratio" in charts:
                chart_path = Path(charts["accuracy_vs_train_test_ratio"])
                if chart_path.exists():
                    with PILImage.open(chart_path) as img:
                        w, h = img.size
                        ratio = h / w
                        pdf_w = 6.5 * inch
                    pdf_h = pdf_w * ratio
                    story.append(Image(str(chart_path), width=pdf_w, height=pdf_h))
                    story.append(Spacer(1, 12))
        
        # Per-class breakdown charts
        story.append(PageBreak())
        story.append(Paragraph("Per-Class Performance Analysis", heading_style))
        story.append(Spacer(1, 8))
        
        for chart_key in ["accuracy_by_class_and_size", "accuracy_by_class_and_weather", 
                         "accuracy_by_class_and_scene", "accuracy_by_class_and_timeofday"]:
            if chart_key in charts:
                chart_path = Path(charts[chart_key])
                if chart_path.exists():
                    with PILImage.open(chart_path) as img:
                        w, h = img.size
                        ratio = h / w
                        pdf_w = 6.5 * inch
                        pdf_h = pdf_w * ratio
                        story.append(Image(str(chart_path), width=pdf_w, height=pdf_h))
                        story.append(Spacer(1, 12))
    
    # Add sample comparisons section with detailed attributes and full-width images
    if comparison_data:
        story.append(PageBreak())
        story.append(Paragraph("Sample Predictions: Ground Truth vs Model", heading_style))
        story.append(Spacer(1, 12))

        for idx, comp_info in enumerate(comparison_data, 1):
            comp_path = comp_info["comparison_image_path"]
            if not comp_path.exists():
                continue
            
            # Add detailed caption with attributes
            original_img = comp_info["original_image_path"]
            attributes = comp_info["attributes"]
            gt_count = comp_info["gt_count"]
            pred_count = comp_info["pred_count"]

            caption_parts = [f"<b>Sample #{idx}: {original_img.name}</b>"]
            caption_parts.append(f"Ground Truth Objects: {gt_count} | Predicted Objects: {pred_count}")

            if attributes:
                weather = attributes.get("weather", "unknown")
                scene = attributes.get("scene", "unknown")
                timeofday = attributes.get("timeofday", "unknown")
                caption_parts.append(f"Weather: {weather} | Scene: {scene} | Time of Day: {timeofday}")

            caption_text = "<br/>".join(caption_parts)
            story.append(Spacer(1, 6))
            story.append(Paragraph(caption_text, styles["Normal"]))
            story.append(Spacer(1, 12))

            # Add comparison image with full page width
            pil_img = PILImage.open(comp_path)
            img_width_px, img_height_px = pil_img.size
            aspect = img_width_px / img_height_px if img_height_px > 0 else 1.0

            # Use full available page width (A4 width minus margins)
            max_width = A4[0] - 60  # Full page width minus left and right margins (30 each)
            img_width = max_width
            img_height = img_width / aspect

            img_flow = Image(str(comp_path), width=img_width, height=img_height)
            story.append(img_flow)



    story.append(Spacer(1, 30))
    story.append(
        Paragraph(
            "Generated by YOLO Quick Test Script",
            ParagraphStyle("Footer", parent=styles["Normal"], alignment=TA_CENTER, textColor=colors.grey),
        )
    )

    doc.build(story)

    comparison_data = {
        "metadata": {
            "model_name": model_name,
            "run_name": run_name,
            "wb_run_name": wb_run_name,
            "timestamp": datetime.now().isoformat(),
            "dataset": used_dataset,
            "data_split": used_split,
            "images_processed": int(metrics["num_images"]),
            "iou_threshold": float(iou_threshold),
            "num_classes": len(class_names),
        },
        "model_info": {
            "parameters": int(model_info.get("params", 0)),
            "model_size_mb": float(model_info.get("size(MB)", 0.0)),
            "flops_gflops": float(model_info.get("FLOPs(G)", 0.0)),
        },
        "performance": {
            "total_time_seconds": float(total_time),
            "avg_inference_time_ms": float(metrics["avg_inference_time"] * 1000.0),
            "fps": float(metrics["fps"]),
            "images_processed": int(metrics["num_images"]),
        },
        "custom_metrics": {
            "overall": {
                "precision": float(metrics["overall"]["precision"]),
                "recall": float(metrics["overall"]["recall"]),
                "f1_score": float(metrics["overall"]["f1"]),
                "true_positives": int(metrics["overall"]["tp"]),
                "false_positives": int(metrics["overall"]["fp"]),
                "false_negatives": int(metrics["overall"]["fn"]),
            },
            "per_class": {},
        },
        "yolo_official_metrics": {
            "overall": metrics["yolo_metrics"],
            "per_class": metrics["yolo_class_metrics"],
        },
        "confusion_matrix": {
            "matrix": metrics["confusion_matrix"].tolist(),
            "diagonal_sum": int(np.trace(metrics["confusion_matrix"])),
            "total_predictions": int(metrics["confusion_matrix"].sum()),
        },
        "class_names": class_names,
    }

    for _, row in df_metrics.iterrows():
        class_name = row["Class"]
        comparison_data["custom_metrics"]["per_class"][class_name] = {
            "true_positives": int(row["TP"]),
            "false_positives": int(row["FP"]),
            "false_negatives": int(row["FN"]),
            "precision": float(row["Precision"]),
            "recall": float(row["Recall"]),
            "f1_score": float(row["F1-Score"]),
        }

    with open(json_report_path, "w") as f:
        json.dump(comparison_data, f, indent=2)

    print("=" * 80)
    print("✓ COMPREHENSIVE REPORT GENERATED (script)")
    print("=" * 80)
    print(f"PDF Report: {pdf_report_path}")
    print(f"JSON Metrics: {json_report_path}")


def run_validation_pipeline(
    model_name: str,
    dataset_name: str = "bdd100k_yolo_limited",
    split: str = "test",
    iou_threshold: float = 0.5,
    base_dir: Path | None = None,
    use_wandb: bool = False,
    save_reports: bool = True,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Run YOLO validation pipeline and return results directly.
    
    Args:
        model_name: YOLO model name (e.g., yolov8n, yolov8s)
        dataset_name: Dataset folder name under base directory
        split: Dataset split (train, val, or test)
        iou_threshold: IoU threshold for validation
        base_dir: Base project directory
        use_wandb: Whether to use W&B logging
        save_reports: Whether to save PDF and JSON reports
        
    Returns:
        Dictionary containing all metrics, figures, and paths
    """
    if base_dir is None:
        base_dir = Path.cwd().parent
    else:
        base_dir = Path(base_dir).resolve()
    
    used_dataset = dataset_name
    used_split = split

    device = setup_environment(use_wandb=use_wandb)

    yolo_dataset_root = base_dir / used_dataset
    data_yaml_path = yolo_dataset_root / "data.yaml"

    data_config = load_data_config(data_yaml_path=data_yaml_path, yolo_dataset_root=yolo_dataset_root)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_testing_{run_timestamp}"
    runs_dir = base_dir / "yolo_test" / "analysis_runs"
    test_run_dir = runs_dir / run_name
    test_run_dir.mkdir(parents=True, exist_ok=True)

    wb_project = f"yolo-{used_dataset}-testing"
    wb_run_name = f"{model_name}_{used_dataset}_{used_split}_{run_timestamp}"

    if use_wandb:
        try:
            wandb.init(
                project=wb_project,
                name=wb_run_name,
                config={
                    "model": model_name,
                    "dataset": used_dataset,
                    "split": used_split,
                    "iou_threshold": iou_threshold,
                    "batch_size": batch_size,
                },
            )
            print(f"\n✓ Weights & Biases initialized: {wb_run_name}")
        except Exception as wandb_error:
            print(f"\n⚠️  W&B initialization error: {wandb_error}")
            print("  Continuing without W&B tracking...")
            use_wandb = False

    dataset_info = load_dataset(used_dataset_root=yolo_dataset_root, used_split=used_split, data_config=data_config)

    models_dir = base_dir / "models" / model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    model, model_info = load_model(model_name=model_name, models_dir=models_dir)

    validation_results, total_time = run_yolo_validation(
        model=model,
        data_yaml_path=data_yaml_path,
        used_split=used_split,
        device=device,
        iou_threshold=iou_threshold,
        test_run_dir=test_run_dir,
        batch_size=batch_size,
    )

    metrics = extract_core_metrics(
        validation_results=validation_results,
        images_dir=dataset_info["images_dir"],
        num_classes=dataset_info["num_classes"],
        class_names=dataset_info["class_names"],
        model_info=model_info,
        total_time=total_time,
    )

    (
        df_metrics,
        overall_precision,
        overall_recall,
        overall_f1,
        total_tp,
        total_fp,
        total_fn,
    ) = build_per_class_dataframe(
        num_classes=dataset_info["num_classes"],
        class_names=dataset_info["class_names"],
        class_tp=metrics["class_tp"],
        class_fp=metrics["class_fp"],
        class_fn=metrics["class_fn"],
        yolo_class_metrics=metrics["yolo_class_metrics"],
    )

    metrics["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

    figure_paths = plot_core_and_map_metrics(
        df_metrics=df_metrics,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        overall_precision=overall_precision,
        overall_recall=overall_recall,
        overall_f1=overall_f1,
        yolo_metrics=metrics["yolo_metrics"],
        test_run_dir=test_run_dir,
    )

    confusion_matrix_path = plot_confusion_matrix(
        confusion_matrix=metrics["confusion_matrix"],
        num_classes=dataset_info["num_classes"],
        class_names=dataset_info["class_names"],
        model_name=model_name,
        test_run_dir=test_run_dir,
    )

    # Generate sample comparison images
    print("\n" + "=" * 80)
    print("GENERATING SAMPLE COMPARISONS")
    print("=" * 80)
    comparison_data = generate_sample_comparisons(
        model=model,
        valid_images=dataset_info["valid_images"],
        labels_dir=dataset_info["labels_dir"],
        class_names=dataset_info["class_names"],
        test_run_dir=test_run_dir,
        num_samples=6,
        device=device,
        image_attributes=dataset_info.get("image_attributes"),
    )

    # Generate failure analysis using expected counts (objects_per_class & total_objects)
    failure_analysis_summary = generate_failure_analysis(
        per_image_records=validation_results.per_image,
        image_attributes=dataset_info.get("image_attributes", {}),
        class_names=dataset_info["class_names"],
        test_run_dir=test_run_dir,
        dataset_root=yolo_dataset_root,
        iou_threshold=iou_threshold,
    )

    if save_reports:
        generate_pdf_and_json_report(
            model_name=model_name,
            run_name=run_name,
            wb_run_name=wb_run_name,
            used_dataset=used_dataset,
            used_split=used_split,
            iou_threshold=iou_threshold,
            test_run_dir=test_run_dir,
            model_info=model_info,
            metrics=metrics,
            df_metrics=df_metrics,
            confusion_matrix=metrics["confusion_matrix"],
            class_names=dataset_info["class_names"],
            total_time=total_time,
            comparison_data=comparison_data,
            failure_analysis_summary=failure_analysis_summary,
        )

    if use_wandb:
        try:
            wandb.finish()
            print("\n✓ Weights & Biases run completed successfully")
        except Exception as finish_error:
            print(f"\n⚠️  Error finishing W&B run: {finish_error}")

    # Clean up model from memory
    print("\n🧹 Cleaning up model from memory...")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ Model removed from memory")

    # Return comprehensive results
    return {
        "model_name": model_name,
        "run_name": run_name,
        "run_dir": test_run_dir,
        "model_info": model_info,
        "total_time": total_time,
        "dataset_info": dataset_info,
        "validation_results": validation_results,
        "metrics": metrics,
        "df_metrics": df_metrics,
        "figures": {
            **figure_paths,
            "confusion_matrix": confusion_matrix_path,
        },
        "comparison_data": comparison_data,
        "failure_analysis_summary": failure_analysis_summary,
        "yolo_validation_dir": test_run_dir / "yolo_validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO validation and generate report")
    parser.add_argument("--model-name", type=str, default="yolov8n", help="YOLO model name (e.g., yolov8n, yolov8s)")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="bdd100k_yolo_limited",
        help="Dataset folder name under base directory",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split: train, val, or test")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for validation")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base project directory (defaults to parent of current working dir)",
    )
    args = parser.parse_args()

    run_validation_pipeline(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split=args.split,
        iou_threshold=args.iou,
        base_dir=Path(args.base_dir) if args.base_dir else None,
        use_wandb=True,
        save_reports=True,
    )


if __name__ == "__main__":
    main()
