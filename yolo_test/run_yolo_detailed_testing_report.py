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

# Import the detailed analysis module
from detailed_analysis import collect_per_image_predictions, analyze_failures_by_attributes

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
            fontsize=22,
        )
        
        ax1.axis('off')

        ax2.imshow(pred_img)
        ax2.set_title(
            f"Prediction ({pred_count} objects)",
            fontweight="bold",
            fontsize=22,
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
        
        results = model.predict(img_path, conf=conf_threshold, verbose=True)
        
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
        ax.set_title(f"{img_path.name}", fontsize=14)
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
    metadata_file = metadata_dir / f"{used_split}_metadata.json"

    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata_data = json.load(f)
        print(f"\n✓ Metadata loaded: {metadata_file.name}")
        print(f"  Images with attributes: {metadata_data.get('total_files', 0)}")

        # The new structure has 'files' as a dict where keys are image IDs (basenames)
        # and values contain weather, scene, timeofday, categories, class_counts, object_count
        image_attributes = metadata_data.get("files", {})
    else:
        print(f"\n⚠️ Metadata not found: {metadata_file}")
        metadata_data = None
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
        "metadata_data": metadata_data,
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
    labels_dir: Path,
    batch_size: int = 16,
) -> Tuple[Any, float]:
    """
    HYBRID APPROACH:
    1. Run official model.val() for accurate metrics (matches standard YOLO validation)
    2. Run model.predict() per-image to collect detailed per-image results for attribute analysis
    
    This ensures metrics match official YOLO while still enabling rich failure analysis.

    Returns the official validation result object and total_time.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: OFFICIAL YOLO VALIDATION (for accurate metrics)")
    print("=" * 80)
    
    # Run official YOLO validation - this gives us the correct confusion matrix and metrics
    start_time = time.time()
    validation_results = model.val(
        data=str(data_yaml_path),
        split=used_split,
        batch=batch_size,
        device=device,
        iou=iou_threshold,
        conf=0.001,  # Very low conf threshold for evaluation (standard for mAP calculation)
                     # This captures ALL predictions; confidence filtering happens during metric computation
                     # Default for val() is 0.001, NOT 0.25 (which is for predict() inference)
        save_json=False,
        save_hybrid=False,
        plots=True,
        verbose=True,
        project=str(test_run_dir),
        name="yolo_validation",
    )
    val_time = time.time() - start_time
    
    print(f"\n✓ Official validation complete in {val_time:.2f}s")
    print(f"✓ Confusion matrix shape: {validation_results.confusion_matrix.matrix.shape if hasattr(validation_results, 'confusion_matrix') else 'N/A'}")
    
    # Extract official metrics
    if hasattr(validation_results, 'box'):
        box_metrics = validation_results.box
        print(f"\n✓ Official Metrics:")
        print(f"  - Precision: {box_metrics.mp:.4f}")
        print(f"  - Recall: {box_metrics.mr:.4f}")
        print(f"  - mAP@0.5: {box_metrics.map50:.4f}")
        print(f"  - mAP@0.5:0.95: {box_metrics.map:.4f}")
    
    print("\n" + "=" * 80)
    print("PHASE 2: PER-IMAGE PREDICTION (for attribute-based analysis)")
    print("=" * 80)
    
    # Collect per-image predictions using the detailed_analysis module
    per_image_records, predict_time = collect_per_image_predictions(
        model=model,
        data_yaml_path=data_yaml_path,
        used_split=used_split,
        device=device,
        iou_threshold=iou_threshold,
        labels_dir=labels_dir,
    )

    # Attach per_image_records to official validation_results for attribute-based analysis
    validation_results.per_image_records = per_image_records
    
    print(f"\n✓ Hybrid validation complete. Official validation time: {val_time:.2f}s")
    return validation_results, val_time


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

    # Get confusion matrix from validation results
    if hasattr(validation_results, "confusion_matrix") and hasattr(validation_results.confusion_matrix, "matrix"):
        confusion_matrix = np.array(validation_results.confusion_matrix.matrix, copy=True)
    else:
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Calculate TP/FP/FN counts from confusion matrix
    # This is the standard approach for extracting per-class metrics from YOLO validation
    class_tp: Dict[int, int] = {}
    class_fp: Dict[int, int] = {}
    class_fn: Dict[int, int] = {}
    
    for i in range(num_classes):
        tp_val = int(confusion_matrix[i, i]) if i < confusion_matrix.shape[0] and i < confusion_matrix.shape[1] else 0
        fp_val = int(confusion_matrix[:, i].sum() - confusion_matrix[i, i]) if i < confusion_matrix.shape[1] else 0
        fn_val = int(confusion_matrix[i, :].sum() - confusion_matrix[i, i]) if i < confusion_matrix.shape[0] else 0
        class_tp[i] = tp_val
        class_fp[i] = fp_val
        class_fn[i] = fn_val
    
    print(f"\n✓ Calculated per-class metrics from confusion matrix:")
    print(f"  Total TP: {sum(class_tp.values())}, FP: {sum(class_fp.values())}, FN: {sum(class_fn.values())}")

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
    ax.set_title("Precision by Class", fontweight="bold", fontsize=24, pad=20)
    ax.set_xlabel("Precision", fontweight="bold", fontsize=22)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=16)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # Add value labels inside bars
    for idx, (bar, value) in enumerate(zip(bars, precision_sorted["Precision"])):
        if value > 0.15:  # Inside bar if there's space
            ax.text(value - 0.05, idx, f'{value:.2f}', 
                   ha='right', va='center', color='black', fontweight='bold', fontsize=18)
        else:  # Outside bar if too small
            ax.text(value + 0.02, idx, f'{value:.2f}', 
                   ha='left', va='center', color='black', fontweight='bold', fontsize=18)
    plt.tight_layout()
    fig_paths["precision_by_class"] = test_run_dir / "precision_by_class.png"
    plt.savefig(fig_paths["precision_by_class"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Recall by class
    recall_sorted = df_metrics.sort_values("Recall")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    bars = ax.barh(recall_sorted["Class"], recall_sorted["Recall"], color="#F25F5C")
    ax.set_title("Recall by Class", fontweight="bold", fontsize=24, pad=20)
    ax.set_xlabel("Recall", fontweight="bold", fontsize=22)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=16)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # Add value labels inside bars
    for idx, (bar, value) in enumerate(zip(bars, recall_sorted["Recall"])):
        if value > 0.15:  # Inside bar if there's space
            ax.text(value - 0.05, idx, f'{value:.2f}', 
                   ha='right', va='center', color='black', fontweight='bold', fontsize=18)
        else:  # Outside bar if too small
            ax.text(value + 0.02, idx, f'{value:.2f}', 
                   ha='left', va='center', color='black', fontweight='bold', fontsize=18)
    plt.tight_layout()
    fig_paths["recall_by_class"] = test_run_dir / "recall_by_class.png"
    plt.savefig(fig_paths["recall_by_class"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    # F1-Score by class
    f1_sorted = df_metrics.sort_values("F1-Score")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    bars = ax.barh(f1_sorted["Class"], f1_sorted["F1-Score"], color="#9BC53D")
    ax.set_title("F1-Score by Class", fontweight="bold", fontsize=24, pad=20)
    ax.set_xlabel("F1-Score", fontweight="bold", fontsize=22)
    ax.set_xlim(0, 1.1)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=16)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # Add value labels inside bars
    for idx, (bar, value) in enumerate(zip(bars, f1_sorted["F1-Score"])):
        if value > 0.15:  # Inside bar if there's space
            ax.text(value - 0.05, idx, f'{value:.2f}', 
                   ha='right', va='center', color='black', fontweight='bold', fontsize=18)
        else:  # Outside bar if too small
            ax.text(value + 0.02, idx, f'{value:.2f}', 
                   ha='left', va='center', color='black', fontweight='bold', fontsize=18)
    plt.tight_layout()
    fig_paths["f1_by_class"] = test_run_dir / "f1_by_class.png"
    plt.savefig(fig_paths["f1_by_class"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Overall detection outcomes
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax.bar(["TP", "FP", "FN"], [total_tp, total_fp, total_fn], color=["#177E89", "#ED6A5A", "#F4A259"])
    ax.set_title("Overall Detection Outcomes", fontweight="bold", fontsize=22, pad=20)
    ax.set_ylabel("Count", fontweight="bold", fontsize=18)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="both", labelsize=18)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Add value labels - place inside if value is large enough, otherwise outside
    max_val = max(total_tp, total_fp, total_fn)
    for bar in bars:
        height = bar.get_height()
        # Place inside if bar height is more than 15% of max value
        if height > 0.15 * max_val:
            y_pos = height - (0.05 * max_val)  # Slightly below top of bar
            va = "center"
        else:  # Place outside if bar is too short
            y_pos = height + (0.02 * max_val)  # Slightly above bar
            va = "bottom"
        text_color = "black"
        
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{int(height)}",
            ha="center",
            va=va,
            fontweight="bold",
            fontsize=18,
            color=text_color,
        )
    plt.tight_layout()
    fig_paths["detection_outcomes"] = test_run_dir / "detection_outcomes.png"
    plt.savefig(fig_paths["detection_outcomes"], dpi=300, bbox_inches="tight")
    plt.close(fig)

    # mAP@0.5 by class
    map_sorted = df_metrics.sort_values("mAP@0.5")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    bars = ax.barh(map_sorted["Class"], map_sorted["mAP@0.5"], color="#B388EB")
    ax.set_title("mAP@0.5 by Class", fontweight="bold", fontsize=24, pad=20)
    ax.set_xlabel("mAP@0.5", fontweight="bold", fontsize=18)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # Add value labels inside bars
    for idx, (bar, value) in enumerate(zip(bars, map_sorted["mAP@0.5"])):
        if value > 0.15:  # Inside bar if there's space
            ax.text(value - 0.05, idx, f'{value:.2f}', 
                   ha='right', va='center', color='black', fontweight='bold', fontsize=18)
        else:  # Outside bar if too small
            ax.text(value + 0.02, idx, f'{value:.2f}', 
                   ha='left', va='center', color='black', fontweight='bold', fontsize=18)
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
    ax.set_title("Overall Metrics", fontweight="bold", fontsize=24, pad=20)
    ax.set_ylabel("Score", fontweight="bold", fontsize=22)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="both", labelsize=16)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Add value labels inside bars (white text) if bar is tall enough, otherwise outside (black text)
    for idx, (bar, value) in enumerate(zip(bars, overall_plot_values.values())):
        if value > 0.15:  # If bar is tall enough, place text inside
            y_pos = value - 0.09
        else:  # Otherwise place text above
            y_pos = value + 0.02
        text_color = "black"
        
        ax.text(
            idx,
            y_pos,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
            fontsize=18,
            color=text_color,
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
                    fontsize=10,
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
    ax.set_xticklabels(class_labels, fontsize=14, fontweight='bold', rotation=45, ha='right')
    ax.set_yticklabels(class_labels, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontweight='bold', fontsize=16)
    ax.set_ylabel('True Class', fontweight='bold', fontsize=16)
    ax.set_title(f'Confusion Matrix ({model_name} validation)', fontweight='bold', fontsize=18)
    ax.grid(False)

    # Center the confusion matrix in the figure
    plt.tight_layout()

    confusion_matrix_path = test_run_dir / "confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print("(Green = Correct Predictions, Red = Incorrect Predictions, White = No Predictions)")
    
    return confusion_matrix_path


def copy_normalized_confusion_matrix(
    test_run_dir: Path,
) -> Tuple[Path | None, Path | None]:
    """Copy confusion matrices from YOLO validation output.
    
    YOLO's validation automatically generates both regular and normalized confusion matrices.
    This function copies both to the main test directory for inclusion in the report.
    
    Returns:
        Tuple of (regular_cm_path, normalized_cm_path) or (None, None) if not found.
    """
    yolo_val_dir = test_run_dir / "yolo_validation"
    
    # Copy regular confusion matrix
    cm_source = yolo_val_dir / "confusion_matrix.png"
    cm_dest = None
    if cm_source.exists():
        cm_dest = test_run_dir / "confusion_matrix_yolo.png"
        shutil.copy2(cm_source, cm_dest)
        print(f"✓ Copied confusion matrix from YOLO validation")
    else:
        print(f"⚠️  Confusion matrix not found in YOLO validation output")
    
    # Copy normalized confusion matrix
    normalized_cm_source = yolo_val_dir / "confusion_matrix_normalized.png"
    normalized_cm_dest = None
    if normalized_cm_source.exists():
        normalized_cm_dest = test_run_dir / "confusion_matrix_normalized.png"
        shutil.copy2(normalized_cm_source, normalized_cm_dest)
        print(f"✓ Copied normalized confusion matrix from YOLO validation")
    else:
        print(f"⚠️  Normalized confusion matrix not found in YOLO validation output")
    
    return cm_dest, normalized_cm_dest


def generate_failure_analysis(
    per_image_records: List[Dict[str, Any]],
    image_attributes: Dict[str, Any],
    class_names: Dict[int, str],
    test_run_dir: Path,
    dataset_root: Path,
    iou_threshold: float = 0.5,
    include_training_exposure_analysis: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive analysis of prediction accuracy in relation to:
    - Image attributes (weather, scene, timeofday)
    - Object counts (object_count, class_counts)
    
    Generates:
    - CSV files with detailed per-image and aggregated data
    - Charts showing accuracy by attribute values
    - Charts showing accuracy by object count ranges
    - Per-class accuracy breakdowns within each attribute
    - Optional: Training exposure vs test performance charts (if include_training_exposure_analysis=True)
    """
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE FAILURE ANALYSIS")
    print("Analyzing relationship between attributes and prediction accuracy...")
    print("=" * 80)

    # build reverse map from class name -> id
    class_name_to_id = {name: cid for cid, name in class_names.items()}
    
    # Initialize train-test comparison DataFrame (will be populated later)
    df_train_test = pd.DataFrame()
    
    # Load training split metadata to analyze training exposure (only if requested)
    train_class_counts = {}
    train_total_images = 0
    
    if include_training_exposure_analysis:
        train_metadata_path = dataset_root / "representative_json" / "train_metadata.json"
        
        print(f"\n🔍 Attempting to load training metadata from: {train_metadata_path}")
        print(f"  File exists: {train_metadata_path.exists()}")
        
        if train_metadata_path.exists():
            print("\nLoading training split metadata for exposure analysis...")
            try:
                with open(train_metadata_path, "r") as f:
                    train_data = json.load(f)
                train_total_images = train_data.get("total_files", 0)
                
                # Extract per-class counts from training data
                # New structure: files is a dict with image IDs as keys
                print(f"  Processing {len(train_data.get('files', {}))} training images...")
                unmatched_classes = set()  # Track unmatched class names
                for img_id, img_data in train_data.get("files", {}).items():
                    class_counts = img_data.get("class_counts", {})
                    for class_name, count in class_counts.items():
                        cid = class_name_to_id.get(class_name)
                        if cid is not None:
                            train_class_counts[cid] = train_class_counts.get(cid, 0) + int(count)
                        else:
                            unmatched_classes.add(class_name)
                
                if unmatched_classes:
                    print(f"  ⚠️  Warning: {len(unmatched_classes)} class names not found in class_name_to_id mapping:")
                    print(f"     Unmatched: {list(unmatched_classes)[:10]}")  # Show first 10
                    print(f"     Available: {list(class_names.values())[:10]}")  # Show first 10 valid names
                
                print(f"✓ Training metadata loaded: {train_total_images} images, {sum(train_class_counts.values())} objects")
                print(f"  - Classes found in training data: {len(train_class_counts)}")
                if train_class_counts:
                    print(f"  - Sample class counts: {dict(list(train_class_counts.items())[:3])}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load training metadata: {e}")
                train_class_counts = {}
        else:
            print(f"⚠️  Training metadata not found: {train_metadata_path}")
            train_class_counts = {}
    else:
        print("\n✓ Skipping training exposure analysis (include_training_exposure_analysis=False)")

    # Call detailed analysis module to compute accuracy by attributes
    # Prepare training metadata for analysis
    train_metadata_dict = {
        "class_counts": train_class_counts,
        "total_images": train_total_images,
    }
    
    analysis_summary, analysis_dfs, _ = analyze_failures_by_attributes(
        per_image_records=per_image_records,
        image_attributes=image_attributes,
        class_names=class_names,
        class_name_to_id=class_name_to_id,
        iou_threshold=iou_threshold,
        test_run_dir=test_run_dir,
        train_metadata_data=train_metadata_dict,
    )
    
    # Extract DataFrames from analysis results with backward-compatible names
    df_images = analysis_dfs["per_image"]
    attr_summary_dfs = {
        "weather": analysis_dfs["weather"],
        "scene": analysis_dfs["scene"],
        "timeofday": analysis_dfs["timeofday"],
    }
    df_count_buckets = analysis_dfs["object_count"]
    df_size_buckets = analysis_dfs["object_size"]
    df_class_size = analysis_dfs["class_size"]
    df_class_weather = analysis_dfs["class_weather"]
    df_class_scene = analysis_dfs["class_scene"]
    df_class_time = analysis_dfs["class_time"]
    
    print(f"\n✓ Attribute-based analysis complete")
    print(f"  Total images analyzed: {analysis_summary['total_images']}")
    print(f"  Overall accuracy: {analysis_summary['overall_accuracy']:.2%}")
    
    # Helper IoU function (needed for subsequent test_class_stats computation)
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

    # =========================================================================
    # BUILD df_train_test BEFORE chart generation (so charts can use it)
    # =========================================================================
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
    
    # Debug logging for training exposure analysis
    print(f"\n📊 Training Exposure Analysis Status:")
    print(f"  - train_class_counts populated: {len(train_class_counts) > 0} ({len(train_class_counts)} classes)")
    print(f"  - df_train_test populated: {not df_train_test.empty} ({len(df_train_test)} rows)")
    if not df_train_test.empty:
        print(f"  - Sample df_train_test rows:")
        print(df_train_test.head(3).to_string(index=False))
    if train_class_counts and not df_train_test.empty:
        print(f"  ✓ Training exposure charts will be generated")
    else:
        if not train_class_counts:
            print(f"  ⚠️  No training class counts - check train_metadata.json exists and is valid")
        if df_train_test.empty:
            print(f"  ⚠️  df_train_test is empty - no classes with test data found")

    # Create performance analysis subfolder for charts
    analysis_dir = test_run_dir / "performance_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Analysis charts will be saved to: {analysis_dir}")

    # Generate charts
    print("\n Generating accuracy analysis charts...")
    charts = {}
    
    sns.set_style("whitegrid")
    
    # Chart 1: Accuracy by Weather
    if not attr_summary_dfs["weather"].empty:
        df_weather = attr_summary_dfs["weather"].sort_values("accuracy")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        bars = ax.barh(df_weather["weather"], df_weather["accuracy"], color="#5BC0EB")
        ax.set_xlabel("Accuracy", fontweight="bold", fontsize=20)
        ax.set_ylabel("Weather", fontweight="bold", fontsize=20)
        ax.set_title("Prediction Accuracy by Weather Condition", fontweight="bold", fontsize=22)
        ax.set_xlim(0, 1)
        ax.tick_params(axis='y', labelsize=18)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        for i, (idx, row) in enumerate(df_weather.iterrows()):

            acc = row["accuracy"]
            if acc > 0.30:  # Place inside if bar is tall enough
                x_pos = acc - 0.25
            else:  # Place outside if bar is too short
                x_pos = acc + 0.02
            text_color = "black"
            ax.text(x_pos, i, f'{acc:.2%} ({row["images"]} imgs)', 
                   va='center', fontsize=16, fontweight='bold', color=text_color)
        plt.tight_layout()
        chart_path = analysis_dir / "accuracy_by_weather.png"
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        charts["accuracy_by_weather"] = str(chart_path)

    # Chart 2: Accuracy by Scene
    if not attr_summary_dfs["scene"].empty:
        df_scene = attr_summary_dfs["scene"].sort_values("accuracy")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        bars = ax.barh(df_scene["scene"], df_scene["accuracy"], color="#F25F5C")
        ax.set_xlabel("Accuracy", fontweight="bold", fontsize=20)
        ax.set_ylabel("Scene", fontweight="bold", fontsize=20)
        ax.set_title("Prediction Accuracy by Scene Type", fontweight="bold", fontsize=22)
        ax.set_xlim(0, 1)
        ax.tick_params(axis='y', labelsize=18)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        for i, (idx, row) in enumerate(df_scene.iterrows()):
            acc = row["accuracy"]
            if acc > 0.30:  # Place inside if bar is tall enough
                x_pos = acc - 0.25
            else:  # Place outside if bar is too short
                x_pos = acc + 0.02
            text_color = "black"
            ax.text(x_pos, i, f'{acc:.2%} ({row["images"]} imgs)', 
                   va='center', fontsize=16, fontweight='bold', color=text_color)
        plt.tight_layout()
        chart_path = analysis_dir / "accuracy_by_scene.png"
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        charts["accuracy_by_scene"] = str(chart_path)

    # Chart 3: Accuracy by Time of Day
    if not attr_summary_dfs["timeofday"].empty:
        df_time = attr_summary_dfs["timeofday"].sort_values("accuracy")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        bars = ax.barh(df_time["timeofday"], df_time["accuracy"], color="#9BC53D")
        ax.set_xlabel("Accuracy", fontweight="bold", fontsize=20)
        ax.set_ylabel("Time of Day", fontweight="bold", fontsize=20)
        ax.set_title("Prediction Accuracy by Time of Day", fontweight="bold", fontsize=22)
        ax.set_xlim(0, 1)
        ax.tick_params(axis='y', labelsize=18)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        for i, (idx, row) in enumerate(df_time.iterrows()):
            acc = row["accuracy"]
            if acc > 0.30:  # Place inside if bar is tall enough
                x_pos = acc - 0.25
            else:  # Place outside if bar is too short
                x_pos = acc + 0.02
            text_color = "black"
            ax.text(x_pos, i, f'{acc:.2%} ({row["images"]} imgs)', 
                   va='center', fontsize=16, fontweight='bold', color=text_color)
        plt.tight_layout()
        chart_path = analysis_dir / "accuracy_by_timeofday.png"
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
            ax.set_xlabel("Objects per Image", fontweight="bold", fontsize=20)
            ax.set_ylabel("Accuracy", fontweight="bold", fontsize=20)
            ax.set_title("Prediction Accuracy by Object Count", fontweight="bold", fontsize=22)
            ax.set_ylim(0, 1)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            for i, row in df_count_valid.iterrows():
                acc = row["accuracy"]
                if acc > 0.18:  # Place inside if bar is tall enough
                    y_pos = acc - 0.12
                else:  # Place outside if bar is too short
                    y_pos = acc + 0.02
                text_color = "black"
                ax.text(i, y_pos, f'{acc:.2%}\n({row["images"]} imgs)', 
                       ha='center', fontsize=16, fontweight='bold', color=text_color)
            plt.tight_layout()
            chart_path = analysis_dir / "accuracy_by_object_count.png"
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
            ax.set_xlabel("Object Size (bbox area relative to image)", fontweight="bold", fontsize=20)
            ax.set_ylabel("Accuracy", fontweight="bold", fontsize=20)
            ax.set_title("Prediction Accuracy by Object Size", 
                        fontweight="bold", fontsize=18, pad=20)
            ax.set_ylim(0, 1)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            for i, row in df_size_valid.iterrows():
                acc = row["accuracy"]
                if acc > 0.18:  # Place inside if bar is tall enough
                    y_pos = acc - 0.12
                else:  # Place outside if bar is too short
                    y_pos = acc + 0.02
                text_color = "black"
                ax.text(i, y_pos, 
                       f'{acc:.2%}\n({row["expected"]} objs)', 
                       ha='center', fontsize=16, fontweight='bold', color=text_color)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            chart_path = analysis_dir / "accuracy_by_size.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_by_size"] = str(chart_path)
    
    # Chart 6: Per-Class Accuracy by Size (Heatmap)
    if not df_class_size.empty:
        pivot_data = df_class_size.pivot(index="class_name", columns="size_bucket", values="accuracy")
        pivot_data = pivot_data[["small", "medium", "large"]]  # Order columns
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
        # Use custom colormap with medium red and green for better text visibility and contrast
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['#ff9999', '#ffff99', '#99ff99']  # Medium red -> Light yellow -> Medium green
        custom_cmap = LinearSegmentedColormap.from_list('custom_RdYlGn', colors_list)
        sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap=custom_cmap, 
                   vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, 
                   annot_kws={'size': 16, 'weight': 'bold', 'color': 'black'}, ax=ax)
        ax.set_title("Per-Class Accuracy by Object Size", fontweight="bold", fontsize=22)
        ax.set_xlabel("Object Size", fontweight="bold", fontsize=20)
        ax.set_ylabel("Class", fontweight="bold", fontsize=20)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        plt.tight_layout()
        chart_path = analysis_dir / "accuracy_by_class_and_size.png"
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        charts["accuracy_by_class_and_size"] = str(chart_path)
    
    # Chart 7: Per-Class Accuracy by Weather (All classes)
    if not df_class_weather.empty:
        pivot_data = df_class_weather.pivot(index="class_name", columns="weather", values="accuracy")
        
        if not pivot_data.empty:
            # Dynamic figure height based on number of classes (minimum 8, 0.6 inches per class)
            num_classes = len(pivot_data)
            fig_height = max(8, num_classes * 0.6)
            fig, ax = plt.subplots(figsize=(12, fig_height), dpi=200)
            # Use custom colormap with medium red and green for better text visibility and contrast
            from matplotlib.colors import LinearSegmentedColormap
            colors_list = ['#ff9999', '#ffff99', '#99ff99']  # Medium red -> Light yellow -> Medium green
            custom_cmap = LinearSegmentedColormap.from_list('custom_RdYlGn', colors_list)
            sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap=custom_cmap, 
                       vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, 
                       annot_kws={'size': 16, 'weight': 'bold', 'color': 'black'}, ax=ax)
            ax.set_title("Per-Class Accuracy by Weather", fontweight="bold", fontsize=22)
            ax.set_xlabel("Weather Condition", fontweight="bold", fontsize=20)
            ax.set_ylabel("Class", fontweight="bold", fontsize=20)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            plt.tight_layout()
            chart_path = analysis_dir / "accuracy_by_class_and_weather.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_by_class_and_weather"] = str(chart_path)
    
    # Chart 8: Per-Class Accuracy by Scene (All classes)
    if not df_class_scene.empty:
        pivot_data = df_class_scene.pivot(index="class_name", columns="scene", values="accuracy")
        
        if not pivot_data.empty:
            # Dynamic figure height based on number of classes (minimum 8, 0.6 inches per class)
            num_classes = len(pivot_data)
            fig_height = max(8, num_classes * 0.6)
            fig, ax = plt.subplots(figsize=(12, fig_height), dpi=200)
            # Use custom colormap with medium red and green for better text visibility and contrast
            from matplotlib.colors import LinearSegmentedColormap
            colors_list = ['#ff9999', '#ffff99', '#99ff99']  # Medium red -> Light yellow -> Medium green
            custom_cmap = LinearSegmentedColormap.from_list('custom_RdYlGn', colors_list)
            sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap=custom_cmap, 
                       vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, 
                       annot_kws={'size': 16, 'weight': 'bold', 'color': 'black'}, ax=ax)
            ax.set_title("Per-Class Accuracy by Scene", fontweight="bold", fontsize=22)
            ax.set_xlabel("Scene Type", fontweight="bold", fontsize=20)
            ax.set_ylabel("Class", fontweight="bold", fontsize=20)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            plt.tight_layout()
            chart_path = analysis_dir / "accuracy_by_class_and_scene.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_by_class_and_scene"] = str(chart_path)
    
    # Chart 9: Per-Class Accuracy by Time of Day (All classes)
    if not df_class_time.empty:
        pivot_data = df_class_time.pivot(index="class_name", columns="timeofday", values="accuracy")
        
        if not pivot_data.empty:
            # Dynamic figure height based on number of classes (minimum 8, 0.6 inches per class)
            num_classes = len(pivot_data)
            fig_height = max(8, num_classes * 0.6)
            fig, ax = plt.subplots(figsize=(12, fig_height), dpi=200)
            # Use custom colormap with medium red and green for better text visibility and contrast
            from matplotlib.colors import LinearSegmentedColormap
            colors_list = ['#ff9999', '#ffff99', '#99ff99']  # Medium red -> Light yellow -> Medium green
            custom_cmap = LinearSegmentedColormap.from_list('custom_RdYlGn', colors_list)
            sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap=custom_cmap, 
                       vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'}, 
                       annot_kws={'size': 16, 'weight': 'bold', 'color': 'black'}, ax=ax)
            ax.set_title("Per-Class Accuracy by Time of Day", fontweight="bold", fontsize=22)
            ax.set_xlabel("Time of Day", fontweight="bold", fontsize=20)
            ax.set_ylabel("Class", fontweight="bold", fontsize=20)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            plt.tight_layout()
            chart_path = analysis_dir / "accuracy_by_class_and_timeofday.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_by_class_and_timeofday"] = str(chart_path)
    
    # Debug: Check conditions for training exposure charts
    if include_training_exposure_analysis:
        print(f"\n🔍 Training Exposure Charts Generation Check:")
        print(f"  - train_class_counts: {len(train_class_counts)} classes" if train_class_counts else "  - train_class_counts: EMPTY")
        print(f"  - df_train_test: {len(df_train_test)} rows" if not df_train_test.empty else "  - df_train_test: EMPTY")
        print(f"  - Condition (not df_train_test.empty and train_class_counts): {not df_train_test.empty and train_class_counts}")
    
    # Chart 10: Train vs Test Class Distribution - REMOVED per user request
    # if not df_train_test.empty and train_class_counts:
    #     df_sorted = df_train_test.sort_values("train_count", ascending=False)
    #     ... (chart generation code removed)
    #     print(f"  ✓ Generated: train_test_distribution.png")
    # else:
    #     print(f"  ✗ Skipped Chart 10: train_test_distribution (condition not met)")
    
    # Chart 11: Test Accuracy vs Training Exposure (Scatter) - Only if requested
    if include_training_exposure_analysis and not df_train_test.empty and train_class_counts:
        df_valid = df_train_test[df_train_test["test_accuracy"].notna()].copy()
        
        if not df_valid.empty:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
            
            # Define marker styles for each class to make them distinguishable
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
            
            # Scatter plot with different markers for each class
            for idx, (_, row) in enumerate(df_valid.iterrows()):
                marker = markers[idx % len(markers)]
                ax.scatter(row["train_count"], row["test_accuracy"],
                          s=150, marker=marker, alpha=0.7,
                          c=[row["test_accuracy"]], cmap="RdYlGn", vmin=0, vmax=1,
                          edgecolors="black", linewidth=1.5)
            
            # Add class labels in bold
            for _, row in df_valid.iterrows():
                ax.annotate(row["class_name"], 
                          (row["train_count"], row["test_accuracy"]),
                          xytext=(5, 5), textcoords="offset points", 
                          fontsize=14, fontweight='bold', alpha=0.9)
            
            # Add trend line if enough data points
            if len(df_valid) > 2:
                z = np.polyfit(df_valid["train_count"], df_valid["test_accuracy"], 1)
                p = np.poly1d(z)
                x_line = np.linspace(df_valid["train_count"].min(), df_valid["train_count"].max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2, label="Trend Line")
            
            ax.set_xlabel("Training Exposure (Object Count in Train Split)", fontweight="bold", fontsize=18)
            ax.set_ylabel("Test Accuracy", fontweight="bold", fontsize=18)
            ax.set_title("Test Accuracy vs Training Exposure", 
                        fontweight="bold", fontsize=20)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            for label in ax.get_xticklabels():
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
            
            # Create a colorbar with a dummy scatter for reference
            sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Test Accuracy", fontweight="bold", fontsize=14)
            
            if len(df_valid) > 2:
                ax.legend(fontsize=12)
            
            plt.tight_layout()
            chart_path = analysis_dir / "accuracy_vs_training_exposure.png"
            plt.savefig(chart_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            charts["accuracy_vs_training_exposure"] = str(chart_path)
            print(f"  ✓ Generated: accuracy_vs_training_exposure.png")
        else:
            print(f"  ✗ Skipped Chart 11: accuracy_vs_training_exposure (df_valid empty)")
    

    # Overall statistics
    overall_expected = df_images["expected_total"].sum()
    overall_matched = df_images["matched"].sum()
    overall_accuracy = (overall_matched / overall_expected) if overall_expected > 0 else None

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
    include_training_exposure_analysis: bool = True,
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
        fontSize=30,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=20,
        textColor=colors.HexColor("#34495e"),
        spaceAfter=12,
        spaceBefore=20,
    )

    # Set title based on include_training_exposure_analysis
    report_title = "YOLO Model Analysis Report" if include_training_exposure_analysis else "YOLO Model Testing Report"
    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 12))

    info_data = [
        ["Model:", model_name],
        ["Model Size:", f"{model_info.get('size(MB)', 0.0):.1f} MB"],
        ["Parameters:", f"{model_info.get('params', 0) / 1e6:.1f} M"],
        ["FLOPs (640x640):", f"{model_info.get('FLOPs(G)', 0.0):.2f} GFLOPs"],
        ["Run Name:", run_name],
        ["Timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Dataset:", f"{used_dataset} - {used_split} split"],
        ["Images Processed:", str(metrics["num_images"])],
         ["Total Execution Time", f"{total_time:.2f}s"],
        ["IoU Threshold:", str(iou_threshold)],
    ]

    info_table = Table(info_data, colWidths=[1.8 * inch, 3.2 * inch])
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ecf0f1")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
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
    perf_table = Table(perf_data, colWidths=[2.2 * inch, 2.8 * inch])
    perf_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#27ae60")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 13),
                ("FONTSIZE", (0, 1), (-1, -1), 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
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
    acc_table = Table(acc_data, colWidths=[2.2 * inch, 2.8 * inch])
    acc_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 13),
                ("FONTSIZE", (0, 1), (-1, -1), 11),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(acc_table)
    
    story.append(Spacer(1, 24))
    story.append(Paragraph("Confusion Matrices", heading_style))

    # Display both confusion matrices from YOLO validation
    # Regular confusion matrix (raw counts)
    confusion_matrix_img_path = test_run_dir / "confusion_matrix_yolo.png"
    if confusion_matrix_img_path.exists():
        story.append(Paragraph("Confusion Matrix (Counts)", styles["Heading2"]))
        story.append(Spacer(1, 6))
        with PILImage.open(confusion_matrix_img_path) as img:
            w, h = img.size
            ratio = h / w
            pdf_w = 6.5 * inch
            pdf_h = pdf_w * ratio
            story.append(Image(str(confusion_matrix_img_path), width=pdf_w, height=pdf_h))
        story.append(Paragraph("Note: Shows raw prediction counts for each class combination.", styles["Normal"]))
        story.append(Spacer(1, 16))
    
    # Normalized confusion matrix
    story.append(Paragraph("Confusion Matrix (Normalized)", styles["Heading2"]))
    story.append(Spacer(1, 6))
    normalized_confusion_matrix_img_path = test_run_dir / "confusion_matrix_normalized.png"
    if normalized_confusion_matrix_img_path.exists():
        with PILImage.open(normalized_confusion_matrix_img_path) as img:
            w, h = img.size
            ratio = h / w
            pdf_w = 6.5 * inch
            pdf_h = pdf_w * ratio
            story.append(Image(str(normalized_confusion_matrix_img_path), width=pdf_w, height=pdf_h))
        story.append(Paragraph("Note: Values are normalized by the number of ground truth instances per class. Shows the proportion of correct vs incorrect classifications.", styles["Normal"]))
    else:
        story.append(Paragraph("Confusion matrices not available from YOLO validation.", styles["Normal"]))
    
    story.append(Spacer(1, 10))
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
            story.append(Spacer(1, 24))

    story.append(Spacer(1, 24))
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
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(per_class_table)

    story.append(Spacer(1, 12))

    # Add Failure Analysis Section
    if failure_analysis_summary and failure_analysis_summary.get("charts"):
        story.append(Spacer(1, 24))
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
        
        story.append(Spacer(1, 24))
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
                        story.append(Spacer(1, 6))
                        
                        # Add detailed note for size chart
                        if chart_key == "accuracy_by_size":
                            size_note = (
                                "<b>Note:</b> Object size categories are defined based on bounding box area as a percentage of total image area: "
                                "<b>Small</b> objects occupy less than 1% of the image (typically distant or small objects), "
                                "<b>Medium</b> objects occupy between 1% and 5% of the image (moderate-sized objects at medium distances), "
                                "and <b>Large</b> objects occupy more than 5% of the image (close or large objects). "
                                "These thresholds help identify how object scale affects detection accuracy."
                            )
                            story.append(Paragraph(size_note, styles["Normal"]))
                        story.append(Spacer(1, 10))
        
        # Train-Test Comparison Section
        if "train_test_distribution" in charts or "accuracy_vs_training_exposure" in charts or "accuracy_vs_train_test_ratio" in charts:
            story.append(Spacer(1, 24))
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
            # Train-test distribution table and chart - REMOVED per user request
            
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
                else:
                    print(f"  ⚠️  Accuracy vs training exposure chart file not found: {chart_path}")
            
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
                else:
                    print(f"  ⚠️  Accuracy vs train/test ratio chart file not found: {chart_path}")
        else:
            print("  ⚠️  Training exposure charts not found in failure_analysis_summary")
            if failure_analysis_summary:
                print(f"  Available chart keys: {list(failure_analysis_summary.get('charts', {}).keys())}")
        
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
            "Generated by YOLO Testing Pipeline",
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
    dataset_path: Path | None = None,
    dataset_name: str = "bdd100k_yolo_limited",
    split: str = "test",
    iou_threshold: float = 0.5,
    base_dir: Path | None = None,
    use_wandb: bool = False,
    save_reports: bool = True,
    batch_size: int = 16,
    include_training_exposure_analysis: bool = False,
) -> Dict[str, Any]:
    """
    Run YOLO validation pipeline and return results directly.
    
    Args:
        model_name: YOLO model name (e.g., yolov8n, yolov8s)
        dataset_path: Absolute path to dataset directory (if provided, overrides dataset_name and base_dir)
        dataset_name: Dataset folder name under base directory (ignored if dataset_path is provided)
        split: Dataset split (train, val, or test)
        iou_threshold: IoU threshold for validation
        base_dir: Base project directory (ignored if dataset_path is provided)
        use_wandb: Whether to use W&B logging
        save_reports: Whether to save PDF and JSON reports
        batch_size: Batch size for inference
        include_training_exposure_analysis: Whether to include training exposure vs test performance analysis
        
    Returns:
        Dictionary containing all metrics, figures, and paths
    """
    # Determine dataset root from dataset_path or construct from base_dir + dataset_name
    if dataset_path is not None:
        yolo_dataset_root = Path(dataset_path).resolve()
        used_dataset = yolo_dataset_root.name
        # Use parent directory for storing results
        base_dir = yolo_dataset_root.parent
    else:
        if base_dir is None:
            base_dir = Path.cwd().parent
        else:
            base_dir = Path(base_dir).resolve()
        used_dataset = dataset_name
        yolo_dataset_root = base_dir / used_dataset
    
    used_split = split

    device = setup_environment(use_wandb=use_wandb)

    data_yaml_path = yolo_dataset_root / "data.yaml"

    data_config = load_data_config(data_yaml_path=data_yaml_path, yolo_dataset_root=yolo_dataset_root)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_testing_{run_timestamp}"
    
    # Choose directory based on include_training_exposure_analysis
    if include_training_exposure_analysis:
        runs_dir = base_dir / "yolo_test" / "analysis_runs"
    else:
        runs_dir = base_dir / "yolo_test" / "runs"
    
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
        labels_dir=dataset_info["labels_dir"],
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

    # Copy confusion matrices from YOLO validation output
    confusion_matrix_yolo_path, normalized_confusion_matrix_path = copy_normalized_confusion_matrix(
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

    # Generate failure analysis using expected counts (class_counts & object_count)
    # Always generate basic performance analysis; training exposure is optional
    print("\n" + "=" * 80)
    print("GENERATING DETAILED PERFORMANCE ANALYSIS")
    print("=" * 80)
    failure_analysis_summary = generate_failure_analysis(
        per_image_records=validation_results.per_image_records,
        image_attributes=dataset_info.get("image_attributes", {}),
        class_names=dataset_info["class_names"],
        test_run_dir=test_run_dir,
        dataset_root=yolo_dataset_root,
        iou_threshold=iou_threshold,
        include_training_exposure_analysis=include_training_exposure_analysis,
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
            include_training_exposure_analysis=include_training_exposure_analysis,
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
            "confusion_matrix": confusion_matrix_yolo_path,
            "confusion_matrix_normalized": normalized_confusion_matrix_path,
        },
        "comparison_data": comparison_data,
        "failure_analysis_summary": failure_analysis_summary,
        "yolo_validation_dir": test_run_dir / "yolo_validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO validation and generate report")
    parser.add_argument("--model-name", type=str, default="yolov8n", help="YOLO model name (e.g., yolov8n, yolov8s)")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Absolute path to dataset directory (if provided, overrides --dataset-name and --base-dir)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="bdd100k_yolo_limited",
        help="Dataset folder name under base directory (ignored if --dataset-path is provided)",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split: train, val, or test")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for validation")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base project directory (defaults to parent of current working dir, ignored if --dataset-path is provided)",
    )
    parser.add_argument(
        "--include-training-exposure",
        dest="include_training_exposure_analysis",
        action="store_true",
        default=False,
        help="Include training exposure vs test performance analysis",
    )
    args = parser.parse_args()

    run_validation_pipeline(
        model_name=args.model_name,
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        dataset_name=args.dataset_name,
        split=args.split,
        iou_threshold=args.iou,
        base_dir=Path(args.base_dir) if args.base_dir else None,
        use_wandb=True,
        save_reports=True,
        include_training_exposure_analysis=args.include_training_exposure_analysis,
    )


if __name__ == "__main__":
    main()
