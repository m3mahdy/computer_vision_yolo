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
) -> List[Path]:
    """Generate sample comparison images (ground truth vs predictions)."""
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
    
    sample_images = random.sample(valid_images, num_comparisons) if len(valid_images) > num_comparisons else valid_images
    print(f"\nGenerating {len(sample_images)} comparison figures...")
    
    comparison_paths = []
    
    for idx, img_path in enumerate(tqdm(sample_images, desc="Generating comparisons"), 1):
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # Run inference
        result = model(str(img_path), verbose=False, device=device)[0]
        
        # Draw ground truth and predictions
        gt_img, gt_count = draw_ground_truth(img_path, label_path, class_names, colors)
        pred_img = draw_predictions_with_consistent_colors(result, colors, class_names)
        pred_count = len(result.boxes)
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(gt_img)
        ax1.set_title(f"Ground Truth ({gt_count} objects)", fontweight="bold", fontsize=14)
        ax1.axis("off")
        
        ax2.imshow(pred_img)
        ax2.set_title(f"Prediction ({pred_count} objects)", fontweight="bold", fontsize=14)
        ax2.axis("off")
        
        fig.suptitle(f"Comparison #{idx}: {img_path.name}", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        comparison_path = comparisons_dir / f"comparison_{idx:02d}.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        comparison_paths.append(comparison_path)
    
    print(f"\u2713 Generated {len(comparison_paths)} comparison images")
    print(f"  Saved to: {comparisons_dir}")
    
    return comparison_paths


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
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    

    model = YOLO(str(model_path))
    info_values = model.info()
    keys = ["layers", "params", "size(MB)", "FLOPs(G)"]
    model_info: Dict[str, float] = {}
    for key, value in zip(keys, info_values):
        model_info[key] = value
    
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
) -> Tuple[Any, float]:
    print("\nRunning YOLO validation...")
    start_time = time.time()
    results = model.val(
        data=data_yaml_path,
        split=used_split,
        device=device,
        save_json=False,
        save_txt=False,
        conf=0.001,
        iou=iou_threshold,
        verbose=True,
        plots=True,
        project=str(test_run_dir),
        name="yolo_validation",
    )
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✓ YOLO validation completed in {total_time:.2f} seconds")
    return results, total_time


def extract_core_metrics(
    validation_results: Any,
    images_dir: Path,
    num_classes: int,
    class_names: Dict[int, str],
    model_info: Dict[str, float],
    total_time: float,
) -> Dict[str, Any]:
    num_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
    avg_inference_time = total_time / num_images if num_images > 0 else 0.0
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0

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

    confusion_matrix = (
        validation_results.confusion_matrix.matrix
        if hasattr(validation_results, "confusion_matrix")
        else np.zeros((num_classes, num_classes), dtype=int)
    )

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
) -> Tuple[Path, Path]:
    """Generate core metrics and mAP visualizations with enhanced formatting."""
    sns.set_style("whitegrid")

    # Figure 1: Core Metrics (Precision, Recall, F1-Score, Detection Outcomes)
    fig1, axes1 = plt.subplots(2, 2, figsize=(18, 12))
    ax_precision, ax_recall, ax_f1, ax_counts = axes1.flatten()

    # Precision by class
    precision_sorted = df_metrics.sort_values("Precision")
    ax_precision.barh(precision_sorted["Class"], precision_sorted["Precision"], color="#5BC0EB")
    ax_precision.set_title("Precision by Class", fontweight="bold", fontsize=18)
    ax_precision.set_xlabel("Precision", fontweight="bold", fontsize=14)
    ax_precision.set_xlim(0, 1)
    ax_precision.grid(axis="x", alpha=0.3)
    ax_precision.tick_params(axis='both', labelsize=14)

    # Recall by class
    recall_sorted = df_metrics.sort_values("Recall")
    ax_recall.barh(recall_sorted["Class"], recall_sorted["Recall"], color="#F25F5C")
    ax_recall.set_title("Recall by Class", fontweight="bold", fontsize=18)
    ax_recall.set_xlabel("Recall", fontweight="bold", fontsize=14)
    ax_recall.set_xlim(0, 1)
    ax_recall.grid(axis="x", alpha=0.3)
    ax_recall.tick_params(axis='both', labelsize=14)

    # F1-Score by class
    f1_sorted = df_metrics.sort_values("F1-Score")
    ax_f1.barh(f1_sorted["Class"], f1_sorted["F1-Score"], color="#9BC53D")
    ax_f1.set_title("F1-Score by Class", fontweight="bold", fontsize=18)
    ax_f1.set_xlabel("F1-Score", fontweight="bold", fontsize=14)
    ax_f1.set_xlim(0, 1)
    ax_f1.grid(axis="x", alpha=0.3)
    ax_f1.tick_params(axis='both', labelsize=14)

    # Detection outcomes bar chart
    bars = ax_counts.bar(["TP", "FP", "FN"], [total_tp, total_fp, total_fn], color=["#177E89", "#ED6A5A", "#F4A259"])
    ax_counts.set_title("Overall Detection Outcomes", fontweight="bold", fontsize=18)
    ax_counts.set_ylabel("Count", fontweight="bold", fontsize=14)
    ax_counts.grid(axis="y", alpha=0.3)
    ax_counts.tick_params(axis='both', labelsize=14)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax_counts.text(
            bar.get_x() + bar.get_width() / 2, height + max(total_tp, total_fp, total_fn) * 0.01,
            f"{int(height)}",
            ha="center",
            fontweight="bold",
            fontsize=14
        )

    plt.tight_layout()
    metrics_fig_path = test_run_dir / "core_metrics_charts.png"
    plt.savefig(metrics_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: mAP Metrics
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))
    ax_map, ax_overall = axes2.flatten()

    # mAP@0.5 by class
    map_sorted = df_metrics.sort_values("mAP@0.5")
    ax_map.barh(map_sorted["Class"], map_sorted["mAP@0.5"], color="#B388EB")
    ax_map.set_title("mAP@0.5 by Class", fontweight="bold", fontsize=18)
    ax_map.set_xlabel("mAP@0.5", fontweight="bold", fontsize=14)
    ax_map.set_xlim(0, 1)
    ax_map.grid(axis="x", alpha=0.3)
    ax_map.tick_params(axis='both', labelsize=14)

    # Overall metrics bar chart
    overall_plot_values = {
        "Precision": overall_precision,
        "Recall": overall_recall,
        "F1-Score": overall_f1,
        "mAP@0.5": yolo_metrics["map50"],
        "mAP@0.5:0.95": yolo_metrics["map50_95"],
    }
    bars = ax_overall.bar(overall_plot_values.keys(), overall_plot_values.values(), color="#FFA630")
    ax_overall.set_ylim(0, 1)
    ax_overall.set_title("Overall Metrics", fontweight="bold", fontsize=18)
    ax_overall.set_ylabel("Score", fontweight="bold", fontsize=14)
    ax_overall.grid(axis="y", alpha=0.3)
    ax_overall.tick_params(axis='both', labelsize=14)
    
    # Add value labels on bars
    for idx, (bar, value) in enumerate(zip(bars, overall_plot_values.values())):
        ax_overall.text(
            idx, value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
            fontsize=14
        )

    plt.tight_layout()
    map_fig_path = test_run_dir / "map_metrics_charts.png"
    plt.savefig(map_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return metrics_fig_path, map_fig_path


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    num_classes: int,
    class_names: Dict[int, str],
    model_name: str,
    test_run_dir: Path,
) -> Path:
    """
    Generate confusion matrix visualization.
    
    Matrix interpretation:
    - Rows (i): True class
    - Columns (j): Predicted class
    - confusion_matrix[i, j]: Count of true class i predicted as class j
    - Diagonal (i==j): Correct predictions (green)
    - Off-diagonal: Misclassifications (red)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw each cell with color coding
    for i in range(num_classes):
        for j in range(num_classes):
            value = confusion_matrix[i, j]
            if value == 0:
                cell_color = "white"
            elif i == j:
                cell_color = "#00A676"  # Correct predictions
            else:
                cell_color = "#D7263D"  # Misclassifications
            
            rect = Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=cell_color,
                edgecolor="black",
                linewidth=1.5
            )
            ax.add_patch(rect)
            
            # Add value text
            if value > 0:
                text_color = "white" if i == j else "#F7F7F7"
                ax.text(
                    j, i, str(int(value)),
                    ha="center", va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold"
                )

    # Set axis properties
    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_ylim(num_classes - 0.5, -0.5)
    ax.set_aspect("equal")

    # Set labels with increased font sizes
    class_labels = [class_names[i] for i in range(num_classes)]
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_labels, fontsize=10, fontweight="bold", rotation=45, ha="right")
    ax.set_yticklabels(class_labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontweight="bold", fontsize=14)
    ax.set_ylabel("True Class", fontweight="bold", fontsize=14)
    ax.set_title(f"Confusion Matrix ({model_name} validation)", fontweight="bold", fontsize=16)
    ax.grid(False)
    
    plt.tight_layout()

    confusion_matrix_path = test_run_dir / "confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return confusion_matrix_path


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
    comparison_image_paths: List[Path] | None = None,
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

    story.append(Paragraph("YOLO Validation Report", title_style))
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
        ["Total Execution Time", f"{total_time:.2f}s"],
        ["Average Inference Time", f"{metrics['avg_inference_time'] * 1000:.2f} ms per image"],
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
    story.append(Paragraph("Performance Visualizations", heading_style))

    core_metrics_path = test_run_dir / "core_metrics_charts.png"
    if core_metrics_path.exists():
        with PILImage.open(core_metrics_path) as img:
            w, h = img.size
            ratio = h / w
            pdf_w = 7 * inch
            pdf_h = pdf_w * ratio
            story.append(Image(str(core_metrics_path), width=pdf_w, height=pdf_h))

    map_metrics_path = test_run_dir / "map_metrics_charts.png"
    if map_metrics_path.exists():
        with PILImage.open(map_metrics_path) as img:
            w, h = img.size
            ratio = h / w
            pdf_w = 7 * inch
            pdf_h = pdf_w * ratio
            story.append(Image(str(map_metrics_path), width=pdf_w, height=pdf_h))

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

    story.append(PageBreak())
    story.append(Paragraph("Confusion Matrix", heading_style))
    story.append(Paragraph(f"Correct Predictions (Diagonal Sum): {int(np.trace(confusion_matrix))}", styles["Normal"]))
    story.append(Paragraph(f"Total Matched Predictions: {int(confusion_matrix.sum())}", styles["Normal"]))

    confusion_matrix_img_path = test_run_dir / "confusion_matrix.png"
    if confusion_matrix_img_path.exists():
        with PILImage.open(confusion_matrix_img_path) as img:
            w, h = img.size
            ratio = h / w
            pdf_w = 6.5 * inch
            pdf_h = pdf_w * ratio
            story.append(Image(str(confusion_matrix_img_path), width=pdf_w, height=pdf_h))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Additional validation plots available in: yolo_validation folder", styles["Normal"]))

    # Add sample comparisons section
    if comparison_image_paths:
        story.append(PageBreak())
        story.append(Paragraph("Sample Predictions: Ground Truth vs Model", heading_style))
        story.append(Spacer(1, 12))
        
        for idx, comp_path in enumerate(comparison_image_paths, 1):
            if comp_path.exists():
                try:
                    # Get image dimensions to fit on page
                    pil_img = PILImage.open(comp_path)
                    img_width, img_height = pil_img.size
                    
                    # Scale to fit page width (A4 width minus margins)
                    max_width = 7.5 * inch
                    max_height = 5 * inch
                    aspect = img_width / img_height
                    
                    if img_width > max_width:
                        img_width = max_width
                        img_height = max_width / aspect
                    
                    if img_height > max_height:
                        img_height = max_height
                        img_width = max_height * aspect
                    
                    img = Image(str(comp_path), width=img_width, height=img_height)
                    story.append(img)
                    story.append(Spacer(1, 12))
                    
                    # Add page break after every 2 comparisons to avoid crowding
                    if idx % 2 == 0 and idx < len(comparison_image_paths):
                        story.append(PageBreak())
                except Exception as e:
                    print(f"⚠️  Could not add comparison image {comp_path.name}: {e}")

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

    metrics_fig_path, map_fig_path = plot_core_and_map_metrics(
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
    comparison_image_paths = generate_sample_comparisons(
        model=model,
        valid_images=dataset_info["valid_images"],
        labels_dir=dataset_info["labels_dir"],
        class_names=dataset_info["class_names"],
        test_run_dir=test_run_dir,
        num_samples=6,
        device=device,
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
            comparison_image_paths=comparison_image_paths,
        )

    if use_wandb:
        try:
            wandb.finish()
            print("\n✓ Weights & Biases run completed successfully")
        except Exception as finish_error:
            print(f"\n⚠️  Error finishing W&B run: {finish_error}")

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
            "core_metrics": metrics_fig_path,
            "map_metrics": map_fig_path,
            "confusion_matrix": confusion_matrix_path,
        },
        "comparison_images": comparison_image_paths,
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
