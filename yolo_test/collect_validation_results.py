"""
Script to collect validation results from JSON files in runs directory.

This script scans the yolo_test/runs directory for metrics_data.json files
and loads them into the same format as the multi_model_yolo_report notebook
uses for results_summary and validation_results.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import argparse


def load_json_from_run(run_dir: Path) -> Dict[str, Any] | None:
    """
    Load metrics_data.json from a run directory.
    
    Args:
        run_dir: Path to the run directory
        
    Returns:
        Dictionary containing the metrics data, or None if file not found
    """
    json_file = run_dir / "metrics_data.json"
    
    if not json_file.exists():
        return None
    
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"⚠️  Error loading {json_file}: {e}")
        return None


def convert_to_result_format(
    json_data: Dict[str, Any],
    run_dir: Path
) -> Dict[str, Any]:
    """
    Convert JSON data to the format expected by validation_results.
    
    Args:
        json_data: Data loaded from metrics_data.json
        run_dir: Path to the run directory
        
    Returns:
        Dictionary in the format used by validation_results
    """
    metadata = json_data.get("metadata", {})
    model_info = json_data.get("model_info", {})
    performance = json_data.get("performance", {})
    custom_metrics = json_data.get("custom_metrics", {})
    yolo_metrics = json_data.get("yolo_official_metrics", {})
    confusion_matrix_data = json_data.get("confusion_matrix", {})
    
    # Build the result structure matching the notebook format
    result = {
        "model_name": metadata.get("model_name", "unknown"),
        "run_name": metadata.get("run_name", "unknown"),
        "wb_run_name": metadata.get("wb_run_name", "unknown"),
        "run_dir": run_dir,
        "model_info": {
            "params": model_info.get("parameters", 0),
            "size(MB)": model_info.get("model_size_mb", 0.0),
            "FLOPs(G)": model_info.get("flops_gflops", 0.0),
        },
        "total_time": performance.get("total_time_seconds", 0.0),
        "metrics": {
            "num_images": performance.get("images_processed", 0),
            "avg_inference_time": performance.get("avg_inference_time_ms", 0.0) / 1000.0,  # Convert ms to seconds
            "fps": performance.get("fps", 0.0),
            "yolo_metrics": yolo_metrics.get("overall", {}),
            "yolo_class_metrics": yolo_metrics.get("per_class", {}),
            "overall": custom_metrics.get("overall", {}),
            "per_class": custom_metrics.get("per_class", {}),
            "confusion_matrix": confusion_matrix_data.get("matrix", []),
        },
    }
    
    return result


def collect_validation_results(
    runs_dir: Path,
    analysis_runs_dir: Path | None = None,
    dataset_name: str | None = None,
    split: str | None = None,
    verbose: bool = True
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Collect all validation results from runs and analysis_runs directories.
    
    Args:
        runs_dir: Path to the runs directory
        analysis_runs_dir: Path to the analysis_runs directory (optional, will auto-detect if None)
        dataset_name: Optional filter by dataset name
        split: Optional filter by split (train, val, test)
        verbose: Whether to print progress messages (default: True)
        
    Returns:
        Tuple of (results_summary, validation_results)
    """
    results_summary: List[Dict[str, Any]] = []
    validation_results: Dict[str, Dict[str, Any]] = {}
    
    # Auto-detect analysis_runs_dir if not provided
    if analysis_runs_dir is None:
        analysis_runs_dir = runs_dir.parent / "analysis_runs"
    
    # Collect directories to scan
    dirs_to_scan = []
    
    if runs_dir.exists():
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        dirs_to_scan.extend(run_dirs)
        if verbose:
            print(f"📁 Found {len(run_dirs)} directories in runs/")
    else:
        if verbose:
            print(f"⚠️  Runs directory not found: {runs_dir}")
    
    if analysis_runs_dir.exists():
        analysis_dirs = [d for d in analysis_runs_dir.iterdir() if d.is_dir()]
        dirs_to_scan.extend(analysis_dirs)
        if verbose:
            print(f"📁 Found {len(analysis_dirs)} directories in analysis_runs/")
    else:
        if verbose:
            print(f"⚠️  Analysis runs directory not found: {analysis_runs_dir}")
    
    if not dirs_to_scan:
        if verbose:
            print(f"❌ No run directories found")
        return results_summary, validation_results
    
    if verbose:
        print(f"📁 Scanning {len(dirs_to_scan)} total run directories...")
    
    for run_dir in sorted(dirs_to_scan):
        json_data = load_json_from_run(run_dir)
        
        if json_data is None:
            continue
        
        metadata = json_data.get("metadata", {})
        
        # Apply filters if specified
        if dataset_name and metadata.get("dataset") != dataset_name:
            continue
        
        if split and metadata.get("data_split") != split:
            continue
        
        # Convert to result format
        result = convert_to_result_format(json_data, run_dir)
        
        # Extract data for results_summary
        model_name = metadata.get("model_name", "unknown")
        iou_threshold = metadata.get("iou_threshold", 0.5)
        
        overall = result["metrics"]["overall"]
        yolo_overall = result["metrics"]["yolo_metrics"]
        
        results_summary.append({
            "model_name": model_name,
            "dataset": metadata.get("dataset", "unknown"),
            "split": metadata.get("data_split", "unknown"),
            "iou": iou_threshold,
            "precision_confusion": overall.get("precision", 0.0),
            "recall_confusion": overall.get("recall", 0.0),
            "f1_confusion": overall.get("f1_score", 0.0),
            "precision_yolo": yolo_overall.get("precision", 0.0),
            "recall_yolo": yolo_overall.get("recall", 0.0),
            "map50": yolo_overall.get("map50", 0.0),
            "map50_95": yolo_overall.get("map50_95", 0.0),
            "params_m": result["model_info"]["params"] / 1e6,
            "size_mb": result["model_info"]["size(MB)"],
            "fps": result["metrics"]["fps"],
            "status": "ok",
            "run_dir": str(result["run_dir"]),
        })
        
        # Store in validation_results with model name as key
        validation_results[model_name] = result
        
        if verbose:
            print(f"✓ Loaded: {model_name} (Run: {run_dir.name})")
    
    if verbose:
        print(f"\n✓ Collected {len(results_summary)} validation results")
    
    return results_summary, validation_results


def print_summary(results_summary: List[Dict[str, Any]]) -> None:
    """
    Print a summary table of collected results.
    
    Args:
        results_summary: List of result summaries
    """
    if not results_summary:
        print("\nNo results found.")
        return
    
    print("\n" + "=" * 120)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 120)
    print(f"{'Model':<15} {'Dataset':<20} {'Split':<8} {'IoU':<6} {'Prec(C)':<9} {'Rec(C)':<9} {'F1(C)':<9} {'mAP@0.5':<9} {'mAP@0.5:0.95':<14} {'FPS':<8}")
    print("-" * 120)
    
    for result in results_summary:
        print(
            f"{result['model_name']:<15} "
            f"{result['dataset']:<20} "
            f"{result['split']:<8} "
            f"{result['iou']:<6.2f} "
            f"{result['precision_confusion']:<9.4f} "
            f"{result['recall_confusion']:<9.4f} "
            f"{result['f1_confusion']:<9.4f} "
            f"{result['map50']:<9.4f} "
            f"{result['map50_95']:<14.4f} "
            f"{result['fps']:<8.2f}"
        )
    
    print("=" * 120)


def save_collected_results(
    results_summary: List[Dict[str, Any]],
    validation_results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Save collected results to JSON files.
    
    Args:
        results_summary: List of result summaries
        validation_results: Dictionary of validation results
        output_dir: Directory to save output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results_summary
    summary_file = output_dir / "results_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"\n✓ Saved results summary: {summary_file}")
    
    # Save validation_results
    validation_file = output_dir / "validation_results.json"
    with open(validation_file, "w") as f:
        json.dump(validation_results, f, indent=2, default=str)
    print(f"✓ Saved validation results: {validation_file}")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Collect validation results from runs directory"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Path to runs directory (default: ./runs)",
    )
    parser.add_argument(
        "--analysis-runs-dir",
        type=str,
        default=None,
        help="Path to analysis_runs directory (default: auto-detect from runs-dir parent)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter by dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Filter by split (train, val, test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save collected results (default: ./collected_results)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to files",
    )
    
    args = parser.parse_args()
    
    # Determine paths
    if args.runs_dir:
        runs_dir = Path(args.runs_dir).resolve()
    else:
        runs_dir = Path(__file__).parent / "runs"
    
    if args.analysis_runs_dir:
        analysis_runs_dir = Path(args.analysis_runs_dir).resolve()
    else:
        analysis_runs_dir = None  # Will auto-detect in collect_validation_results
    
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = runs_dir.parent / "collected_results"
    
    print("\n" + "=" * 80)
    print("COLLECTING VALIDATION RESULTS")
    print("=" * 80)
    print(f"Runs directory: {runs_dir}")
    if analysis_runs_dir:
        print(f"Analysis runs directory: {analysis_runs_dir}")
    else:
        print(f"Analysis runs directory: {runs_dir.parent / 'analysis_runs'} (auto-detect)")
    if args.dataset:
        print(f"Filter by dataset: {args.dataset}")
    if args.split:
        print(f"Filter by split: {args.split}")
    print("=" * 80)
    
    # Collect results
    results_summary, validation_results = collect_validation_results(
        runs_dir=runs_dir,
        analysis_runs_dir=analysis_runs_dir,
        dataset_name=args.dataset,
        split=args.split
    )
    
    # Print summary
    print_summary(results_summary)
    
    # Save results if requested
    if not args.no_save:
        save_collected_results(
            results_summary=results_summary,
            validation_results=validation_results,
            output_dir=output_dir
        )
    
    print(f"\n✓ Collection complete!")
    print(f"  Total runs: {len(results_summary)}")
    print(f"  Models: {', '.join(sorted(validation_results.keys()))}")


if __name__ == "__main__":
    main()
