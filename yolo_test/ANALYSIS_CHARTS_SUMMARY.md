# Comprehensive Analysis Charts Summary

## All Analysis Charts Generated and Included in PDF Report

### ✅ Status: COMPLETE
All 12 analysis charts are automatically generated as PNG images and integrated into the PDF report.

---

## 1. Environmental Attribute Analysis (3 Charts)

### Chart 1: Accuracy by Weather
- **File**: `accuracy_by_weather.png`
- **Type**: Horizontal bar chart
- **Shows**: Prediction accuracy for each weather condition (clear, rainy, snowy, etc.)
- **PDF Section**: "Accuracy by Environmental Attributes"
- **CSV**: `accuracy_by_weather.csv`

### Chart 2: Accuracy by Scene  
- **File**: `accuracy_by_scene.png`
- **Type**: Horizontal bar chart
- **Shows**: Prediction accuracy for each scene type (highway, city street, residential, etc.)
- **PDF Section**: "Accuracy by Environmental Attributes"
- **CSV**: `accuracy_by_scene.csv`

### Chart 3: Accuracy by Time of Day
- **File**: `accuracy_by_timeofday.png`
- **Type**: Horizontal bar chart
- **Shows**: Prediction accuracy for each time period (daytime, dawn/dusk, night)
- **PDF Section**: "Accuracy by Environmental Attributes"
- **CSV**: `accuracy_by_timeofday.csv`

---

## 2. Object Characteristics Analysis (2 Charts)

### Chart 4: Accuracy by Object Count
- **File**: `accuracy_by_object_count.png`
- **Type**: Vertical bar chart
- **Shows**: Accuracy by number of objects per image (1-5, 6-10, 11-20, 21-50, 50+)
- **PDF Section**: "Accuracy by Object Characteristics"
- **CSV**: `accuracy_by_object_count.csv`

### Chart 5: Accuracy by Object Size (Scale/Distance)
- **File**: `accuracy_by_size.png`
- **Type**: Vertical bar chart
- **Shows**: Accuracy by object size (small <1%, medium 1-5%, large >5% of image)
- **Identifies**: Whether model fails on small/distant objects
- **PDF Section**: "Accuracy by Object Characteristics"
- **CSV**: `accuracy_by_size.csv`

---

## 3. Training Exposure vs Test Performance (3 Charts)

### Chart 6: Train vs Test Class Distribution
- **File**: `train_test_distribution.png`
- **Type**: Side-by-side bar chart
- **Shows**: Object count per class in train split vs test split
- **Identifies**: Class imbalance between splits
- **PDF Section**: "Training Exposure vs Test Performance"
- **CSV**: `train_test_class_comparison.csv`

### Chart 7: Test Accuracy vs Training Exposure
- **File**: `accuracy_vs_training_exposure.png`
- **Type**: Scatter plot with trend line
- **Shows**: Correlation between training object count and test accuracy
- **Bubble size**: Test object count
- **Labels**: Each class name
- **PDF Section**: "Training Exposure vs Test Performance"
- **CSV**: `train_test_class_comparison.csv`

### Chart 8: Accuracy vs Train/Test Ratio
- **File**: `accuracy_vs_train_test_ratio.png`
- **Type**: Two-panel chart
  - Left: Horizontal bars colored by train/test ratio
  - Right: Scatter showing ratio vs accuracy
- **Shows**: Whether underrepresented classes in training perform worse
- **PDF Section**: "Training Exposure vs Test Performance"
- **CSV**: `train_test_class_comparison.csv`

---

## 4. Per-Class Performance Breakdowns (4 Heatmaps)

### Chart 9: Per-Class Accuracy by Size
- **File**: `accuracy_by_class_and_size.png`
- **Type**: Heatmap (RdYlGn colormap)
- **Shows**: Accuracy for each class at each size level (small/medium/large)
- **Identifies**: Which classes struggle with small objects
- **PDF Section**: "Per-Class Performance Analysis"
- **CSV**: `accuracy_by_class_and_size.csv`

### Chart 10: Per-Class Accuracy by Weather (Top 5 Classes)
- **File**: `accuracy_by_class_and_weather.png`
- **Type**: Heatmap (RdYlGn colormap)
- **Shows**: Accuracy for top 5 classes across weather conditions
- **Identifies**: Class-specific weather vulnerabilities
- **PDF Section**: "Per-Class Performance Analysis"
- **CSV**: `accuracy_by_class_and_weather.csv`

### Chart 11: Per-Class Accuracy by Scene (Top 5 Classes)
- **File**: `accuracy_by_class_and_scene.png`
- **Type**: Heatmap (RdYlGn colormap)
- **Shows**: Accuracy for top 5 classes across scene types
- **Identifies**: Class-specific scene performance
- **PDF Section**: "Per-Class Performance Analysis"
- **CSV**: `accuracy_by_class_and_scene.csv`

### Chart 12: Per-Class Accuracy by Time of Day (Top 5 Classes)
- **File**: `accuracy_by_class_and_timeofday.png`
- **Type**: Heatmap (RdYlGn colormap)
- **Shows**: Accuracy for top 5 classes across different times
- **Identifies**: Class-specific temporal performance
- **PDF Section**: "Per-Class Performance Analysis"
- **CSV**: `accuracy_by_class_and_timeofday.csv`

---

## PDF Report Structure

The complete PDF report includes these sections in order:

1. **Model Information & Configuration**
2. **Inference Performance Metrics**
3. **Overall Accuracy Metrics**
4. **Confusion Matrix**
5. **Performance Visualizations** (Precision, Recall, F1, mAP by class)
6. **Per-Class Performance Table**
7. **Comprehensive Failure Analysis** ⭐ NEW
   - Overall summary table
   - Environmental attributes (Charts 1-3)
   - Object characteristics (Charts 4-5)
   - Training exposure analysis (Charts 6-8)
   - Per-class breakdowns (Charts 9-12)
8. **Sample Predictions** (Ground truth vs model with attributes)

---

## Data Outputs

### CSV Files (9 files)
1. `per_image_accuracy.csv` - Per-image breakdown with all attributes
2. `accuracy_by_weather.csv` - Aggregated by weather
3. `accuracy_by_scene.csv` - Aggregated by scene
4. `accuracy_by_timeofday.csv` - Aggregated by time of day
5. `accuracy_by_object_count.csv` - Aggregated by object count ranges
6. `accuracy_by_size.csv` - Aggregated by object size buckets
7. `accuracy_by_class_and_size.csv` - Per-class × size matrix
8. `accuracy_by_class_and_weather.csv` - Per-class × weather matrix
9. `accuracy_by_class_and_scene.csv` - Per-class × scene matrix
10. `accuracy_by_class_and_timeofday.csv` - Per-class × time matrix
11. `train_test_class_comparison.csv` - Train vs test comparison

### JSON Output
- `failure_analysis_comprehensive.json` - Complete analysis summary with all metrics and chart paths

---

## Console Output Example

```
================================================================================
COMPREHENSIVE FAILURE ANALYSIS
Analyzing relationship between attributes and prediction accuracy...
================================================================================

Loading training split metadata for exposure analysis...
✓ Training metadata loaded: 70000 images, 567890 objects

...

================================================================================
ANALYSIS SUMMARY
================================================================================
Overall Accuracy: 87.32%
Total Images: 2000
Expected Objects: 45678
Matched Objects: 39876

Weakest Weather Conditions:
  - snowy: 72.14% (45 images)
  - rainy: 81.53% (120 images)

Weakest Scenes:
  - highway: 78.91% (200 images)
  - parking lot: 84.22% (50 images)

Weakest Times of Day:
  - dawn/dusk: 79.34% (150 images)
  - night: 82.15% (100 images)

Accuracy by Object Size (Scale/Distance):
  - small: 68.52% (1245 objects)
  - medium: 82.34% (897 objects)
  - large: 91.78% (432 objects)

Train-Test Comparison (Classes with Lowest Accuracy):
  - pedestrian: 71.23% accuracy | Train: 1245 objs, Test: 89 objs | Ratio: 14.0x
  - traffic light: 68.54% accuracy | Train: 450 objs, Test: 67 objs | Ratio: 6.7x
  - truck: 75.32% accuracy | Train: 892 objs, Test: 45 objs | Ratio: 19.8x

✓ Comprehensive failure analysis complete
  - Charts saved: 12
  - CSV files saved: 11
  - JSON summary: .../failure_analysis_comprehensive.json
================================================================================
```

---

## Chart Specifications

### Visual Design
- **Resolution**: 200 DPI for all charts
- **Color Schemes**:
  - Environmental attributes: Blue (#5BC0EB), Red (#F25F5C), Green (#9BC53D)
  - Size analysis: Red-Orange-Teal gradient
  - Heatmaps: RdYlGn (Red-Yellow-Green) for immediate identification of weak areas
  - Train-test: Blue (train) and Red (test)
- **Font**: Bold titles (16pt), axis labels (14pt), annotations (9-10pt)
- **Layout**: Clean, professional with gridlines for readability

### PDF Integration
- All charts automatically resized to fit page width (5.5-6.5 inches)
- Proper aspect ratios maintained
- Strategic page breaks for optimal reading flow
- Clear section headings with consistent styling

---

## Technical Implementation

- **Generate charts**: `generate_failure_analysis()` function
- **Save as PNG**: All charts saved to test run directory
- **Add to PDF**: `generate_pdf_and_json_report()` function
- **Automatic**: No manual intervention required
- **Conditional**: Only generates charts when data is available
- **Error handling**: Gracefully handles missing data

---

## Usage

Run the script normally:
```bash
python yolo_test/run_yolo_detailed_testing_report.py --model-name yolov8m --dataset-name bdd100k_yolo_tiny --split test
```

All charts are automatically:
1. Generated during analysis
2. Saved as PNG files
3. Included in the PDF report
4. Referenced in JSON summary
5. Listed in console output

The complete analysis typically takes 2-5 minutes depending on dataset size.
