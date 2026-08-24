import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Setup paths to import pipeline
workspace_root = Path(__file__).resolve().parents[2]
sys.path.append(str(workspace_root))

from src.inference.pipeline import TrafficSignPipeline

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def calculate_iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0
        
    return interArea / float(unionArea)

def main():
    config = load_config()
    
    # Locate gt.txt annotations file
    gt_candidates = list(Path(".").glob("**/gt.txt"))
    if len(gt_candidates) == 0:
        print("Error: Could not locate gt.txt annotations file anywhere in the workspace.")
        return
    gt_file = gt_candidates[0]
    
    valid_images_dir = Path("data/detection/valid/images")
    if not valid_images_dir.exists():
        print(f"Error: Validation images folder not found at {valid_images_dir.absolute()}.")
        print("Please run preprocess_detection.py first.")
        return
        
    # Read validation images file names
    valid_filenames = {p.name for p in valid_images_dir.glob("*.jpg")}
    if len(valid_filenames) == 0:
        print("Error: No preprocessed JPEG images found in validation folder.")
        return
        
    # Parse annotations from gt.txt
    # Format: filename;x1;y1;x2;y2;class_id
    annotations = {}
    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 6:
                continue
            
            ppm_name = parts[0]
            jpg_name = Path(ppm_name).stem + ".jpg"
            
            # We only evaluate on validation split images
            if jpg_name not in valid_filenames:
                continue
                
            x1, y1, x2, y2 = map(int, parts[1:5])
            class_id = int(parts[5])
            
            if jpg_name not in annotations:
                annotations[jpg_name] = []
            annotations[jpg_name].append({
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id
            })

    print("Initializing full end-to-end inference pipeline...")
    pipeline = TrafficSignPipeline(config_path="config.yaml")
    pipeline.load_models()
    
    total_gt_signs = 0
    total_pred_signs = 0
    true_positives = 0  # Bbox overlap >= 0.5 AND class matches
    classification_correct = 0
    localization_correct = 0  # Bbox overlap >= 0.5 (regardless of class)
    
    # Count total ground truth signs in validation set
    for filename, gt_list in annotations.items():
        total_gt_signs += len(gt_list)
        
    print(f"\nEvaluating pipeline on {len(annotations)} validation images containing {total_gt_signs} ground truth signs...")
    
    for i, (filename, gt_list) in enumerate(annotations.items(), 1):
        img_path = valid_images_dir / filename
        
        try:
            # Run prediction through full pipeline (YOLO localization + padded cropping + CNN classification)
            _, detections = pipeline.predict(str(img_path))
            total_pred_signs += len(detections)
            
            # Keep track of matched ground truth signs to prevent double-matching
            matched_gt_indices = set()
            
            for pred in detections:
                pred_bbox = pred["bbox"]
                pred_class = pred["class_id"]
                
                best_iou = 0
                best_gt_idx = -1
                
                # Find matching ground truth sign
                for gt_idx, gt in enumerate(gt_list):
                    if gt_idx in matched_gt_indices:
                        continue
                    iou = calculate_iou(pred_bbox, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # If overlap is high enough (IoU >= 0.5), it's a localization match
                if best_iou >= 0.5:
                    matched_gt_indices.add(best_gt_idx)
                    localization_correct += 1
                    
                    # If the CNN class classification is also correct, it's a true positive
                    if pred_class == gt_list[best_gt_idx]["class_id"]:
                        true_positives += 1
                        classification_correct += 1
                        
        except Exception as e:
            print(f"Error evaluating image {filename}: {e}")
            
        if i % 30 == 0 or i == len(annotations):
            print(f"  Processed {i}/{len(annotations)} images...")

    # Calculate overall pipeline metrics
    precision = true_positives / total_pred_signs if total_pred_signs > 0 else 0
    recall = true_positives / total_gt_signs if total_gt_signs > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Classifier performance on correctly localized bounding boxes
    loc_acc = classification_correct / localization_correct if localization_correct > 0 else 0
    
    print("\n=======================================================")
    print("--- COMBINED PIPELINE END-TO-END EVALUATION METRICS ---")
    print(f"Total Validation Images Evaluated: {len(annotations)}")
    print(f"Total Bounding Boxes in Ground Truth: {total_gt_signs}")
    print(f"Total Bounding Boxes Predicted:      {total_pred_signs}")
    print(f"True Positives (Correct Box + Class): {true_positives}")
    print("-------------------------------------------------------")
    print(f"End-to-End Pipeline Precision:       {precision*100:.2f}%")
    print(f"End-to-End Pipeline Recall:          {recall*100:.2f}%")
    print(f"End-to-End Pipeline F1-Score:        {f1_score*100:.2f}%")
    print(f"CNN Accuracy on Localized Boxes:     {loc_acc*100:.2f}%")
    print("=======================================================")

if __name__ == "__main__":
    main()
