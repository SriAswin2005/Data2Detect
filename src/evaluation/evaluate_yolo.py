import os
import yaml
from pathlib import Path
from ultralytics import YOLO

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    weights_path = Path(config["paths"]["yolo_weights"])
    yolo_data_yaml = Path(config["paths"]["yolo_data_dir"]) / "data.yaml"
    
    if not weights_path.exists():
        print(f"Error: YOLO weights file not found at {weights_path.absolute()}.")
        print("Please train the 1-class YOLOv8 model on Google Colab first,")
        print("download the best.pt weights, rename them to yolo_best.pt,")
        print(f"and place them in the {weights_path.parent} directory.")
        return
        
    if not yolo_data_yaml.exists():
        print(f"Error: YOLO dataset configuration file not found at {yolo_data_yaml.absolute()}.")
        print("Please run preprocess_detection.py first to generate the dataset and data.yaml.")
        return

    print(f"Loading YOLOv8 model weights from {weights_path}...")
    model = YOLO(weights_path)
    
    print("Running validation evaluation on test/validation set...")
    # Run validation. By default, it uses the split specified by 'val' in data.yaml
    metrics = model.val(data=str(yolo_data_yaml))
    
    print("\n--- YOLOv8 1-Class Detection Evaluation Metrics ---")
    print(f"mAP@0.5: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP@0.5:0.95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.4f}")
    print("---------------------------------------------------")

if __name__ == "__main__":
    main()
