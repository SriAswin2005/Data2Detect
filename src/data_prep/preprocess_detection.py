import os
import shutil
import random
import yaml
from pathlib import Path
from PIL import Image

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def convert_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    # Calculate YOLO normalized coordinates
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return xc * dw, yc * dh, w * dw, h * dh

def main():
    config = load_config()
    raw_dir = Path(config["paths"]["raw_data_dir"])
    yolo_dir = Path(config["paths"]["yolo_data_dir"])
    
    # Define source paths
    # The zip extract creates data/raw/GTSDB/FullIJCNN2013
    gtsdb_src = raw_dir / "GTSDB" / "FullIJCNN2013"
    gt_file = gtsdb_src / "gt.txt"
    
    if not gtsdb_src.exists() or not gt_file.exists():
        print(f"Error: GTSDB source directory or gt.txt not found at {gtsdb_src}.")
        print("Please run download_datasets.py first or verify the folder structure.")
        return
        
    print("Preprocessing GTSDB for 1-class YOLOv8 localization...")
    
    # Initialize YOLO output directories
    splits = ["train", "valid"]
    for split in splits:
        (yolo_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
    # Read annotations
    # Format of gt.txt: filename;x1;y1;x2;y2;class_id
    annotations = {}
    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 6:
                continue
            filename = parts[0]
            x1 = float(parts[1])
            y1 = float(parts[2])
            x2 = float(parts[3])
            y2 = float(parts[4])
            class_id = int(parts[5])
            
            if filename not in annotations:
                annotations[filename] = []
            annotations[filename].append((x1, y1, x2, y2, class_id))
            
    # Find all PPM images
    all_ppm = list(gtsdb_src.glob("*.ppm"))
    print(f"Found {len(all_ppm)} images in source directory.")
    
    # Shuffle and split (80% train, 20% validation)
    random.seed(42)
    random.shuffle(all_ppm)
    split_idx = int(len(all_ppm) * 0.8)
    train_images = all_ppm[:split_idx]
    valid_images = all_ppm[split_idx:]
    
    image_splits = {
        "train": train_images,
        "valid": valid_images
    }
    
    for split, images in image_splits.items():
        print(f"Processing {split} split ({len(images)} images)...")
        for ppm_path in images:
            filename = ppm_path.name
            jpg_filename = ppm_path.stem + ".jpg"
            
            # Target paths
            dst_img_path = yolo_dir / split / "images" / jpg_filename
            dst_lbl_path = yolo_dir / split / "labels" / (ppm_path.stem + ".txt")
            
            # Convert PPM to JPG and save
            try:
                with Image.open(ppm_path) as img:
                    img_w, img_h = img.size
                    img.convert("RGB").save(dst_img_path, "JPEG")
            except Exception as e:
                print(f"Error converting {filename}: {e}")
                continue
                
            # If the image has annotations, write YOLO label file (all classes mapped to 0)
            if filename in annotations:
                with open(dst_lbl_path, "w") as out_f:
                    for (x1, y1, x2, y2, _) in annotations[filename]:
                        xc, yc, w, h = convert_bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
                        # Write class 0 (traffic_sign) followed by YOLO coordinates
                        out_f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            else:
                # Create an empty label file for images with no signs (background images)
                open(dst_lbl_path, "w").close()
                
    # Create data.yaml
    data_yaml_content = {
        "path": str(yolo_dir.absolute().as_posix()),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": {
            0: "traffic_sign"
        }
    }
    
    with open(yolo_dir / "data.yaml", "w") as yaml_f:
        yaml.dump(data_yaml_content, yaml_f, default_flow_style=False)
        
    print(f"YOLO dataset generation complete. Saved data.yaml configuration in {yolo_dir}.")

if __name__ == "__main__":
    main()
