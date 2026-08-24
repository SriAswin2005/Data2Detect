import os
import csv
import random
import yaml
from pathlib import Path
from PIL import Image

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    raw_dir = Path(config["paths"]["raw_data_dir"])
    cnn_dir = Path(config["paths"]["cnn_data_dir"])
    
    # Source path
    # The zip extract creates data/raw/GTSRB_Train/GTSRB/Final_Training/Images
    gtsrb_src = raw_dir / "GTSRB_Train" / "GTSRB" / "Final_Training" / "Images"
    
    if not gtsrb_src.exists():
        print(f"Error: GTSRB source directory not found at {gtsrb_src}.")
        print("Please run download_datasets.py first or verify the folder structure.")
        return
        
    print("Preprocessing GTSRB classification dataset...")
    
    # Clean up output directory first to prevent mixing training classes
    if cnn_dir.exists():
        print(f"Cleaning existing processed directory at {cnn_dir}...")
        import shutil
        shutil.rmtree(cnn_dir)
        
    # Prepare train and validation output splits
    splits = ["train", "validation"]
    for split in splits:
        for i in range(43):
            # Use 5-digit zero-padded folder names like '00000', '00001' to avoid any index alignment issues
            class_folder = cnn_dir / split / f"{i:05d}"
            class_folder.mkdir(parents=True, exist_ok=True)
            
    # Process each class directory (00000 to 00042)
    random.seed(42)
    for class_id in range(43):
        class_src_dir = gtsrb_src / f"{class_id:05d}"
        csv_file = class_src_dir / f"GT-{class_id:05d}.csv"
        
        if not class_src_dir.exists() or not csv_file.exists():
            print(f"Warning: Folder or CSV not found for class {class_id:05d}, skipping.")
            continue
            
        print(f"Processing class {class_id:05d}...")
        
        # Read the CSV annotations
        # Columns in GT-XXXXX.csv: Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
        records = []
        with open(csv_file, "r") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader) # skip header
            for row in reader:
                if len(row) < 8:
                    continue
                filename = row[0]
                roi_x1 = int(row[3])
                roi_y1 = int(row[4])
                roi_x2 = int(row[5])
                roi_y2 = int(row[6])
                records.append((filename, roi_x1, roi_y1, roi_x2, roi_y2))
                
        # Shuffle and split records (80% train, 20% validation)
        random.shuffle(records)
        split_idx = int(len(records) * 0.8)
        splits_map = {
            "train": records[:split_idx],
            "validation": records[split_idx:]
        }
        
        for split, records_split in splits_map.items():
            for filename, rx1, ry1, rx2, ry2 in records_split:
                ppm_path = class_src_dir / filename
                jpg_filename = Path(filename).stem + ".jpg"
                dst_path = cnn_dir / split / f"{class_id:05d}" / jpg_filename
                
                # Crop and resize image
                try:
                    with Image.open(ppm_path) as img:
                        # Crop to the exact Region of Interest (ROI)
                        cropped = img.crop((rx1, ry1, rx2, ry2))
                        # Resize to standard size (32x32)
                        resized = cropped.resize((32, 32), Image.Resampling.LANCZOS)
                        # Save as JPEG
                        resized.convert("RGB").save(dst_path, "JPEG")
                except Exception as e:
                    print(f"Error processing image {filename} in class {class_id}: {e}")
                    
    print("GTSRB classification dataset preprocessing complete.")

if __name__ == "__main__":
    main()
