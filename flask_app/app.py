import os
import sys
import uuid
import yaml
import cv2
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash

# Add parent directory of flask_app to python path to allow importing src
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.inference.pipeline import TrafficSignPipeline

app = Flask(__name__)
app.secret_key = "trafficsense_secret_key"

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import time

# Define and create upload/result folders
uploads_dir = Path(config["paths"]["uploads_dir"])
results_dir = Path(config["paths"]["results_dir"])
weights_dir = Path("weights")

uploads_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
weights_dir.mkdir(parents=True, exist_ok=True)

def cleanup_temp_files(max_age_seconds=3600):
    """Deletes files in uploads and results directories older than max_age_seconds."""
    now = time.time()
    for directory in [uploads_dir, results_dir]:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file():
                try:
                    file_age = now - os.path.getmtime(path)
                    if file_age > max_age_seconds:
                        path.unlink()
                except Exception as e:
                    print(f"Error cleaning up file {path}: {e}")

# Clean up any leftover files on startup
cleanup_temp_files()

# Initialize pipeline as None first
pipeline = None
weights_exist = False

def check_weights():
    global weights_exist
    yolo_w = Path(config["paths"]["yolo_weights"])
    cnn_w = Path(config["paths"]["cnn_weights"])
    weights_exist = yolo_w.exists() and cnn_w.exists()
    return weights_exist

# Try loading pipeline if weights are present
if check_weights():
    try:
        pipeline = TrafficSignPipeline(config_path="config.yaml")
        pipeline.load_models()
    except Exception as e:
        print(f"Error loading models on startup: {e}")
        pipeline = None

@app.route("/")
def index():
    # Check weights on every home load to see if they were recently added
    has_weights = check_weights()
    return render_template("index.html", has_weights=has_weights)

@app.route("/predict", methods=["POST"])
def predict():
    global pipeline
    # Clean up files older than 1 hour on new predictions
    cleanup_temp_files()
    
    if not check_weights():
        flash("Cannot process image: Trained weights are missing in the weights directory.", "danger")
        return redirect(url_for("index"))
        
    if "file" not in request.files:
        flash("No file part in the request.", "danger")
        return redirect(url_for("index"))
        
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file.", "danger")
        return redirect(url_for("index"))
        
    if file:
        # Load pipeline if not already loaded (delayed loading)
        if pipeline is None:
            try:
                pipeline = TrafficSignPipeline(config_path="config.yaml")
                pipeline.load_models()
            except Exception as e:
                flash(f"Error initializing models: {str(e)}", "danger")
                return redirect(url_for("index"))
                
        # Generate thread-safe unique filenames using UUID
        unique_id = uuid.uuid4().hex[:12]
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in [".jpg", ".jpeg", ".png"]:
            flash("Invalid file format. Please upload JPG, JPEG, or PNG.", "danger")
            return redirect(url_for("index"))
            
        orig_filename = f"upload_{unique_id}{file_extension}"
        result_filename = f"result_{unique_id}.jpg"
        
        orig_path = uploads_dir / orig_filename
        result_path = results_dir / result_filename
        
        # Save uploaded file
        file.save(str(orig_path))
        
        try:
            # Run inference pipeline
            annotated_img, detections = pipeline.predict(str(orig_path))
            
            # Save the annotated output image as JPEG
            cv2.imwrite(str(result_path), annotated_img)
            
            # Read original image to generate crops for the UI summary table
            img_bgr = cv2.imread(str(orig_path))
            if img_bgr is not None:
                for idx, det in enumerate(detections):
                    x1, y1, x2, y2 = det["bbox"]
                    # Boundary safe crop
                    crop_img = img_bgr[y1:y2, x1:x2]
                    if crop_img.size > 0:
                        crop_filename = f"crop_{unique_id}_{idx}.jpg"
                        crop_path = results_dir / crop_filename
                        cv2.imwrite(str(crop_path), crop_img)
                        det["crop_url"] = url_for("static", filename=f"results/{crop_filename}")
                    else:
                        det["crop_url"] = None
            
            # Generate static relative URLs for rendering in HTML
            orig_url = url_for("static", filename=f"uploads/{orig_filename}")
            result_url = url_for("static", filename=f"results/{result_filename}")
            
            return render_template(
                "result.html",
                orig_url=orig_url,
                result_url=result_url,
                detections=detections,
                num_detections=len(detections)
            )
            
        except Exception as e:
            flash(f"Error during image processing: {str(e)}", "danger")
            return redirect(url_for("index"))

if __name__ == "__main__":
    # Run the server locally, bind to all interfaces to allow local network testing
    app.run(host="0.0.0.0", port=5000, debug=True)
