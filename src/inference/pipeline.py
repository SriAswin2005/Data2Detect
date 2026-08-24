import os
import cv2
import yaml
import numpy as np
from pathlib import Path
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO

# Add parent directory of flask_app to python path to allow importing src
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.cnn_architecture import create_cnn_model

class TrafficSignPipeline:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.yolo_weights = Path(self.config["paths"]["yolo_weights"])
        self.cnn_weights = Path(self.config["paths"]["cnn_weights"])
        
        self.yolo_conf = self.config["inference"]["yolo_conf_threshold"]
        self.cnn_conf_threshold = self.config["inference"]["cnn_conf_threshold"]
        self.cnn_img_size = self.config["inference"]["cnn_img_size"]
        
        self.keras_class_map = self.config["keras_class_map"]
        self.class_names = self.config["class_names"]
        
        self.yolo_model = None
        self.cnn_model = None
        self.models_loaded = False

    def load_models(self):
        """Loads both YOLO detection and Keras classification weights."""
        if not self.yolo_weights.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {self.yolo_weights.absolute()}. "
                "Please train the model on Google Colab and place the weights file."
            )
        if not self.cnn_weights.exists():
            raise FileNotFoundError(
                f"CNN weights not found at {self.cnn_weights.absolute()}. "
                "Please train the model on Google Colab and place the weights file."
            )
            
        print("Loading YOLOv8 detector...")
        self.yolo_model = YOLO(self.yolo_weights)
        
        print("Loading Custom CNN classifier...")
        try:
            self.cnn_model = tf.keras.models.load_model(str(self.cnn_weights))
        except Exception:
            print("Rebuilding CNN architecture from definition file and loading weights...")
            self.cnn_model = create_cnn_model()
            self.cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.cnn_model.load_weights(str(self.cnn_weights))
            
        self.models_loaded = True
        print("All models loaded successfully.")

    def pad_to_square(self, img):
        """
        Pads the shorter dimension of the image with black borders 
        to make it square, preserving the aspect ratio.
        """
        h, w, _ = img.shape
        if h == w:
            return img
            
        max_dim = max(h, w)
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        
        if h < max_dim:
            pad_total = max_dim - h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
        elif w < max_dim:
            pad_total = max_dim - w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            
        padded_img = cv2.copyMakeBorder(
            img, 
            pad_top, 
            pad_bottom, 
            pad_left, 
            pad_right, 
            borderType=cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        return padded_img

    def predict(self, image_path):
        """
        Runs the full detection and classification pipeline on an image.
        Uses batched predictions to classify all detected signs in a single call.
        """
        if not self.models_loaded:
            self.load_models()
            
        # Read the image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image from path: {image_path}")
            
        img_h, img_w, _ = img.shape
        
        # Step 1: Detect signs using YOLOv8 (1-class detector)
        yolo_results = self.yolo_model.predict(
            source=img,
            conf=self.yolo_conf,
            verbose=False
        )
        
        detections = []
        annotated_img = img.copy()
        
        # Gather all valid proposals and crop them
        crops_list = []
        bboxes_list = []
        
        if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes
            
            for box in boxes:
                # Extract coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Constrain coordinates to image dimensions
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                
                # Check for minimum valid crop dimensions
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                    
                # Crop and apply aspect-ratio preserving padding
                crop = img[y1:y2, x1:x2]
                padded_crop = self.pad_to_square(crop)
                
                # Convert BGR (OpenCV) to RGB (CNN target format)
                crop_rgb = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2RGB)
                # Resize to standard size (32x32)
                crop_resized = cv2.resize(crop_rgb, (self.cnn_img_size, self.cnn_img_size))
                # Normalize pixel values
                crop_norm = crop_resized.astype(np.float32) / 255.0
                
                crops_list.append(crop_norm)
                bboxes_list.append((x1, y1, x2, y2))
                
        # Run CNN predictions in batch
        if len(crops_list) > 0:
            crops_batch = np.array(crops_list)
            # Single parallel prediction call for all crops
            predictions = self.cnn_model.predict(crops_batch, verbose=False)
            
            for i, cnn_pred in enumerate(predictions):
                pred_keras_idx = np.argmax(cnn_pred)
                cnn_conf = float(cnn_pred[pred_keras_idx])
                
                # Double-threshold verification
                if cnn_conf < self.cnn_conf_threshold:
                    continue
                    
                x1, y1, x2, y2 = bboxes_list[i]
                true_class_id = int(pred_keras_idx)
                class_name = self.class_names.get(true_class_id, "Unknown Sign")
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": true_class_id,
                    "class_name": class_name,
                    "confidence": cnn_conf
                })
                
                # Annotate image
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ({cnn_conf*100:.1f}%)"
                text_y = y1 - 10 if y1 - 10 > 20 else y1 + 20
                cv2.putText(
                    annotated_img,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                
        return annotated_img, detections

if __name__ == "__main__":
    print("Inference pipeline script loaded. Run via app.py or a test script.")
    pipeline = TrafficSignPipeline()
    print("Class mapping length:", len(pipeline.keras_class_map))
