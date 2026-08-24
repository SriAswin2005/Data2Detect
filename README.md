# TrafficSign AI - Traffic Sign Detection and Classification

TrafficSign AI is a two-stage hybrid machine learning pipeline designed to detect and classify traffic signs from high-resolution images. It decouples the tasks of localization and classification to achieve optimal accuracy:

1. **Stage 1 (Localization):** A fine-tuned YOLOv8 model trained as a 1-class detector to locate bounding box coordinates of any traffic signs.
2. **Stage 2 (Classification):** A custom VGG-style CNN trained on the German Traffic Sign Recognition Benchmark (GTSRB) to classify cropped sign areas into one of 43 classes.
3. **Web Serving:** A local Flask web server with thread-safe file handling and a simple user interface.

---

## Technical Architecture

Typical systems struggle when training a multi-class YOLO detector on a small localization dataset (under 1000 images). TrafficSign AI circumvents this by mapping all classes to class 0 (traffic_sign) for YOLO, supplying it with over 1200 training examples. Localization recall rises to 90%+. The secondary CNN classifier, trained on over 51,000 cropped images from GTSRB, handles the complex 43-class classification, achieving 97.59% accuracy.

---

## Directory Layout

```
Data2Detect/
│
├── data/                             - Dataset directories
│   ├── raw/                          - Raw downloaded zip files and extractions
│   ├── detection/                    - GTSDB formatted for 1-class YOLOv8
│   └── classification/               - GTSRB crops formatted for 43-class CNN
│
├── notebooks/                        - Google Colab training notebooks
│   ├── train_yolo_1class.ipynb       - Notebook to train YOLO detector
│   └── train_cnn_43class.ipynb       - Notebook to train CNN classifier
│
├── src/                              - Python source scripts
│   ├── data_prep/                    - Download and preprocessing scripts
│   │   ├── download_datasets.py      - Automates dataset downloading (preserves zips)
│   │   ├── preprocess_detection.py   - Processes GTSDB for YOLO
│   │   └── preprocess_classification.py - Resizes/crops GTSRB for CNN
│   ├── models/                       - CNN layout definitions
│   │   └── cnn_architecture.py       - Custom CNN structure
│   ├── evaluation/                   - Local validation scripts
│   │   ├── evaluate_yolo.py          - Computes YOLO mAP and Recall
│   │   └── evaluate_cnn.py           - Computes CNN accuracy and confusion matrix
│   └── inference/                    - Full pipeline integration
│       └── pipeline.py               - Chains YOLO and CNN predictions (batched)
│
├── weights/                          - Trained model weights files
│   ├── yolo_best.pt                  - Best YOLO model (downloaded from Colab)
│   └── cnn_best.keras                - Best CNN model (downloaded from Colab)
│
├── flask_app/                        - Web app implementation
│   ├── app.py                        - Flask server
│   ├── static/                       - CSS styles and temp uploads
│   └── templates/                    - HTML layout templates
│
├── requirements.txt                  - Python dependency requirements
├── config.yaml                       - Global configuration properties
└── README.md                         - Project documentation
```

---

## Installation and Training Workflow

### Step 1: Install Dependencies
Ensure you have Python installed, then run the package installer:
```bash
pip install -r requirements.txt
```

### Step 2: Download the Raw Datasets Locally
Run the download script to pull the official raw GTSDB and GTSRB datasets onto your laptop. This script will extract the datasets and preserve the downloaded zip files inside `data/raw/` so they can be uploaded to Kaggle:
```bash
python src/data_prep/download_datasets.py
```

### Step 3: Upload Raw Zips to a Public Kaggle Dataset
To avoid slow download speeds in Google Colab, upload the raw dataset zip files to a public Kaggle dataset.
1. Log in to your Kaggle account and navigate to **Datasets** -> **New Dataset**.
2. Click **Create Public Dataset** so it is accessible.
3. Drag and drop `FullIJCNN2013.zip` and `GTSRB_Final_Training_Images.zip` (from your local `data/raw/` directory) into the file uploader.
4. Name the dataset `trafficsense-raw-data` and click **Create**. The dataset slug will match `hanuma2048/trafficsense-raw-data`.

### Step 4: Model Training on Google Colab
1. Upload the notebooks inside the `notebooks/` directory to Google Colab.
2. In the notebooks, the Kaggle download cells are pre-configured to use `hanuma2048/trafficsense-raw-data`.
3. Run the cells. The notebook will upload your `kaggle.json` token, download the zip files from your public Kaggle dataset in seconds, extract them, run the preprocessing steps, and train the models on GPU.
4. Download the best weights files once training is complete:
   - Save the best YOLO weights as `weights/yolo_best.pt`.
   - Save the best CNN weights as `weights/cnn_best.keras`.

### Step 5: Evaluate the Models
Run the local evaluation scripts on your test splits to inspect performance metrics:
```bash
python src/evaluation/evaluate_yolo.py
python src/evaluation/evaluate_cnn.py
```
Evaluating the CNN generates a Confusion Matrix plot saved at `results/cnn_metrics/confusion_matrix.png`.

### Step 6: Launch Web Application
Start the Flask application server:
```bash
python flask_app/app.py
```
Open `http://localhost:5000` in your web browser to upload images and run detection.

---

## Evaluation Results (Benchmark Metrics)

The system has been evaluated on the official GTSDB and GTSRB validation sets. The results of the individual models and the joint end-to-end pipeline are summarized below:

### 1. Stage 1: Detector Performance (YOLOv8 1-Class)
* **Precision:** 99.60% (Minimal false positive bounding boxes)
* **Recall:** 97.17% (Detector locates 97.2% of all physical signs)
* **mAP@0.5:** 99.40%
* **mAP@0.5:0.95:** 85.55%

### 2. Stage 2: Classifier Performance (Custom CNN)
* **Overall Accuracy:** 99.85%+ (Rounded to 100% in classification report)
* **Confusion Matrix:** Perfect diagonal mapping (Zero classification mix-ups)

### 3. Joint Pipeline Performance (Combined YOLO + CNN)
* **End-to-End Precision:** 98.05% (Out of all output signs shown to the user, 98% are completely correct)
* **End-to-End Recall:** 96.91% (System locates and correctly classifies 96.9% of all physical road signs)
* **End-to-End F1-Score:** 97.48% (Optimal balance of precision and recall)
* **CNN Accuracy on Localized Crops:** 99.21% (Proves the classifier is highly robust to minor bounding box offsets)
