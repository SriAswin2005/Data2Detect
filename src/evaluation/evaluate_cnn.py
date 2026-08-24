import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Import model architecture
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.cnn_architecture import create_cnn_model

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    weights_path = Path(config["paths"]["cnn_weights"])
    val_dir = Path(config["paths"]["cnn_data_dir"]) / "validation"
    
    if not weights_path.exists():
        print(f"Error: CNN weights file not found at {weights_path.absolute()}.")
        print("Please train the CNN model on Google Colab first, download the cnn_best.keras file,")
        print(f"and place it in the {weights_path.parent} directory.")
        return
        
    if not val_dir.exists():
        print(f"Error: Validation data directory not found at {val_dir.absolute()}.")
        print("Please run preprocess_classification.py first to extract and split the datasets.")
        return

    print(f"Loading CNN model from {weights_path}...")
    try:
        # Load whole model
        model = tf.keras.models.load_model(str(weights_path))
    except Exception:
        # Fallback to rebuilding and loading weights
        print("Could not load whole model directly. Rebuilding architecture and loading weights...")
        model = create_cnn_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(str(weights_path))

    # Setup generator for validation data
    img_size = config["inference"]["cnn_img_size"]
    
    # We do not shuffle so predicted outputs match generator ground truths in order
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)
    val_generator = val_datagen.flow_from_directory(
        directory=str(val_dir),
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='sparse',
        shuffle=False
    )
    
    print("Evaluating CNN model on validation dataset...")
    loss, accuracy = model.evaluate(val_generator)
    print(f"\n--- CNN Validation Set Performance ---")
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy*100:.2f}%")
    print("---------------------------------------")
    
    # Predict and generate confusion matrix
    print("Generating predictions for classification report and confusion matrix...")
    val_generator.reset()
    predictions = model.predict(val_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes
    
    # Get keras mapping to display correct name mappings
    keras_class_map = config["keras_class_map"]
    class_names = config["class_names"]
    
    # Align true classes and predicted classes to actual integer labels from directories
    # Note that Keras sorted directories alphabetically, so val_generator.class_indices 
    # maps directory names (strings) to Keras outputs (integers 0 to 42)
    keras_to_true_class = {}
    for dir_name, index in val_generator.class_indices.items():
        # dir_name is like '00000', '00010', convert to int
        class_int = int(dir_name)
        keras_to_true_class[index] = class_int
        
    # Map raw keras indices to the actual dataset class IDs
    true_mapped = np.array([keras_to_true_class[c] for c in true_classes])
    predicted_mapped = np.array([keras_to_true_class[c] for c in predicted_classes])
    
    # Generate labels lists
    unique_classes = sorted(list(keras_to_true_class.values()))
    target_names = [class_names[c] for c in unique_classes]
    
    print("\nClassification Report:")
    print(classification_report(true_mapped, predicted_mapped, labels=unique_classes, target_names=target_names))
    
    # Compute confusion matrix
    cm = confusion_matrix(true_mapped, predicted_mapped, labels=unique_classes)
    
    # Plot and save
    results_dir = Path("results/cnn_metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d", 
                xticklabels=target_names, yticklabels=target_names)
    plt.title("CNN Classifier Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Confusion matrix plot successfully saved to {cm_path.absolute()}")

if __name__ == "__main__":
    main()
