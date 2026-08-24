import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(32, 32, 3), num_classes=43):
    """
    Creates the custom VGG-style CNN architecture for classifying 
    cropped traffic sign images into one of 43 classes.
    Batch Normalization is excluded to prevent numerical variance underflow
    and stabilize training on small dataset batches.
    """
    model = models.Sequential()
    
    # Block 1
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Fully Connected Block
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

if __name__ == "__main__":
    # Test compilation
    model = create_cnn_model()
    model.summary()
