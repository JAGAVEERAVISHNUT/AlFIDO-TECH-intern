import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Build and return a CNN model for MNIST."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First Conv block
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second Conv block
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    
    return model


def train_model(
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    model_name="mnist_model",
    output_dir="models"
):
    """Train CNN on MNIST dataset."""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess MNIST
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Reshape to (28, 28, 1) for CNN
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"Training set shape: {x_train.shape}, Test set shape: {x_test.shape}")
    
    # Build model
    model = build_cnn_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Callbacks
    model_path = os.path.join(output_dir, f"{model_name}.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            verbose=1,
            restore_best_weights=True
        ),
    ]
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Save training history
    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    history_data = {
        "loss": [float(l) for l in history.history["loss"]],
        "accuracy": [float(a) for a in history.history["accuracy"]],
        "val_loss": [float(l) for l in history.history["val_loss"]],
        "val_accuracy": [float(a) for a in history.history["val_accuracy"]],
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    
    print(f"\nModel saved to: {model_path}")
    print(f"History saved to: {history_path}")
    
    return model, history_data, (x_test, y_test)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train CNN on MNIST dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--model-name", default="mnist_model", help="Model name for saving")
    parser.add_argument("--output-dir", default="models", help="Output directory for model and history")
    args = parser.parse_args(argv)
    
    model, history, (x_test, y_test) = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        model_name=args.model_name,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
