import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras


def load_and_prepare_data():
    """Load and preprocess MNIST test set."""
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)
    return x_test, y_test


def evaluate_model(model_path, output_dir="results"):
    """Evaluate model on test set and generate metrics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    x_test, y_test = load_and_prepare_data()
    
    # Convert to one-hot for model evaluation
    y_test_onehot = keras.utils.to_categorical(y_test, 10)
    
    # Predictions
    print("Generating predictions...")
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    test_loss, test_accuracy = model.evaluate(x_test, y_test_onehot, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
    
    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - MNIST CNN")
    plt.colorbar(im, ax=ax)
    
    # Add text annotations
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.tight_layout()
    cm_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved to {cm_path}")
    plt.close()
    
    # Find misclassified samples
    misclassified_idx = np.where(y_pred != y_test)[0]
    print(f"\nMisclassified: {len(misclassified_idx)} out of {len(y_test)} ({100*len(misclassified_idx)/len(y_test):.2f}%)")
    
    # Plot sample predictions and misclassifications
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("Sample Predictions (Top: Correct, Bottom: Misclassified)")
    
    # Correct predictions
    correct_idx = np.where(y_pred == y_test)[0]
    for i in range(5):
        idx = correct_idx[i]
        axes[0, i].imshow(x_test[idx].squeeze(), cmap="gray")
        axes[0, i].set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}", color="green")
        axes[0, i].axis("off")
    
    # Misclassified predictions
    for i in range(5):
        if i < len(misclassified_idx):
            idx = misclassified_idx[i]
            axes[1, i].imshow(x_test[idx].squeeze(), cmap="gray")
            axes[1, i].set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}", color="red")
        axes[1, i].axis("off")
    
    plt.tight_layout()
    samples_path = Path(output_dir) / "sample_predictions.png"
    plt.savefig(samples_path, dpi=150)
    print(f"Sample predictions saved to {samples_path}")
    plt.close()


def plot_training_history(history_path, output_dir="results"):
    """Plot training history from saved JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading history from {history_path}...")
    with open(history_path, "r") as f:
        history = json.load(f)
    
    # Plot Loss and Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history["loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history["accuracy"], label="Train Accuracy", marker="o")
    axes[1].plot(history["val_accuracy"], label="Val Accuracy", marker="o")
    axes[1].axhline(y=history["test_accuracy"], color="r", linestyle="--", label=f"Test Accuracy: {history['test_accuracy']:.4f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_path_out = Path(output_dir) / "training_history.png"
    plt.savefig(history_path_out, dpi=150)
    print(f"Training history plot saved to {history_path_out}")
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate MNIST CNN model")
    parser.add_argument("--model-path", default="models/mnist_model.keras", help="Path to trained model")
    parser.add_argument("--history-path", default="models/mnist_model_history.json", help="Path to training history")
    parser.add_argument("--output-dir", default="results", help="Output directory for visualizations")
    args = parser.parse_args(argv)
    
    # Evaluate model
    if Path(args.model_path).exists():
        evaluate_model(args.model_path, args.output_dir)
    else:
        print(f"Model not found at {args.model_path}. Train the model first using mnist_cnn.py")
    
    # Plot history
    if Path(args.history_path).exists():
        plot_training_history(args.history_path, args.output_dir)
    else:
        print(f"History not found at {args.history_path}")


if __name__ == "__main__":
    main()
