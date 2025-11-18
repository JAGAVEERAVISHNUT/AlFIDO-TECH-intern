
# Image Classification with CNN on MNIST

A complete project to train and evaluate a Convolutional Neural Network (CNN) on the MNIST handwritten digit dataset using TensorFlow/Keras.

## Overview

- **Dataset**: MNIST (60,000 training images, 10,000 test images of 28×28 pixel grayscale digits)
- **Model**: CNN with 2 convolutional blocks, max pooling, and dense layers
- **Framework**: TensorFlow 2.12+
- **Goal**: Classify handwritten digits (0-9) with high accuracy

## Project Structure

```
Image-Classification-CNN-MNIST/
├── requirements.txt           # Python dependencies
├── mnist_cnn.py              # Main training script
├── evaluate.py               # Evaluation and visualization script
├── README.md                 # This file
├── models/                   # (Created after training)
│   ├── mnist_model.keras     # Trained model
│   └── mnist_model_history.json
└── results/                  # (Created after evaluation)
    ├── confusion_matrix.png
    ├── sample_predictions.png
    └── training_history.png
```

## Setup

Install dependencies (PowerShell):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Training

Train the CNN model on MNIST (takes ~2–5 minutes depending on GPU):

```powershell
python mnist_cnn.py --epochs 10 --batch-size 128 --model-name mnist_model
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 128)
- `--validation-split`: Validation split ratio (default: 0.1)
- `--model-name`: Name for saving model (default: mnist_model)
- `--output-dir`: Output directory (default: models)

**Output:**
- `models/mnist_model.keras` — trained model weights
- `models/mnist_model_history.json` — training metrics (loss, accuracy per epoch)

## Evaluation

Evaluate the trained model and generate visualizations:

```powershell
python evaluate.py --model-path models/mnist_model.keras --history-path models/mnist_model_history.json
```

**Outputs:**
- `results/confusion_matrix.png` — per-digit classification confusion matrix
- `results/sample_predictions.png` — correct and misclassified examples
- `results/training_history.png` — loss and accuracy curves

## Model Architecture

```
Input (28, 28, 1)
├─ Conv2D(32, 3×3, relu)
├─ MaxPooling2D(2×2)
├─ Conv2D(64, 3×3, relu)
├─ MaxPooling2D(2×2)
├─ Flatten
├─ Dropout(0.5)
├─ Dense(128, relu)
├─ Dropout(0.5)
└─ Dense(10, softmax)
```

## Expected Results

With 10 epochs and the default configuration:
- **Test Accuracy**: ~98–99%
- **Training Time**: ~2–3 minutes (CPU) or ~30–60 seconds (GPU)

## Quick Start Commands

```powershell
# Install dependencies
pip install -r requirements.txt

# Train the model (10 epochs)
python mnist_cnn.py --epochs 10

# Evaluate and visualize results
python evaluate.py
```

## Notes

- First run will download the MNIST dataset (~11 MB).
- Model is saved automatically with best validation accuracy.
- Early stopping triggers after 3 epochs without improvement.
- GPU support is automatic if CUDA/cuDNN is available.

## Customization

Modify `mnist_cnn.py` to experiment with:
- Different CNN architectures (add/remove Conv2D layers)
- Learning rates (modify optimizer)
- Regularization (adjust dropout rates)
- Batch normalization and other techniques
