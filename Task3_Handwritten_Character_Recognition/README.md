# Handwritten Character Recognition using CNN

## CodeAlpha Machine Learning Internship — Task 3

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset.

## Objective

The objective of this project is to build a deep learning model capable of identifying handwritten digits from 0 to 9.

## Dataset

The project uses the MNIST handwritten digit dataset.

- Training images: 60,000
- Testing images: 10,000
- Image size: 28 × 28 pixels
- Number of classes: 10
- Classes: digits 0–9

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Model Architecture

The CNN consists of:

1. Convolutional Layer — 32 filters
2. Max Pooling Layer
3. Convolutional Layer — 64 filters
4. Max Pooling Layer
5. Flatten Layer
6. Dense Layer — 128 neurons
7. Output Layer — 10 neurons with Softmax activation

## Model Performance

The trained CNN achieved:

**Test Accuracy: 98.89%**

**Test Loss: 0.0356**

The project also includes:

- Training and validation accuracy graph
- Training and validation loss graph
- Confusion matrix
- Classification report
- Trained CNN model

## Project Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
│
├── models/
│   └── handwritten_digit_cnn.keras
│
├── notebooks/
│   └── Handwritten_Character_Recognition.ipynb
│
├── results/
│   ├── accuracy_loss_accuracy.png
│   ├── accuracy_loss_loss.png
│   └── confusion_matrix.png
│
├── src/
│   └── predict.py
│
├── README.md
└── requirements.txt