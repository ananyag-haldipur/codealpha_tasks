# Disease Prediction from Medical Data

## CodeAlpha Machine Learning Internship — Task 4

### Overview

This project uses machine learning to predict the presence of heart disease from structured medical data.

The project uses the UCI Heart Disease dataset and compares two classification algorithms:

* Logistic Regression
* Random Forest Classifier

The UCI target variable is converted into a binary classification problem:

* `0` → No Disease
* `1` → Disease

## Project Structure

```text
Task4_Disease_Prediction/
│
├── data/
│   └── heart_disease.csv
│
├── models/
│   └── disease_prediction_model.pkl
│
├── results/
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   └── roc_curve.png
│
└── src/
    ├── download_data.py
    ├── train_model.py
    └── evaluate_model.py
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* UCI ML Repository

## Dataset

The project uses the UCI Heart Disease dataset containing 303 records and 13 input features.

The features include medical attributes such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG
* Maximum heart rate
* Exercise-induced angina
* ST depression
* Slope
* Number of major vessels
* Thalassemia

## Machine Learning Approach

### 1. Data Preparation

The dataset is loaded from the CSV file.

Missing values are handled using median imputation.

The original target is converted into a binary target:

```text
0 = No Disease
1 = Disease
```

### 2. Models

Two models were trained and compared:

* Logistic Regression
* Random Forest Classifier

### 3. Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

A confusion matrix and ROC curve were also generated.

## Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   86.89% |    81.25% | 92.86% |   86.67% |  0.9513 |
| Random Forest       |   90.16% |    84.38% | 96.43% |   90.00% |  0.9545 |

### Best Model

The Random Forest classifier achieved the best ROC-AUC score of **0.9545** and an accuracy of approximately **90.16%** on the test set.

Final evaluation results:

* No Disease precision: 0.97
* No Disease recall: 0.85
* Disease precision: 0.84
* Disease recall: 0.96
* Overall accuracy: 0.90
* ROC-AUC: 0.9545

## How to Run

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib ucimlrepo
```

Download the dataset:

```bash
python src/download_data.py
```

Train the models:

```bash
python src/train_model.py
```

Evaluate the best model:

```bash
python src/evaluate_model.py
```

The generated evaluation results will be stored in the `results/` directory.

## Disclaimer

This project is intended for educational and internship purposes only. It is a machine learning demonstration and should not be used as a medical diagnostic system.
