# Disease Prediction from Medical Data

## CodeAlpha Machine Learning Internship — Task 4

This project uses machine learning classification algorithms to predict the presence of heart disease from structured medical data.

The project demonstrates data preprocessing, missing-value handling, model training, model comparison, evaluation, and visualization.

## Objective

The objective is to build a machine learning model that predicts whether a patient is likely to have heart disease based on medical attributes.

This project converts the original UCI Heart Disease target into a binary classification problem:

* `0` → No Disease
* `1` → Disease

## Dataset

The project uses the UCI Heart Disease dataset.

The dataset contains 303 patient records and 13 input features, including:

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

The dataset is downloaded using the `ucimlrepo` Python package and saved locally as:

```text
data/heart_disease.csv
```

## Machine Learning Models

Two classification algorithms were trained and compared:

1. Logistic Regression
2. Random Forest

The preprocessing pipeline includes:

* Missing-value imputation using the median
* Feature standardization for Logistic Regression
* Train-test splitting with stratification

## Model Performance

### Logistic Regression

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 86.89% |
| Precision | 81.25% |
| Recall    | 92.86% |
| F1-Score  | 86.67% |
| ROC-AUC   | 95.13% |

### Random Forest

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 90.16% |
| Precision | 84.38% |
| Recall    | 96.43% |
| F1-Score  | 90.00% |
| ROC-AUC   | 95.45% |

### Best Model

The **Random Forest classifier** achieved the best ROC-AUC score of **0.9545** and was selected as the final model.

The trained model is saved as:

```text
models/disease_prediction_model.pkl
```

## Results

The following evaluation files are generated in the `results/` directory:

* `confusion_matrix.png`
* `roc_curve.png`
* `model_comparison.csv`

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
├── notebooks/
│
├── results/
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   └── roc_curve.png
│
├── src/
│   ├── download_data.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── README.md
└── requirements.txt
```

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Download the dataset

```bash
python src/download_data.py
```

### Train the models

```bash
python src/train_model.py
```

### Evaluate the best model

```bash
python src/evaluate_model.py
```

The trained model and evaluation results will be stored in the `models/` and `results/` directories.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* UCI ML Repository

## Internship Task

**CodeAlpha Machine Learning Internship — Task 4: Disease Prediction from Medical Data**

## Disclaimer

This project is intended for educational and internship purposes only. It is a machine learning demonstration and should not be used as a medical diagnostic system.
