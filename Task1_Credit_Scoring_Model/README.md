# CodeAlpha Credit Scoring Model

## 📌 Project Overview

This project was developed as part of the CodeAlpha Machine Learning Internship.

The objective is to build a machine learning model that predicts the creditworthiness of an applicant based on historical financial and personal information.

## 🎯 Objective

The system classifies applicants into:

- Good Credit Risk
- Poor Credit Risk

Two classification algorithms were implemented and compared:

1. Logistic Regression
2. Random Forest Classifier

## 📊 Dataset

The project uses the German Credit dataset containing 1,000 credit applicants.

Features include:

- Age
- Sex
- Job
- Housing
- Saving accounts
- Checking account
- Credit amount
- Duration
- Purpose

Target:

- `0` → Good Credit Risk
- `1` → Poor Credit Risk

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Google Colab

## 🔄 Data Preprocessing

The following preprocessing techniques were applied:

- Missing-value imputation
- Categorical feature encoding using One-Hot Encoding
- Numerical feature standardization
- Train-test split
- Stratified sampling

## 🤖 Machine Learning Models

### Logistic Regression

Used as the baseline classification model.

### Random Forest

Used as the second classification model and compared with Logistic Regression.

## 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

## 🔍 Prediction Example

The trained model was tested with a sample applicant.

Example prediction:

**Good Credit Risk**

Poor-risk probability:

**14.50%**

## 📁 Project Structure

```text
CodeAlpha_CreditScoringModel/
│
├── notebooks/
│   └── Credit_Scoring_Model.ipynb
│
├── models/
│   └── credit_scoring_model.pkl
│
├── results/
│   ├── model_comparison.png
│   ├── logistic_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   └── roc_curve.png
│
├── src/
│   └── predict.py
│
├── README.md
└── requirements.txt