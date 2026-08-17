import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
import joblib


# Paths
DATA_PATH = "data/heart_disease.csv"
MODEL_PATH = "models/disease_prediction_model.pkl"
RESULTS_DIR = "results"

os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Load dataset
data = pd.read_csv(DATA_PATH)

print("Dataset shape:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())


# Features and target
X = data.drop("target", axis=1)

# Convert the UCI target into binary classification
# 0 = No disease
# 1, 2, 3, 4 = Disease
y = (data["target"] > 0).astype(int)

print("\nTarget distribution:")
print(y.value_counts())


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Logistic Regression pipeline
logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Random Forest pipeline
random_forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])


models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}


results = {}
best_model = None
best_auc = 0
best_name = ""


# Train and evaluate models
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": auc
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_name = name


# Save best model
joblib.dump(best_model, MODEL_PATH)

print(f"\nBest model: {best_name}")
print(f"Model saved to: {MODEL_PATH}")


# Confusion matrix for best model
best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Disease"]
)

disp.plot()
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "confusion_matrix.png")
)
plt.close()


# ROC curves
plt.figure()

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "roc_curve.png")
)

plt.close()


# Save model comparison
comparison = pd.DataFrame(results).T
comparison.to_csv(
    os.path.join(RESULTS_DIR, "model_comparison.csv")
)

print("\nModel comparison:")
print(comparison)

print("\nTraining completed successfully!")