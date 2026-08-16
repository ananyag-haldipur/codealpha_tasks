from pathlib import Path
import joblib
import pandas as pd


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model path
MODEL_PATH = PROJECT_ROOT / "models" / "credit_scoring_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)


def predict_credit_risk(
    age,
    sex,
    job,
    housing,
    saving_accounts,
    checking_account,
    credit_amount,
    duration,
    purpose
):
    applicant = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "Job": [job],
        "Housing": [housing],
        "Saving accounts": [saving_accounts],
        "Checking account": [checking_account],
        "Credit amount": [credit_amount],
        "Duration": [duration],
        "Purpose": [purpose]
    })

    prediction = model.predict(applicant)[0]
    probability = model.predict_proba(applicant)[0][1]

    if prediction == 1:
        risk = "Poor Credit Risk"
    else:
        risk = "Good Credit Risk"

    return risk, probability


if __name__ == "__main__":

    risk, probability = predict_credit_risk(
        age=30,
        sex="male",
        job=2,
        housing="own",
        saving_accounts="little",
        checking_account="moderate",
        credit_amount=2500,
        duration=12,
        purpose="car"
    )

    print("Predicted Risk:", risk)
    print(f"Poor-risk probability: {probability:.2%}")