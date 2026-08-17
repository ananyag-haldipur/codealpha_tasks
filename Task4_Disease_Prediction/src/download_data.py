from ucimlrepo import fetch_ucirepo

# Fetch UCI Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Get features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine into one dataframe
data = X.copy()
data["target"] = y.iloc[:, 0]

# Save locally
data.to_csv("data/heart_disease.csv", index=False)

print("Dataset downloaded successfully!")
print("Shape:", data.shape)
print(data.head())