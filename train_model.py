# Train and Save Diabetes Prediction Model

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
# PIMA Diabetes dataset (from UCI repo, 768 rows)
# If you don't have it, you can download diabetes.csv from Kaggle
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")

# The dataset has no headers, so add them:
df.columns = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "Outcome"
]

# -------------------------------
# 2. Split Features and Labels
# -------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------
# 4. Scale Data
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 5. Train Model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -------------------------------
# 6. Evaluate
# -------------------------------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Model trained successfully ✅ | Accuracy: {acc*100:.2f}%")

# -------------------------------
# 7. Save Model and Scaler
# -------------------------------
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("trained_model.pkl and scaler.pkl saved 🎉")