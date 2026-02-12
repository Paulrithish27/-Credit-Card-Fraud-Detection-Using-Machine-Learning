import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score
)

from imblearn.over_sampling import SMOTE

print("Loading Dataset...")

df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nChecking Missing Values:")
print(df.isnull().sum().sum())

fraud_percentage = (df["Class"].sum() / len(df)) * 100
print(f"\nFraud Percentage: {fraud_percentage:.4f}%")


print("\nScaling 'Amount' and 'Time'...")

scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df["Time"] = scaler.fit_transform(df[["Time"]])

#
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nApplying SMOTE to balance dataset...")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE class distribution:")
print(np.bincount(y_train_res))

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)

print("\nEvaluating Model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

if not os.path.exists("model"):
    os.makedirs("model")

joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel and Scaler saved successfully!")

print("\nTesting with sample transaction...")

sample = X_test.iloc[0:1]
prediction = model.predict(sample)
probability = model.predict_proba(sample)[0][1]

print("Prediction:", "FRAUD" if prediction[0] == 1 else "NORMAL")
print("Fraud Probability:", probability)

print("\nProject Completed Successfully ✅")
