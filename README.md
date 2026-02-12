# -Credit-Card-Fraud-Detection-Using-Machine-Learning
# 💳 Credit Card Fraud Detection Using Machine Learning

##📌 Project Overview

This project builds a Machine Learning model to detect fraudulent credit card transactions.  
The dataset is highly imbalanced, making fraud detection a challenging real-world problem.

The model uses SMOTE to handle imbalance and Random Forest for classification.


## 🚀 Features

- Data preprocessing
- Feature scaling (Amount & Time)
- Handling imbalanced dataset using SMOTE
- Random Forest Classifier
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Model saving using Joblib
- Sample transaction prediction


## 📊 Dataset

Dataset: Kaggle Credit Card Fraud Detection Dataset

Features:
- Time
- Amount
- V1 to V28 (PCA transformed features)
- Class (Target variable)
  - 0 → Normal Transaction
  - 1 → Fraudulent Transaction

Fraud cases represent less than 1% of total transactions.

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- PyCharm (IDE)
  
## Key Learning Outcomes

*Handling imbalanced datasets
*Applying SMOTE
*Evaluating classification models properly
*Building end-to-end ML pipeline
*Saving and reusing trained models
