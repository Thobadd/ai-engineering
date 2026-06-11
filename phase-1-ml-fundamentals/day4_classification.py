# Day 4 - Classification with Scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# What are we predicting?
print("Classes:", data.target_names)
print("Dataset shape:", X.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Evaluate
print("\nLogistic Regression:")
print(classification_report(y_test, lr_preds, target_names=data.target_names))

print("Random Forest:")
print(classification_report(y_test, rf_preds, target_names=data.target_names))

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, rf_preds))
print("\nRows = Actual, Columns = Predicted")
print("Labels:", list(data.target_names))