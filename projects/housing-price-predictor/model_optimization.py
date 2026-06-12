# Day 3 - Cross Validation & Model Tuning
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np

# Load data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Cross validation - tests model on 5 different splits
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')

print("Cross Validation R² scores:", scores.round(2))
print(f"Average R²: {scores.mean():.2f}")
print(f"Standard deviation: {scores.std():.2f}")

# Try different settings and compare
for n_trees in [50, 100, 200]:
    rf = RandomForestRegressor(n_estimators=n_trees, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"Trees: {n_trees} → R²: {r2:.2f}")