# Day 2 - Random Forest vs Linear Regression
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

# Load and split data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# Compare
print("Linear Regression:")
print(f"  MAE: {mean_absolute_error(y_test, lr_preds):.2f}")
print(f"  R²:  {r2_score(y_test, lr_preds):.2f}")

print("\nRandom Forest:")
print(f"  MAE: {mean_absolute_error(y_test, rf_preds):.2f}")
print(f"  R²:  {r2_score(y_test, rf_preds):.2f}")