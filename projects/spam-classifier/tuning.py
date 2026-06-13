# Week 2 Day 2 - Hyperparameter Tuning with GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
df['message'] = df['message'].str.lower()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define what to try
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
}

# Search for the best combination
print("Searching for best parameters... (this takes a minute)")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train_vec, y_train)

print("\nBest parameters found:")
print(grid_search.best_params_)
print(f"Best F1 score during search: {grid_search.best_score_:.3f}")

# Evaluate the best model on test set
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_vec)

print("\n=== Final Results with Best Parameters ===")
print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))