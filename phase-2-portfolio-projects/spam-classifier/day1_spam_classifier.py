# Week 2 Day 1 - Spam Classifier
# Real SMS spam dataset with data cleaning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Download and load the SMS Spam Collection dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

# Explore the data
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['label'].value_counts())
print("\nMessage lengths:")
print(df['message'].str.len().describe())

# Clean the data
df['message'] = df['message'].str.lower()  # Lowercase
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert to 0/1

# Split
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_vec, y_train)

# Evaluate
predictions = clf.predict(X_test_vec)
print("\n=== Results ===")
print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))