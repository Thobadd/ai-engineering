# Week 2 Day 3 - Save model and predict on new messages
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
df['message'] = df['message'].str.lower()
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully")

# Load and test on new messages
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

# Test with real messages
test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "WINNER! You have been selected for a £1000 prize. Call 09061234567 now!",
    "Can you pick up some milk on your way home?",
    "FREE entry to win FA Cup final! Text FA to 87121",
    "I'll be late to the meeting, stuck in traffic",
    "Urgent! Your account has been compromised. Click here to verify now!"
]

print("\n=== Predictions on New Messages ===")
for msg in test_messages:
    vec = loaded_vectorizer.transform([msg.lower()])
    prediction = loaded_model.predict(vec)[0]
    label = "SPAM" if prediction == 1 else "HAM"
    print(f"[{label}] {msg}")