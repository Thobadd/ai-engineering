# Week 2 Day 4 - Spam Classifier API
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Create the API
app = FastAPI(title="Spam Classifier API")

# Define what the request looks like
class Message(BaseModel):
    text: str

# Define the endpoint
@app.post("/predict")
def predict(message: Message):
    vec = vectorizer.transform([message.text.lower()])
    prediction = model.predict(vec)[0]
    label = "spam" if prediction == 1 else "ham"
    return {
        "message": message.text,
        "prediction": label,
        "is_spam": bool(prediction)
    }

@app.get("/")
def home():
    return {"status": "Spam Classifier API is running"}