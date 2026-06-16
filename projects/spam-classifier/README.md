# SMS Spam Classifier

A machine learning API that classifies SMS messages as spam or legitimate with 98% accuracy.

## Overview

Built with Random Forest and TF-IDF vectorization, trained on the UCI SMS Spam Collection dataset (5,572 messages). Served as a REST API using FastAPI.

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 98% |
| Spam Precision | 100% |
| Ham Recall | 100% |

## API Usage

Start the server:
```bash
python3 -m uvicorn api:app --reload
```

Make a prediction:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"text": "Your message here"}'
```

Response:
```json
{
  "message": "Your message here",
  "prediction": "spam",
  "is_spam": true
}
```

## Project Structure

spam-classifier/

├── api.py          # FastAPI server

├── main.py         # Training pipeline

├── tuning.py       # Hyperparameter tuning

├── predict.py      # Prediction script

├── model.pkl       # Saved model

├── vectorizer.pkl  # Saved vectorizer

└── data/

└── SMSSpamCollection

## Technologies

- Python 3
- scikit-learn (RandomForest, TfidfVectorizer, GridSearchCV)
- FastAPI + Uvicorn
- pandas, pickle