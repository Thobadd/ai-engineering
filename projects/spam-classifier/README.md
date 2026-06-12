# SMS Spam Classifier

A machine learning classifier that identifies spam SMS messages with 98% accuracy.

## Overview

This project uses a Random Forest classifier with TF-IDF vectorization to classify SMS messages as spam or legitimate. Trained on the UCI SMS Spam Collection dataset (5,572 messages).

## Results

- **Accuracy**: 98%
- **Spam Detection Rate**: 89%
- **False Positive Rate**: 0.1%

## Model Performance
          precision    recall  f1-score
    ham       0.98      0.98      0.98
   spam       0.98      0.89      0.93
accuracy                           0.98

## Files

- `main.py` - Complete classifier pipeline
- `data/SMSSpamCollection` - Training dataset

## How to Use

```bash
python3 main.py
```

## Technologies

- Python 3
- scikit-learn (RandomForest, TfidfVectorizer)
- pandas