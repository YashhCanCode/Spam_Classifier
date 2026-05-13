# Spam Classifier — Naive Bayes vs Logistic Regression

## Overview
A machine learning project that classifies SMS messages as spam or ham (not spam),
comparing two classification algorithms on a real-world dataset.

## Dataset
- SMS Spam Collection Dataset (UCI / Kaggle)
- 5,572 messages — 4,825 ham (87%) and 747 spam (13%)
- Imbalanced dataset — evaluated using AUC, not just accuracy

## Key Insight from EDA
Spam messages average 139 characters vs 71 for ham — nearly 2x longer.

## Approach
1. Loaded and explored the dataset
2. Encoded labels and analysed class distribution
3. Converted text to numerical features using TF-IDF (3,000 features)
4. Trained Multinomial Naive Bayes and Logistic Regression
5. Evaluated using confusion matrix, classification report, and ROC-AUC

## Results

| Model | AUC | Spam Recall | Ham Precision |
|-------|-----|-------------|---------------|
| Naive Bayes | 0.9820 | 81% | 97% |
| Logistic Regression | 0.9863 | 79% | 97% |

## Key Finding
- Logistic Regression wins on AUC and never false-alarms on ham (0 false positives)
- Naive Bayes catches more spam (higher recall) and handles tricky urgent-language spam better
- Choice depends on use case — missing spam vs annoying false alarms

## Tech Stack
Python, pandas, scikit-learn, matplotlib, Google Colab

## Files
- `spam_classifier.ipynb` — full notebook with code, outputs, and charts
- `spam.csv` — dataset
