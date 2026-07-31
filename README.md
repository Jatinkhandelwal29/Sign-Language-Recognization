# Fake News Detection

An NLP-based classifier that flags news headlines/articles as
**REAL** or **FAKE**, using **TF-IDF** vectorization and comparing
**Logistic Regression** vs **Naive Bayes**, built with Python and
Scikit-learn.

## Overview

Article text is cleaned, converted into TF-IDF feature vectors
(unigrams + bigrams), and classified by two models trained side by
side. The better-performing model (by F1-score) is automatically
saved and used for predictions on new text.

## Features

- Regex-based text cleaning (no extra corpus downloads required)
- TF-IDF vectorization with unigrams + bigrams
- Trains and compares Logistic Regression and Naive Bayes
- Prints accuracy, F1-score, and full classification report
- Interactive CLI, single-string, or file-based prediction modes
- Includes a small built-in sample dataset for instant testing

## Tech Stack

Python · Scikit-learn · Pandas · NumPy

## Project Structure

```
fake_news_detection/
├── preprocess.py       # Text cleaning
├── train_model.py      # Step 1: train + compare models
├── predict.py           # Step 2: classify new text
├── requirements.txt
├── PROJECT_GUIDE.md    # Full step-by-step walkthrough
├── data/
│   └── sample_news.csv # Small built-in sample dataset
└── models/              # Trained model + vectorizer (generated)
```

## Installation

```bash
git clone <your-repo-url>
cd fake_news_detection
pip install -r requirements.txt
```

## Usage

```bash
# 1. Train on the built-in sample dataset
python train_model.py

# (Optional) Train on your own dataset
python train_model.py --data data/your_dataset.csv

# 2. Predict — interactive mode
python predict.py

# Or a single string
python predict.py --text "Breaking: local elections certified after standard audit"

# Or a text file
python predict.py --file article.txt
```

See `PROJECT_GUIDE.md` for the full step-by-step explanation,
including how to plug in the real Kaggle "Fake and Real News
Dataset" for stronger, resume-ready accuracy.

## Results

On the included 30-row sample dataset, Logistic Regression reaches
~87.5% accuracy — this dataset only proves the pipeline works.
With a full-scale dataset (~45k articles), accuracy typically
exceeds 90%.

## Author

**Jatin Khandelwal**
