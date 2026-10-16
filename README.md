# Spam Message Detector

A minimal Streamlit app that classifies a text message as **SPAM** or **NOT SPAM**
using a scikit-learn pipeline (`TfidfVectorizer` + `MultinomialNB`) trained on the
public SMS Spam Collection dataset.

> **Note:** Predictions can be incorrect. This is a simple demo, not a production filter.

## Project structure

```
spam_detector/
├── app.py            # Streamlit interface + prediction
├── train_model.py    # Training, evaluation, saving the model
├── requirements.txt
├── README.md
├── data/
│   └── SMSSpamCollection     # you download this (see below)
└── model/
    └── spam_model.joblib     # created by train_model.py
```

## 1. Set up the environment

Python 3.9+ is recommended.

```bash
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Download the dataset

Source: UCI Machine Learning Repository, "SMS Spam Collection"
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

1. Click **Download** on that page (or use
   https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip).
2. Unzip it. Inside you'll find a file named `SMSSpamCollection` (no extension).
3. Put that file at:

```
data/SMSSpamCollection
```

**Expected format:** plain text, one message per line, tab-separated, no header:

```
ham<TAB>Go until jurong point, crazy.. Available only in bugis n great world
spam<TAB>Free entry in 2 a wkly comp to win FA Cup final tkts ...
```

Labels are `ham` (not spam) or `spam`.

## 3. Train the model

```bash
python train_model.py
```

This does a stratified 80/20 train/test split (random seed 42), fits the pipeline on
the training data only, prints accuracy and spam precision/recall on the test data,
and saves the model to `model/spam_model.joblib`.

## 4. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501). The app loads the
saved model once; it never retrains. To retrain, re-run `python train_model.py` and
restart the app.

## Troubleshooting

- **"Dataset not found"** – check the file is exactly `data/SMSSpamCollection`.
- **"Trained model not found"** – run `python train_model.py` first.
