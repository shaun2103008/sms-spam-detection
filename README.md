# 🚀 SMS Spam Detection System (Production-Ready)

A deployed machine learning-powered SMS Spam Detection system built using TF-IDF and Logistic Regression, served via a Flask backend with a real-time web interface.

---

## 📌 Project Overview

This project demonstrates how a traditional ML model can be converted from a Jupyter notebook into a production-ready inference system.

It includes:
- Model training pipeline
- Model serialization
- REST-style backend
- Web interface for real-time predictions
- Threshold tuning for classification sensitivity

---

## 🧠 Model Details

- Feature Extraction: TF-IDF Vectorization
- Algorithm: Logistic Regression
- Threshold-based classification (customizable)
- Dataset: SMS Spam Collection Dataset

---

## 🏗️ System Architecture

User Input → Flask Backend → TF-IDF Transform → Logistic Regression → Prediction Output

---

## ⚙️ Tech Stack

- Python
- scikit-learn
- Flask
- HTML
- Gunicorn (for deployment)

---

## 🚀 How To Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open:

http://127.0.0.1:5000/