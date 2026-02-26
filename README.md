# 🚀 SMS Spam Detection System (Production-Ready)

A deployed machine learning-powered SMS Spam Detection system built using TF-IDF and Logistic Regression, served via a Flask backend with a real-time web interface.

---
## 🔗 Live Demo

https://sms-spam-detection-no1p.onrender.com

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/26e4d7ab-4945-4a29-a573-78622d6664cd" />

## 📌 Project Overview

This project demonstrates how a traditional Machine Learning model can be transformed from a notebook experiment into a production-ready deployed web application.

The system classifies SMS messages as **Spam** or **Not Spam** using a trained Logistic Regression model and serves predictions through a Flask-based web interface deployed on the cloud.

This project highlights the complete ML lifecycle:
- Data preprocessing
- Model training
- Model serialization
- Backend API integration
- Cloud deployment

---

## 🧠 Model Details

- **Feature Extraction:** TF-IDF Vectorization  
- **Algorithm:** Logistic Regression  
- **Threshold-based classification (customizable)**  
- **Dataset:** SMS Spam Collection Dataset  
- **Evaluation:** Accuracy-based validation with train/test split  

The system uses probability-based prediction with a configurable threshold to balance sensitivity and false positives.

---

## 🏗️ System Architecture

User Input  
→ Flask Backend  
→ TF-IDF Transformation  
→ Logistic Regression Model  
→ Probability Calculation  
→ Threshold Decision  
→ Result Displayed to User  

---

## ⚙️ Tech Stack

- Python  
- scikit-learn  
- Flask  
- HTML  
- Gunicorn  
- Render (Cloud Deployment)  
- Git & GitHub  

---

## 🚀 How To Run Locally

Clone the repository:

```bash
git clone https://github.com/shaun2103008/sms-spam-detection.git
cd sms-spam-detection

## 💡 Key Highlights

- End-to-end Machine Learning pipeline (training → inference → deployment)
- TF-IDF feature engineering with Logistic Regression classifier
- Model serialization using pickle for production inference
- Separation of training and serving logic
- Flask-based backend with real-time web interface
- Custom probability threshold tuning
- Cloud deployment using Render
- Version controlled using Git & GitHub

## 👨‍💻 Author

**Shaun Banis**  
Machine Learning & Backend Development Enthusiast  
📍 India  

Feel free to connect on LinkedIn for collaboration and discussions.
