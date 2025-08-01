# Audit Risk Prediction App 🎯

This project predicts audit risk levels for internal departments based on past data using a machine learning model. Built during a summer internship at PwC Tunisia, it combines a trained classifier with a Flask web interface.

## 🧠 Model & Data
- Models used: Random Forest, Logistic Regression, Decision Tree
- Best model selected using F1-macro score
- Encoded categorical features using `LabelEncoder`
- Dataset: Anonymized audit reports from 2022–2023

## 🖥️ Tech Stack
- Python, Flask
- Scikit-learn, Pandas, NumPy, Joblib
- HTML (Jinja2 templates)

## 📦 How to Run

```bash
git clone https://github.com/yourusername/RISKRADAR.git
cd audit-risk-prediction-app
pip install -r requirements.txt
cd app
python app.py
