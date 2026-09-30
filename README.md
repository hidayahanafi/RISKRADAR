# RiskRadar – Audit Risk Prediction App 🎯

Predicts the risk level (Low / Medium / High) of an internal audit from its department, finding type and month, using a scikit-learn model served by a Flask web app. Built during a summer internship at PwC Tunisia.

## Quick start
```bash
pip install -r requirements.txt
python scripts/train_model.py   # trains, writes app/model.joblib + app/model_meta.json
python app/app.py               # http://127.0.0.1:5000
```

## Endpoints
- `GET /` – web form; `POST /predict` – form submit, shows class probabilities
- `POST /api/predict` – JSON `{"department": "IT", "finding": "Control Weakness", "month": 3}`
- `GET /healthz` – health check

## Model
- Candidates: Logistic Regression, Decision Tree, Random Forest, compared against a most-frequent baseline
- One sklearn `Pipeline` (one-hot encoding + classifier), so training and serving can't drift apart
- Selection by 5-fold stratified CV F1-macro; final score on a held-out 20% split
- Rare departments (<20 rows, e.g. single-row typos) are dropped

## ⚠️ Known limitation
In `data/audit_dataset_updated.csv` the risk levels are ~33/33/33% for every department, finding and month, and ~99.8% of rows are from 2022. The best model scores F1-macro ≈ 0.34 (chance for 3 classes), so predictions carry essentially no signal. The app displays probabilities and this caveat. Real value requires richer features (e.g. prior findings, control scores, amounts) or real data.

## Changes from the original
- Training script and app were out of sync (different feature sets and encoder files, wrong data path) – now unified
- Replaced the hardcoded "most common finding" input with a user-selected finding; dropped the near-constant `Year` feature
- Added input validation, probabilities, JSON API, responsive UI, `requirements.txt`; removed unused template and binary pickles from git
