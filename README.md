# RiskRadar

**Audit risk prediction for internal departments.** RiskRadar trains a scikit-learn classifier on historical audit records and serves it through a Flask web interface and a JSON API.

> Originally developed during a summer internship at PwC Tunisia.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dataset](#dataset)
- [Modeling Approach](#modeling-approach)
- [Results and Limitations](#results-and-limitations)
- [Roadmap](#roadmap)

## Overview
Given a **department**, an **audit finding type** and a **month**, RiskRadar estimates the probability of each risk level (`Low`, `Medium`, `High`) and reports the most likely one.

**Tech stack:** Python 3.10+, Flask, scikit-learn, pandas, joblib.

## Project Structure
```
RISKRADAR/
├── app/
│   ├── app.py               # Flask application (UI + JSON API)
│   ├── templates/index.html # Web interface
│   ├── model.joblib         # Trained pipeline (generated)
│   └── model_meta.json      # Model name, classes, metrics (generated)
├── data/
│   └── audit_dataset_updated.csv
├── scripts/
│   └── train_model.py       # Training and evaluation
└── requirements.txt
```

## Getting Started
```bash
git clone https://github.com/hidayahanafi/riskradar.git
cd riskradar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/train_model.py   # train and write app/model.joblib
python app/app.py               # serve at http://127.0.0.1:5000
```

## Usage
**Web interface:** open `http://127.0.0.1:5000`, choose the inputs and submit.

**REST API**
| Method | Endpoint       | Description                              |
|--------|----------------|------------------------------------------|
| GET    | `/`            | Web form                                 |
| POST   | `/predict`     | Form submission, renders the result      |
| POST   | `/api/predict` | JSON prediction                          |
| GET    | `/healthz`     | Liveness check                           |

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"department": "IT", "finding": "Control Weakness", "month": 3}'
```
```json
{"risk_level": "Low", "probabilities": {"High": 0.26, "Low": 0.40, "Medium": 0.34}}
```
Invalid input returns HTTP `400` with an `error` message.

## Dataset
`data/audit_dataset_updated.csv` — 10,003 anonymized audit records.

| Column          | Description                                  |
|-----------------|----------------------------------------------|
| `Audit_ID`      | Unique identifier                            |
| `Department`    | Operations, HR, Finance, IT, Marketing       |
| `Audit_Finding` | Fraudulent Activity, Control Weakness, Non-compliance |
| `Risk_Level`    | Target: Low, Medium, High                    |
| `Audit_Date`    | Date of the audit (almost entirely 2022)     |
| `Auditor_Name`  | Auditor (not used as a feature)              |

Preprocessing removes departments with fewer than 20 records (two single-row entries, `Legal` and `Compliance`) and derives `Month` from `Audit_Date`.

## Modeling Approach
- A single scikit-learn `Pipeline` (one-hot encoding + classifier) is used for both training and serving, so the two cannot diverge.
- Candidates: Logistic Regression, Decision Tree, Random Forest, plus a most-frequent **baseline**.
- Selection by 5-fold stratified cross-validation on F1-macro; final metrics are reported on a 20% held-out split. The deployed model is then refit on all data.

## Results and Limitations
| Model                    | CV F1-macro |
|--------------------------|-------------|
| Baseline (most frequent) | 0.168       |
| Logistic Regression      | 0.325       |
| Decision Tree            | 0.326       |
| **Random Forest**        | **0.339**   |

Held-out test F1-macro: **0.336**.

The best model is only marginally above chance (0.333 for three balanced classes). In this dataset the risk levels are spread almost evenly (~33% each) across every department, finding type and month, so the available features carry little predictive signal. **Predictions should be treated as illustrative and not used for audit decisions.**

## Related: Solvency 2 SCR tool
The [`solvency2/`](solvency2/) folder contains a small actuarial tool that aggregates risk module charges into the SCR with the standard-formula correlation matrix, with tests and an Excel report.

## Roadmap
- Add informative features (prior findings, control test results, financial exposure, audit scope).
- Extend data beyond 2022 to support temporal validation.
- Add unit tests and CI.
- Containerize for deployment (Docker, gunicorn).
