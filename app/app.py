import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent

app = Flask(__name__)

model = joblib.load(BASE / "model.joblib")
meta = json.loads((BASE / "model_meta.json").read_text())
DEPARTMENTS = meta["departments"]
FINDINGS = meta["findings"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def predict_risk(department: str, finding: str, month: int) -> dict:
    if department not in DEPARTMENTS:
        raise ValueError(f"Unknown department: {department!r}")
    if finding not in FINDINGS:
        raise ValueError(f"Unknown audit finding: {finding!r}")
    if not 1 <= month <= 12:
        raise ValueError("Month must be between 1 and 12")

    row = pd.DataFrame([{"Department": department, "Audit_Finding": finding, "Month": month}])
    proba = model.predict_proba(row)[0]
    probabilities = {c: round(float(p), 4) for c, p in zip(model.classes_, proba)}
    top = max(probabilities, key=probabilities.get)
    return {"risk_level": top, "probabilities": probabilities}


def render(result=None, error=None, form=None):
    return render_template(
        "index.html", departments=DEPARTMENTS, findings=FINDINGS, months=MONTHS,
        meta=meta, result=result, error=error, form=form or {},
    ), (400 if error else 200)


@app.get("/")
def home():
    return render()


@app.post("/predict")
def predict():
    form = request.form
    try:
        result = predict_risk(
            form.get("department", ""), form.get("finding", ""), int(form.get("month", "")),
        )
    except ValueError as e:
        return render(error=str(e), form=form)
    return render(result=result, form=form)


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(predict_risk(data.get("department", ""), data.get("finding", ""),
                                    int(data.get("month", 0))))
    except (ValueError, TypeError) as e:
        return jsonify(error=str(e)), 400


@app.get("/healthz")
def healthz():
    return jsonify(status="ok", model=meta["model"])


if __name__ == "__main__":
    app.run(debug=False)
