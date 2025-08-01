from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

departments = list(label_encoders['Department'].classes_)

@app.route('/')
def home():
    from datetime import datetime
    next_year = datetime.now().year + 1
    return render_template('index.html', departments=departments, year=next_year, prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    department = request.form['department']
    year = int(request.form['year'])

    department_encoded = label_encoders['Department'].transform([department])[0]

    most_common_audit_finding = label_encoders['Audit_Finding'].classes_[0]  
    audit_finding_encoded = label_encoders['Audit_Finding'].transform([most_common_audit_finding])[0]

    input_data = np.array([[department_encoded, audit_finding_encoded, year]])
    prediction_encoded = model.predict(input_data)[0]
    prediction = label_encoders['Risk_Level'].inverse_transform([prediction_encoded])[0]

    return render_template('index.html', departments=departments, year=year, prediction_text=f'Predicted Risk Level for {year}: {prediction}')
