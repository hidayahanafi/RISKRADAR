import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
import joblib

df = pd.read_csv("audit_dataset_updated.csv")
df['Audit_Date'] = pd.to_datetime(df['Audit_Date'])
df['Year'] = df['Audit_Date'].dt.year
df['Month'] = df['Audit_Date'].dt.month
df_filtered = df[df['Year'].isin([2022, 2023])].copy()

le_dept = LabelEncoder()
le_risk_level = LabelEncoder()

df['Department'] = le_dept.fit_transform(df['Department'])
df['Risk_Level'] = le_risk_level.fit_transform(df['Risk_Level'])

joblib.dump(le_dept, 'label_encoder_department.pkl')
joblib.dump(le_risk_level, 'label_encoder_risk.pkl')

X = df[['Department', 'Year']]
y = df['Risk_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    score = f1_score(y_test, predictions, average='macro')
    
    print(f"\n{name} F1 Score (macro): {score:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=le_risk_level.classes_, zero_division=1))

    if score > best_score:
        best_score = score
        best_model = model
        best_model_name = name

joblib.dump(best_model, 'best_model.pkl')

print(f"\n✅ Best model: {best_model_name} (F1 macro: {best_score:.4f})")
