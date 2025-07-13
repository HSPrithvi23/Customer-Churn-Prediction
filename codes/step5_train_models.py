# step5_train_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load cleaned dataset
df = pd.read_csv("cleaned_churn_data.csv")
print("✅ Data loaded")

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate
for name, model in models.items():
    print(f"\n🔍 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"📊 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("🧾 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("📉 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
