import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("cleaned_churn_data.csv")

# Safety check
print("✅ Columns in data:", df.columns.tolist())
print("✅ 'Churn' column NaNs:", df['Churn'].isna().sum())

# Define X and y
X = df.drop("Churn", axis=1)
y = df["Churn"]

# If any non-numeric columns in X, drop or encode them
X = df.drop("Churn", axis=1)

# Fit model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X, y)

# Feature importances
importance = pd.Series(logreg.coef_[0], index=X.columns)
importance = importance.sort_values()

# Plot
importance.plot(kind="barh", figsize=(10, 8), title="Logistic Regression Feature Importance")
plt.tight_layout()
plt.show()
