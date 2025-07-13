# step2_preprocessing.py

import pandas as pd

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert 'TotalCharges' to numeric, coerce errors (invalids -> NaN)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Drop customerID (not useful for modeling)
df.drop("customerID", axis=1, inplace=True)

# Encode target variable
df["Churn"] = df["Churn"].astype(str).str.strip()
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
print("🧪 Unique values in 'Churn' after mapping:", df["Churn"].unique())
print("❌ Rows with NaN in 'Churn':", df["Churn"].isna().sum())

df.dropna(subset=["Churn"], inplace=True)

# Convert categorical features to dummy/one-hot encoded columns
df = pd.get_dummies(df, drop_first=True)

# Save cleaned dataset
df.to_csv("cleaned_churn_data.csv", index=False)
print("✅ Cleaned data saved as 'cleaned_churn_data.csv'")
print("✅ Final dataset shape:", df.shape)
