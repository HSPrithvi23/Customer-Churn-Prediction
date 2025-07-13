import pandas as pd

file_path = 'F:\github projects\WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

# Display basic info
print("✅ Data Loaded Successfully!\n")
print("🔹 Dataset Shape:", df.shape)
print("\n🔹 Data Types & Non-Null Count:\n")
print(df.info())
print("\n🔹 First 5 Rows:\n")
print(df.head())

# Check missing values
print("\n🔹 Missing Values Per Column:\n")
print(df.isnull().sum())

# View churn class distribution
print("\n🔹 Churn Value Counts:\n")
print(df['Churn'].value_counts())

# Optional cleaning: convert TotalCharges to numeric (some rows might be empty/strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Recheck missing values after conversion
print("\n🔹 Missing After Conversion (TotalCharges might have NaN):\n")
print(df.isnull().sum())

# Drop rows with missing TotalCharges
df.dropna(inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Confirm clean shape
print("\n✅ Final Cleaned Dataset Shape:", df.shape)
