# step2_eda_visualize.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
file_path = 'F:\github projects\WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

# Convert TotalCharges to numeric and drop rows with NaNs (cleaning, just in case)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Basic info
print("\n✅ Cleaned Data for EDA")
print(df.shape)
print(df['Churn'].value_counts())

# Set style
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))

# --- 1. Churn Count Plot ---
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Customer Churn Count')
plt.savefig("churn_count.png")
plt.clf()

# --- 2. Churn by Gender ---
sns.countplot(data=df, x='gender', hue='Churn', palette='coolwarm')
plt.title('Churn by Gender')
plt.savefig("churn_gender.png")
plt.clf()

# --- 3. Churn by Contract Type ---
sns.countplot(data=df, x='Contract', hue='Churn', palette='Set1')
plt.title('Churn by Contract Type')
plt.xticks(rotation=15)
plt.savefig("churn_contract.png")
plt.clf()

# --- 4. Monthly Charges vs Churn ---
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', kde=True, multiple='stack', palette='Set2')
plt.title('Monthly Charges Distribution by Churn')
plt.savefig("monthly_charges_churn.png")
plt.clf()

# --- 5. Tenure vs Churn ---
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30, palette='Paired')
plt.title('Tenure Distribution by Churn')
plt.savefig("tenure_churn.png")
plt.clf()

# --- 6. Correlation Heatmap ---
numerical = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
corr = numerical.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig("correlation_matrix.png")
plt.clf()

print("\n📊 Plots saved in the current directory.")
