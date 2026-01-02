import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file_path = "C:\EV\programs\python\StudentsPerformance.csv"
df = pd.read_csv(file_path)

print("===== FIRST FIVE ROWS =====")
print(df.head())

print("\n===== BASIC INFO =====")
print(df.info())

print("\n===== SUMMARY STATISTICS =====")
print(df.describe(include='all'))

print("\n===== NULL VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.show()

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[col])
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()
