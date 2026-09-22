import pandas as pd

# =========================
# 1. Load Dataset
# =========================
file_path = "xAPI-Edu-Data.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== UNIQUE VALUES ==========")

for column in df.columns:
    print(f"\n--- {column} ---")
    print(df[column].unique()[:20])

print("\n========== DATASET INFO ==========")
df.info()

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe(include="all"))