# ================================
# STEP 1: IMPORT LIBRARIES
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

print("Libraries loaded successfully")

# ================================
# STEP 2: LOAD DATASET
# ================================

df = pd.read_csv("startup_dataset.csv", low_memory=False)

print("Dataset Loaded Successfully")
# ================================
# STEP 3: BASIC INSPECTION
# ================================

print("Shape of dataset:")
print(df.shape)

print("\nColumn Information:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())
# ================================
# STEP 4: CHECK STATUS DISTRIBUTION
# ================================

print("\nStartup Status Distribution:")
print(df['status'].value_counts())
# ================================
# CONVERT FUNDING TO NUMERIC
# ================================

df['funding_total_usd'] = pd.to_numeric(df['funding_total_usd'], errors='coerce')

print("Funding column converted.")
print(df['funding_total_usd'].describe())

missing_percentage = df['funding_total_usd'].isna().mean() * 100
print(f"Missing Funding Percentage: {missing_percentage:.2f}%")

# ================================
# CONVERT DATE COLUMNS
# ================================

df['founded_at'] = pd.to_datetime(df['founded_at'], errors='coerce')
df['first_funding_at'] = pd.to_datetime(df['first_funding_at'], errors='coerce')
df['last_funding_at'] = pd.to_datetime(df['last_funding_at'], errors='coerce')

print("Date columns converted.")
df['founded_at'].isna().mean() * 100
missing_founded = df['founded_at'].isna().mean() * 100
print(f"Missing Founded Date Percentage: {missing_founded:.2f}%")

# =================================
# CREATE CLEAN WORKING DATASET
# =================================

analysis_df = df[
    (df['funding_total_usd'].notna()) &
    (df['founded_at'].notna())
].copy()

print("Original shape:", df.shape)
print("Cleaned shape:", analysis_df.shape)

# =================================
# CREATE FAILURE VARIABLE
# =================================

analysis_df['failure'] = np.where(analysis_df['status'] == 'closed', 1, 0)

print(analysis_df['failure'].value_counts())

# =================================
# CREATE SURVIVAL DURATION
# =================================

today = pd.to_datetime("today")

analysis_df['end_date'] = np.where(
    analysis_df['failure'] == 1,
    analysis_df['last_funding_at'],
    today
)

analysis_df['end_date'] = pd.to_datetime(analysis_df['end_date'], errors='coerce')

analysis_df['survival_years'] = (
    (analysis_df['end_date'] - analysis_df['founded_at']).dt.days
) / 365

print(analysis_df['survival_years'].describe())

# =================================
# FIX SURVIVAL LOGIC
# =================================

# Remove rows where end_date is earlier than founded_at
analysis_df = analysis_df[
    analysis_df['end_date'] >= analysis_df['founded_at']
]

# Remove unrealistic survival values
analysis_df = analysis_df[
    (analysis_df['survival_years'] >= 0) &
    (analysis_df['survival_years'] <= 50)
]

print("New shape after survival cleaning:", analysis_df.shape)
print(analysis_df['survival_years'].describe())

result = analysis_df.groupby('failure')['survival_years'].describe()
print(result)

correlation = analysis_df[['funding_total_usd', 'survival_years']].corr()
print(correlation)

failure_prob = analysis_df.groupby('failure')['funding_total_usd'].mean()
print(failure_prob)

check_median =analysis_df.groupby('failure')['funding_total_usd'].median()
print(check_median)

# Remove rows with missing funding
temp_df = analysis_df.dropna(subset=['funding_total_usd'])

# Create funding quartiles
temp_df['funding_category'] = pd.qcut(
    temp_df['funding_total_usd'],
    q=4,
    labels=['Low', 'Mid-Low', 'Mid-High', 'High']
)

# Calculate failure rate
result = temp_df.groupby('funding_category')['failure'].mean()

print(result)


