# Step 2: Data Cleaning & Preprocessing

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('G:\Accenture\Marketing_Campaign_Analytics_Project\marketing_campaign_dataset.csv')

# --- 1. Convert Date column to datetime ---
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

# --- 2. Clean Acquisition_Cost (remove $ and commas, convert to float) ---
df['Acquisition_Cost'] = df['Acquisition_Cost'].replace('[\$,]', '', regex=True).astype(float)

# --- 3. Ensure numeric columns are in proper dtype ---
numeric_cols = ['Conversion_Rate', 'ROI', 'Clicks', 'Impressions', 'Engagement_Score']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- 4. Handle categorical variables ---
categorical_cols = ['Company', 'Campaign_Type', 'Target_Audience', 
                    'Channel_Used', 'Location', 'Language', 'Customer_Segment']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# --- 5. Handle missing values ---
# Example: fill numeric NaNs with median, categorical NaNs with mode
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col].fillna(df[col].median(), inplace=True)
    elif df[col].dtype.name == 'category':
        df[col].fillna(df[col].mode()[0], inplace=True)

# --- 6. Feature Engineering Example: Cost per Click ---
df['Cost_per_Click'] = np.where(df['Clicks'] > 0, 
                                df['Acquisition_Cost'] / df['Clicks'], 
                                np.nan)

# --- 7. Final Check ---
print(df.info())
print(df.head())
