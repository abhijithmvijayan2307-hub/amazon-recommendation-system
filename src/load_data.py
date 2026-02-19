import pandas as pd
import json
from tqdm import tqdm
import os

# -----------------------------
# File Path Setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "Electronics_5.json.gz")

print("Reading from:", file_path)

# -----------------------------
# Load JSON dataset
# -----------------------------
data = []

with open(file_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        data.append(json.loads(line))

df = pd.DataFrame(data)

print("\nOriginal Shape:", df.shape)

# -----------------------------
# Keep only needed columns
# -----------------------------
df = df[['reviewerID', 'asin', 'overall']]
print("After selecting required columns:", df.shape)

# Remove missing values
df = df.dropna()

# =====================================================
# DENSE FILTERING  (VERY IMPORTANT FOR COLLAB FILTERING)
# =====================================================
print("\nApplying dense filtering...")

# Users with >= 20 ratings
user_counts = df['reviewerID'].value_counts()
active_users = user_counts[user_counts >= 20].index
df = df[df['reviewerID'].isin(active_users)]

# Products with >= 20 ratings
product_counts = df['asin'].value_counts()
popular_products = product_counts[product_counts >= 20].index
df = df[df['asin'].isin(popular_products)]

print("After dense filtering:", df.shape)

# Sample manageable dense dataset
df = df.sample(n=80000, random_state=42)
print("Final Dense Shape:", df.shape)

# -----------------------------
# Save cleaned dataset
# -----------------------------
output_path = os.path.join(BASE_DIR, "..", "data", "cleaned_amazon.csv")
df.to_csv(output_path, index=False)

print("\nSaved cleaned dataset to:", output_path)

print("\nColumns:", df.columns)
print("\nFirst rows:")
print(df.head())
