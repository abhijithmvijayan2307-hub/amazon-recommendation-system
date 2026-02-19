import pandas as pd
import os

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "cleaned_amazon.csv")

# Load cleaned dataset
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

# -----------------------------
# Popularity Model
# -----------------------------

# Calculate average rating and total ratings per product
product_stats = df.groupby('asin').agg({
    'overall': ['mean', 'count']
})

product_stats.columns = ['average_rating', 'rating_count']
product_stats = product_stats.reset_index()

# Sort by highest average rating & rating count
product_stats = product_stats.sort_values(
    by=['average_rating', 'rating_count'],
    ascending=False
)

print("\nTop 10 Recommended Products (Popularity Based):")
print(product_stats.head(10))
