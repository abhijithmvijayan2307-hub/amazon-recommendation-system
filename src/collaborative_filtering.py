# src/collaborative_filtering.py

import pandas as pd
import joblib
import os

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_PATH = os.path.join(BASE_DIR, "..", "models", "item_cf_topk.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_amazon.csv")
PRODUCT_PATH = os.path.join(BASE_DIR, "..", "data", "products.csv")

# ==============================
# Load precomputed similarity
# ==============================
print("Loading precomputed item similarity...")
similar_items, asin_to_title = joblib.load(SIM_PATH)
print("Loaded similarity data.")

# Load raw data for popularity fallback
df = pd.read_csv(DATA_PATH)
products = pd.read_csv(PRODUCT_PATH)

# Inverse mapping: Title → ASIN
title_to_asin = {v: k for k, v in asin_to_title.items()}

# ==============================
# Precompute top popular products (fallback)
# ==============================
top_popular_asins = (
    df.groupby('asin')['overall']
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .index
    .tolist()
)

# ==============================
# Recommendation Function
# ==============================
def recommend_products(product_title, top_n=5):
    """
    Returns top N similar products for a given product.
    Falls back to popularity if the product is missing.
    """
    asin = title_to_asin.get(product_title)

    # Fallback if product not in similarity data
    if not asin or asin not in similar_items:
        results = {asin_to_title.get(a, a): "Popular Item" for a in top_popular_asins[:top_n]}
        return results

    # Otherwise, return precomputed top-N similar items
    results = {}
    for other_asin, score in similar_items[asin][:top_n]:
        title = asin_to_title.get(other_asin, other_asin)
        results[title] = round(score, 3)
    return results

# ==============================
# Safe product list for Streamlit selectbox
# ==============================
def get_available_products():
    """Returns only products that exist in similarity data"""
    return [title for title, asin in title_to_asin.items() if asin in similar_items]

# ==============================
# Test block
# ==============================
if __name__ == "__main__":
    sample_title = list(title_to_asin.keys())[100]
    print("\nSelected Product:", sample_title)
    recs = recommend_products(sample_title)
    print("\nSimilar Products:")
    for title, score in recs.items():
        print(title, "->", score)
