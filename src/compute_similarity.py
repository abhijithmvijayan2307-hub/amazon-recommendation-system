# src/compute_similarity.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
from tqdm import tqdm

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_amazon.csv")
PRODUCT_PATH = os.path.join(BASE_DIR, "..", "data", "products.csv")
SAVE_PATH = os.path.join(BASE_DIR, "..", "models", "item_cf_topk.pkl")

# ==============================
# Load data
# ==============================
df = pd.read_csv(DATA_PATH)
products = pd.read_csv(PRODUCT_PATH)

# Create mapping: ASIN → Title
asin_to_title = dict(zip(products['asin'], products['title']))

# ==============================
# Create User-Item Matrix
# ==============================
user_item_matrix = df.pivot_table(
    index='reviewerID',
    columns='asin',
    values='overall'
).fillna(0)

item_matrix = user_item_matrix.T
item_index = list(item_matrix.index)

# ==============================
# Compute Cosine Similarity
# ==============================
print("Computing cosine similarity (may take a few minutes)...")
similarity_matrix = cosine_similarity(item_matrix)

# ==============================
# Save Top-K Similar Items
# ==============================
top_k = 20  # number of similar items to keep
similar_items = {}

for i in tqdm(range(len(item_index))):
    scores = list(enumerate(similarity_matrix[i]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_k+1]  # skip itself
    similar_items[item_index[i]] = [(item_index[j], float(score)) for j, score in scores]

# Save for fast access in Streamlit
joblib.dump((similar_items, asin_to_title), SAVE_PATH)

print(f"Saved top-{top_k} similarity matrix to {SAVE_PATH}")
