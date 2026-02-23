import pandas as pd
import os

# ==============================
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_amazon.csv")
PRODUCT_PATH = os.path.join(BASE_DIR, "..", "data", "products.csv")

# ==============================
# Load data once
# ==============================
df = pd.read_csv(DATA_PATH)
products = pd.read_csv(PRODUCT_PATH)

# Map ASIN -> Title
asin_to_title = dict(zip(products['asin'], products['title']))

# ==============================
# Popularity Model
# ==============================
# Popularity = average rating * number of ratings

rating_stats = df.groupby("asin").agg({
    "overall": ["mean", "count"]
})

rating_stats.columns = ["avg_rating", "rating_count"]

# Score formula (Amazon style weighted popularity)
rating_stats["score"] = rating_stats["avg_rating"] * rating_stats["rating_count"]

# Sort descending
popular_items = rating_stats.sort_values("score", ascending=False)


# ==============================
# Function required by app.py
# ==============================
def get_popular_items(top_n=5):
    """
    Returns top popular products (fallback recommendations)
    """

    top = popular_items.head(top_n)

    results = []
    for asin in top.index:
        title = asin_to_title.get(asin, asin)
        score = round(top.loc[asin, "avg_rating"], 2)
        results.append((title, score))

    return results


# ==============================
# Test
# ==============================
if __name__ == "__main__":
    print("\nTop Popular Items:\n")
    for item in get_popular_items(5):
        print(item)
