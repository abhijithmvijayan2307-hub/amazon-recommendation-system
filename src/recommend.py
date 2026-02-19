import pandas as pd
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "svd_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_amazon.csv")
PRODUCT_PATH = os.path.join(BASE_DIR, "..", "data", "products.csv")

# Load model and data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
products = pd.read_csv(PRODUCT_PATH)

# Fill missing titles
products['title'] = products['title'].fillna(products['asin'])

# Mapping: ASIN → Title
product_map = dict(zip(products['asin'], products['title']))

# Recommendation function
def get_recommendations(user_id, top_n=5):
    if user_id not in df['reviewerID'].values:
        return [("User not found in dataset", 0)]

    all_products = df['asin'].unique()
    user_products = df[df['reviewerID'] == user_id]['asin'].values

    products_to_predict = [p for p in all_products if p not in user_products]

    predictions = [(product_map.get(p, p), model.predict(user_id, p).est) for p in products_to_predict]

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]

# Test block
if __name__ == "__main__":
    sample_user = df['reviewerID'].iloc[0]
    print(f"\nRecommendations for user: {sample_user}")

    recs = get_recommendations(sample_user)
    for product, score in recs:
        print(product, "Predicted Rating:", round(score, 3))
