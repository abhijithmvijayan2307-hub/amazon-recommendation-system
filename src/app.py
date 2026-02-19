# src/app.py

import streamlit as st
import pandas as pd
import joblib
from collaborative_filtering import recommend_products, get_available_products

# ==============================
# Paths for SVD model & data
# ==============================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "svd_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_amazon.csv")
PRODUCT_PATH = os.path.join(BASE_DIR, "..", "data", "products.csv")

# ==============================
# Load SVD model + data
# ==============================
svd_model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
products = pd.read_csv(PRODUCT_PATH)
product_map = dict(zip(products['asin'], products['title']))

# ==============================
# Streamlit App Title
# ==============================
st.title("Amazon Product Recommendation System")

# ==============================
# User Selection (for SVD)
# ==============================
user_list = df['reviewerID'].unique()
selected_user = st.selectbox("Select a User (for SVD recommendations):", user_list)

# ==============================
# Product Selection (for Item-CF)
# ==============================
product_list = get_available_products()
selected_product = st.selectbox("Select a Product (for Item-Based CF):", product_list)

# ==============================
# Item-Based Collaborative Filtering
# ==============================
st.subheader("Item-Based CF Recommendations:")
item_recs = recommend_products(selected_product, top_n=5)
for title, score in item_recs.items():
    clean_title = title.replace("&amp;", "&")
    st.write(f"• {clean_title} -> {score}")

# ==============================
# SVD User-Based Recommendations
# ==============================
st.subheader("User-Based SVD Recommendations:")

# Get products the user has already rated
user_products = df[df['reviewerID'] == selected_user]['asin'].values
all_products = df['asin'].unique()
products_to_predict = [p for p in all_products if p not in user_products]

predictions = []
for product in products_to_predict:
    pred = svd_model.predict(selected_user, product)
    title = product_map.get(product, product)
    predictions.append((title, pred.est))

# Sort by predicted rating
predictions.sort(key=lambda x: x[1], reverse=True)

# Show top 5
for title, score in predictions[:5]:
    st.write(f"• {title} -> Predicted Rating: {round(score,2)}")
