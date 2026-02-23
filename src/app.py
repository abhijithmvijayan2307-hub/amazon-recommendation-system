import streamlit as st
import pandas as pd
import joblib
import os
from collaborative_filtering import recommend_products, get_available_products

# ============================== 
# Page Config
# ==============================
st.set_page_config(page_title="Smart Product Recommender", page_icon="🛍️", layout="wide")

# ============================== 
# Custom CSS Styling
# ==============================
st.markdown("""
<style>
.main-title {font-size:40px !important; font-weight:700; color:#4CAF50;}
.subtitle {font-size:18px !important; color:gray;}
.card {
    background-color:#ffffff;
    padding:18px;
    border-radius:14px;
    box-shadow:0px 4px 14px rgba(0,0,0,0.08);
    margin-bottom:10px;
}
.metric-card {
    background: linear-gradient(135deg,#4CAF50,#2E7D32);
    padding:20px;
    border-radius:16px;
    color:white;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ============================== 
# Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "svd_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "cleaned_amazon.csv")
PRODUCT_PATH = os.path.join(BASE_DIR, "..", "data", "products.csv")

# ============================== 
# Cached Loading
# ==============================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    products = pd.read_csv(PRODUCT_PATH)
    return df, products

svd_model = load_model()
df, products = load_data()
product_map = dict(zip(products['asin'], products['title']))

# ============================== 
# Header Section
# ==============================
st.markdown('<p class="main-title">🛍️ Smart Recommendation Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI powered personalized & similar product suggestions</p>', unsafe_allow_html=True)
st.divider()

# ============================== 
# Sidebar Filters
# ==============================
st.sidebar.header("⚙️ Controls")
user_list = df['reviewerID'].drop_duplicates().sample(min(1000, len(df['reviewerID'].unique())), random_state=42)
selected_user = st.sidebar.selectbox("Choose User", user_list)

product_list = get_available_products()
selected_product = st.sidebar.selectbox("Choose Product", product_list)

generate = st.sidebar.button("✨ Generate Recommendations")

# ============================== 
# Dashboard Metrics
# ==============================
col1, col2, col3 = st.columns(3)
col1.markdown(f'<div class="metric-card"><h2>{df["reviewerID"].nunique()}</h2><p>Users</p></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="metric-card"><h2>{df["asin"].nunique()}</h2><p>Products</p></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="metric-card"><h2>{len(df)}</h2><p>Reviews</p></div>', unsafe_allow_html=True)

st.divider()

# ============================== 
# Generate Recommendations
# ==============================
if generate:

    # -------- Item Based CF --------
    st.subheader("🔎 Similar Products You May Like")
    item_recs = recommend_products(selected_product, top_n=5)

    cols = st.columns(5)
    for idx, (title, score) in enumerate(item_recs.items()):
        clean_title = str(title).replace("&amp;", "&")
        with cols[idx % 5]:
            st.markdown(f"""
            <div class="card">
            <b>{clean_title}</b><br><br>
            Similarity Score: <b>{round(score,3)}</b>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # -------- User Based SVD --------
    st.subheader("🤖 Personalized For You")

    user_products = df[df['reviewerID'] == selected_user]['asin'].values
    all_products = df['asin'].unique()
    products_to_predict = [p for p in all_products if p not in user_products][:500]

    predictions = []
    for product in products_to_predict:
        pred = svd_model.predict(selected_user, product)
        title = product_map.get(product, product)
        predictions.append((title, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    cols = st.columns(5)
    for idx, (title, score) in enumerate(predictions[:5]):
        with cols[idx % 5]:
            st.markdown(f"""
            <div class="card">
            <b>{title}</b><br><br>
            Predicted Rating ⭐ <b>{round(score,2)}</b>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Select filters from the sidebar and click **Generate Recommendations** to view results.")