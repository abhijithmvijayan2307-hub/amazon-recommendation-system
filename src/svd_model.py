import pandas as pd
import os
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# -----------------------------
# Load Cleaned Dataset
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "cleaned_amazon.csv")

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

# -----------------------------
# Prepare Data for Surprise
# -----------------------------
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['reviewerID', 'asin', 'overall']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# -----------------------------
# Train SVD Model
# -----------------------------
model = SVD(
    n_factors=100,     # latent features
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02
)

print("\nTraining SVD model...")
model.fit(trainset)

# -----------------------------
# Evaluate Model
# -----------------------------
predictions = model.test(testset)

print("\nModel Evaluation:")
accuracy.rmse(predictions)
accuracy.mae(predictions)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_products(user_id, df, model, top_n=5):
    all_products = df['asin'].unique()
    user_products = df[df['reviewerID'] == user_id]['asin'].values

    products_to_predict = [p for p in all_products if p not in user_products]

    predictions = []
    for product in products_to_predict:
        pred = model.predict(user_id, product)
        predictions.append((product, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:top_n]

# -----------------------------
# Test Recommendation
# -----------------------------
sample_user = df['reviewerID'].iloc[0]
print("\nRecommendations for user:", sample_user)

recommendations = recommend_products(sample_user, df, model)

for product, score in recommendations:
    print(product, "Predicted Rating:", round(score, 3))
