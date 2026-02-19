import pandas as pd
import os
import joblib

from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


# ==============================
# 1. Load cleaned dataset
# ==============================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned_amazon.csv")
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)


# ==============================
# 2. Prepare Surprise Dataset
# ==============================
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['reviewerID', 'asin', 'overall']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# ==============================
# 3. Train SVD Model
# ==============================
print("\nTraining SVD Model...\n")

model = SVD(
    n_factors=100,      # number of latent features
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02
)

model.fit(trainset)


# ==============================
# 4. Evaluate Model
# ==============================
predictions = model.test(testset)

rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"\nFinal RMSE: {rmse:.4f}")
print(f"Final MAE : {mae:.4f}")


# ==============================
# 5. Save Model
# ==============================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "svd_model.pkl")

joblib.dump(model, MODEL_PATH)

print("\nModel saved at:", MODEL_PATH)
