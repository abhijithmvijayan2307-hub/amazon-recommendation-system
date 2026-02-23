# 🛒 Amazon Product Recommendation System

A machine learning–based recommendation system that suggests products using multiple recommendation strategies including Popularity-Based Filtering, Collaborative Filtering, and SVD Matrix Factorization.

---

## 📌 Project Overview

This project implements a modular recommendation system inspired by real-world e-commerce platforms like Amazon.

The system supports multiple recommendation approaches:

- ⭐ Popularity-Based Recommendation
- 👥 User-Based Collaborative Filtering
- 🎯 SVD Matrix Factorization Model

The application is built using **Python**, **Scikit-Learn**, and **Streamlit** for interactive UI.

---

##  Features

- Multi-mode recommendation system
- Modular architecture (separated ML logic & UI)
- Clean and interactive Streamlit interface
- Matrix factorization using SVD
- Scalable design for adding new recommendation models

---

##  Recommendation Techniques Used

### 1️⃣ Popularity-Based Filtering
Recommends top-rated or most frequently interacted products.

### 2️⃣ Collaborative Filtering
Uses user-item interaction similarity to recommend products.

### 3️⃣ SVD (Singular Value Decomposition)
Applies matrix factorization to predict user preferences.

---

##  Project Structure


amazon-recommendation-system/
│
├── src/
│ ├── app.py # Streamlit application
│ ├── load_data.py # Data loading utilities
│ ├── prepare_products.py # Data preprocessing
│ ├── popularity_model.py # Popularity-based recommender
│ ├── collaborative_filtering.py # CF implementation
│ ├── compute_similarity.py # Similarity calculations
│ ├── svd_model.py # SVD model logic
│ ├── train_svd.py # SVD training script
│ └── recommend.py # Unified recommendation interface
│
├── requirements.txt
└── README.md


---

## 🖥️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/amazon-recommendation-system.git
cd amazon-recommendation-system
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
streamlit run src/app.py
 Tech Stack

Python

NumPy

Pandas

Scikit-Learn

Scikit-Surprise (for SVD)

Streamlit


---

## 🖥️ Future Improvements

Hybrid recommendation system

Model evaluation metrics (RMSE, Precision@K)

Deployment using Streamlit Cloud

Docker support

REST API integration


---

## Author

Abhijith M Vijayan
Machine Learning Enthusiast | Data Science | Recommendation Systems
