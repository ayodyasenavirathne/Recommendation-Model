import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split, cross_validate
import os
import pickle

# ========================= CONFIG =========================
PROCESSED_DIR = 'processed_data'
INTERACTIONS_FILE = f'{PROCESSED_DIR}/user_interactions.csv'
MODEL_SAVE_PATH = f'{PROCESSED_DIR}/cf_model.pkl'

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ====================== 1. LOAD DATA ======================
print("Loading user interactions...")
if not os.path.exists(INTERACTIONS_FILE):
    print("❌ Interactions file not found. Run data preparation script first!")
    exit()

interactions = pd.read_csv(INTERACTIONS_FILE)
print(f"Interactions shape: {interactions.shape}")
print(interactions.head())

# Ensure correct column names (adjust if your CSV differs)
# Expected: user_id, product_id, rating
if 'rating' not in interactions.columns:
    raise ValueError("CSV must have 'user_id', 'product_id', and 'rating' columns")

# ====================== 2. PREPARE FOR SURPRISE ======================
reader = Reader(rating_scale=(1, 5))  # Adjust if your ratings differ
data = Dataset.load_from_df(interactions[['user_id', 'product_id', 'rating']], reader)

# ====================== 3. TRAIN-TEST SPLIT & CROSS VALIDATION ======================
print("\nPerforming cross-validation...")
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Use SVD (Matrix Factorization) - best for sparse data
algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)

# Cross-validation for robust evaluation
cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print(f"\nMean RMSE: {np.mean(cv_results['test_rmse']):.4f}")
print(f"Mean MAE:  {np.mean(cv_results['test_mae']):.4f}")

# Train on full trainset
algo.fit(trainset)

# ====================== 4. EVALUATE ON TEST SET ======================
print("\nEvaluating on test set...")
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# ====================== 5. GENERATE RECOMMENDATIONS ======================
def get_top_n_recommendations(user_id, n=10):
    """Get top N recommendations for a user"""
    # Get all product ids
    all_products = interactions['product_id'].unique()
    
    # Products already interacted with by the user
    user_rated = interactions[interactions['user_id'] == user_id]['product_id'].values
    
    # Predict for items not yet rated
    predictions = []
    for product_id in all_products:
        if product_id not in user_rated:
            pred = algo.predict(user_id, product_id)
            predictions.append((product_id, pred.est))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

# Example usage
sample_user = interactions['user_id'].iloc[0]  # Pick first user
print(f"\nTop 5 recommendations for user {sample_user}:")
top_recs = get_top_n_recommendations(sample_user, n=5)
for product_id, score in top_recs:
    print(f"Product ID: {product_id} | Predicted Rating: {score:.2f}")

# ====================== 6. SAVE MODEL ======================
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(algo, f)
print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")

# ====================== 7. OPTIONAL: KNN BASELINE ======================
print("\nTraining KNN (Item-based) for comparison...")
sim_options = {'name': 'cosine', 'user_based': False}  # Item-based
knn = KNNBasic(sim_options=sim_options)
knn.fit(trainset)
print("KNN trained.")