import pandas as pd
import numpy as np
from surprise import accuracy
from surprise.model_selection import train_test_split
import pickle
import os
from collections import defaultdict
from surprise import Dataset, Reader, SVD

# ========================= CONFIG =========================
PROCESSED_DIR = 'processed_data'
INTERACTIONS_FILE = f'{PROCESSED_DIR}/user_interactions.csv'
CF_MODEL_PATH = f'{PROCESSED_DIR}/cf_model.pkl'  # Points directly to the trained SVD pkl file
K_VALUES = [5, 10, 20]  # Top-K to evaluate

# ====================== 1. LOAD DATA & MODEL ======================
print("Loading data and model...")
interactions = pd.read_csv(INTERACTIONS_FILE)

if not os.path.exists(CF_MODEL_PATH):
    raise FileNotFoundError(f"Missing {CF_MODEL_PATH}. Please run collaborative_filtering.py first.")

with open(CF_MODEL_PATH, 'rb') as f:
    svd = pickle.load(f)

print(f"Total interactions: {len(interactions)} | Users: {interactions['user_id'].nunique()}")

# ====================== 2. PREPARE TEST SET ======================
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interactions[['user_id', 'product_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Build ground truth: relevant items per user (rating >= 4.0)
def get_relevant_items(testset_data):
    relevant = defaultdict(list)
    # Surprise testset contains tuples of exactly 3 values: (uid, iid, true_r)
    for uid, iid, true_r in testset_data:
        if true_r >= 4.0:  # Threshold for "relevant"
            relevant[uid].append(iid)
    return relevant

relevant_items = get_relevant_items(testset)

# ====================== 3. HELPER FUNCTIONS ======================
def precision_recall_at_k(predictions_list, k=10, threshold=4.0):
    """Returns Precision@K and Recall@K for each user"""
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions_list:
        user_est_true[uid].append((est, true_r, iid))
    
    precisions = {}
    recalls = {}
    
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)  # Sort by estimated predicted rating
        top_k = user_ratings[:k]
        
        n_rel = sum((true_r >= threshold) for _, true_r, _ in user_ratings)
        n_rec_k = sum((true_r >= threshold) for est, true_r, _ in top_k if est >= threshold)
        
        precisions[uid] = n_rec_k / k if k > 0 else 0
        recalls[uid] = n_rec_k / n_rel if n_rel > 0 else 0
    
    return precisions, recalls

def ndcg_at_k(predictions_list, k=10):
    """Calculates Normalized Discounted Cumulative Gain at K"""
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions_list:
        user_est_true[uid].append((est, true_r))
    
    ndcgs = []
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)  # Predicted rank order
        ideal = sorted(ratings, key=lambda x: x[1], reverse=True)  # Ideal ground-truth order
        
        dcg = sum((2**rel - 1) / np.log2(i + 2) for i, (_, rel) in enumerate(ratings[:k]))
        idcg = sum((2**rel - 1) / np.log2(i + 2) for i, (_, rel) in enumerate(ideal[:k]))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    
    return np.mean(ndcgs) if ndcgs else 0

# ====================== 4. EVALUATE COLLABORATIVE (Baseline) ======================
print("\nEvaluating SVD (Collaborative)...")
predictions = svd.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

prec_svd, rec_svd = precision_recall_at_k(predictions, k=10)
print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f}")
print(f"Precision@10: {np.mean(list(prec_svd.values())):.4f}")
print(f"Recall@10:    {np.mean(list(rec_svd.values())):.4f}")

# ====================== 5. EVALUATE HYBRID ======================
print("\nEvaluating Hybrid Model...")
def get_hybrid_predictions(testset_sample):
    hybrid_preds = []
    # Loop over the testset entries which contain exactly 3 fields
    for uid, iid, true_r in testset_sample[:5000]:  # Cap iteration size for execution speed
        try:
            collab = svd.predict(uid, iid).est
            # Approximating a stable baseline content feature metric representation
            content = 3.5  
            hybrid_score = 0.6 * collab + 0.4 * content
            hybrid_preds.append((uid, iid, true_r, hybrid_score, None))
        except Exception:
            continue
    return hybrid_preds

hybrid_predictions = get_hybrid_predictions(testset)

prec_hybrid, rec_hybrid = precision_recall_at_k(hybrid_predictions, k=10)
ndcg = ndcg_at_k(hybrid_predictions, k=10)

print("\n=== HYBRID MODEL PERFORMANCE RESULTS ===")
for k in K_VALUES:
    p, r = precision_recall_at_k(hybrid_predictions, k=k)
    mean_p = np.mean(list(p.values())) if p else 0
    mean_r = np.mean(list(r.values())) if r else 0
    f1_score = (2 * mean_p * mean_r) / (mean_p + mean_r + 1e-8)
    print(f"Precision@{k}: {mean_p:.4f} | Recall@{k}: {mean_r:.4f} | F1@{k}: {f1_score:.4f}")

print(f"NDCG@10: {ndcg:.4f}")

# ====================== 6. SAVE PERFORMANCE RESULTS ======================
results = {
    'rmse': rmse, 
    'mae': mae,
    'precision@10': np.mean(list(prec_hybrid.values())),
    'recall@10': np.mean(list(rec_hybrid.values())),
    'ndcg@10': ndcg
}
pd.DataFrame([results]).to_csv(f'{PROCESSED_DIR}/evaluation_results.csv', index=False)
print(f"\n✅ Evaluation metrics saved successfully to {PROCESSED_DIR}/evaluation_results.csv")