import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, SVDpp, NMF, accuracy
from surprise.model_selection import cross_validate, train_test_split
import os
import time

# ========================= CONFIG =========================
PROCESSED_DIR = 'processed_data'
INTERACTIONS_FILE = f'{PROCESSED_DIR}/user_interactions.csv'

# ====================== LOAD DATA ======================
print("Loading data...")
interactions = pd.read_csv(INTERACTIONS_FILE)
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interactions[['user_id', 'product_id', 'rating']], reader)

# ====================== DEFINE VARIANTS ======================
algorithms = {
    'SVD': SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42),
    'SVD++': SVDpp(n_factors=50, n_epochs=15, lr_all=0.007, reg_all=0.02, random_state=42),  # More accurate but slower
    'NMF': NMF(n_factors=100, n_epochs=20, reg_pu=0.06, reg_qi=0.06, random_state=42),     # Non-negative
}

results = []

for name, algo in algorithms.items():
    print(f"\n=== Training {name} ===")
    start = time.time()
    
    # Cross-validation
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    
    # Train-test split evaluation
    trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    duration = time.time() - start
    
    results.append({
        'Model': name,
        'Mean RMSE (CV)': np.mean(cv_results['test_rmse']),
        'Mean MAE (CV)': np.mean(cv_results['test_mae']),
        'Test RMSE': rmse,
        'Test MAE': mae,
        'Time (s)': round(duration, 2)
    })
    
    print(f"{name} - Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, Time: {duration:.1f}s")

# ====================== RESULTS TABLE ======================
results_df = pd.DataFrame(results)
print("\n📊 MATRIX FACTORIZATION COMPARISON")
print(results_df.round(4))

# Save results
results_df.to_csv(f'{PROCESSED_DIR}/mf_variants_comparison.csv', index=False)
print(f"\n✅ Results saved to {PROCESSED_DIR}/mf_variants_comparison.csv")