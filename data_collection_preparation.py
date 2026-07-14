import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ========================= CONFIG =========================
DATA_PATH = 'etsy.csv'  # Path to your main dataset
OUTPUT_DIR = 'processed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====================== 1. LOAD & EXPLORE ======================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# Generate an explicit product_id since the dataset doesn't have a direct numeric ID
df['product_id'] = range(1, len(df) + 1)

# ====================== 2. CLEANING ======================
print("Cleaning data...")
# Drop rows where critical metadata columns are missing
df = df.dropna(subset=['name', 'description', 'category'])

# Clean and convert price to numeric, keeping only valid positive prices
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df[df['price'] > 0]

# Combine textual features into a single string for content-based filtering
df['combined_features'] = (
    df['category'].fillna('') + " " +
    df['name'].fillna('') + " " +
    df['brand'].fillna('') + " " +
    df['description'].fillna('')
).str.lower()
print(f"Cleaned dataset shape: {df.shape}")

# ====================== 3. CONTENT-BASED FEATURES ======================
print("Generating TF-IDF features...")
# Extract top 5000 text features excluding common English stop words
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Save the extracted TF-IDF feature matrix array to the output directory
np.save(f'{OUTPUT_DIR}/tfidf_matrix.npy', tfidf_matrix.toarray())

# ====================== 4. SAMPLE SIMILARITY TEST ======================
# Function to fetch top N most similar items based on cosine similarity
def get_similar_products(p_id, top_n=5):
    # Locate the matrix index of the given product_id
    idx = df[df['product_id'] == p_id].index[0]
    # Calculate cosine similarity scores between this item and all other items
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    # Sort and get the indices of top N similar products (excluding the item itself)
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]
    return df.iloc[similar_indices][['product_id', 'name', 'category', 'price']]

print("\n--- Testing Recommendation (Sample) ---")
test_id = df.iloc[0]['product_id']
print(f"Products similar to '{df.iloc[0]['name']}':")
print(get_similar_products(test_id))

# ====================== 5. USER INTERACTIONS SIMULATION ======================
print("\nSimulating user interactions...")
np.random.seed(42)
# Simulate interaction records (2x the length of the dataset) for 500 unique users
users = np.random.randint(1001, 1500, size=len(df) * 2)  
product_ids = np.random.choice(df['product_id'].values, size=len(df) * 2)
ratings = np.random.randint(1, 6, size=len(df) * 2)

interactions = pd.DataFrame({
    'user_id': users,
    'product_id': product_ids,
    'rating': ratings,
    'timestamp': pd.date_range(start='2026-01-01', periods=len(df) * 2, freq='min')
})
# Drop potential duplicate reviews from the same user on the same product
interactions = interactions.drop_duplicates(subset=['user_id', 'product_id'])

# Save cleaned master datasets and simulated interactions as CSVs
interactions.to_csv(f'{OUTPUT_DIR}/user_interactions.csv', index=False)
df.to_csv(f'{OUTPUT_DIR}/cleaned_products.csv', index=False)

print("\n✅ Data preparation completed successfully!")