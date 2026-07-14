import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from collections import defaultdict

# ========================= CONFIG =========================
PROCESSED_DIR = 'processed_data'
PRODUCTS_FILE = f'{PROCESSED_DIR}/cleaned_products.csv'
INTERACTIONS_FILE = f'{PROCESSED_DIR}/user_interactions.csv'
MODEL_SAVE_PATH = f'{PROCESSED_DIR}/final_hybrid_recommender.pkl'

MAX_HISTORY_LEN = 20
WEIGHT_CONTENT = 0.35
WEIGHT_HISTORY_ATTENTION = 0.45
WEIGHT_COLLAB = 0.20   # Residual traditional collab signal

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ====================== 1. LOAD DATA ======================
print("Loading data...")
products = pd.read_csv(PRODUCTS_FILE)
interactions = pd.read_csv(INTERACTIONS_FILE)

# Map IDs
user_ids = interactions['user_id'].unique()
item_ids = products['product_id'].unique()  # Adjust column if needed
user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

interactions['user_idx'] = interactions['user_id'].map(user_to_idx)
interactions['item_idx'] = interactions['product_id'].map(item_to_idx)

# ====================== 2. CONTENT-BASED (TF-IDF) ======================
print("Building TF-IDF Content Model...")
if 'combined_features' not in products.columns:
    products['combined_features'] = (
        products.get('category', '').fillna('') + " " +
        products.get('tags', '').fillna('') + " " +
        products.get('description', '').fillna('')
    ).str.lower()

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(products['combined_features'])

def content_score(target_idx, user_history_indices):
    """Average similarity between target and user's past items"""
    if len(user_history_indices) == 0:
        return 0.5
    sims = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix[user_history_indices]).flatten()
    return np.mean(sims)

# ====================== 3. USER HISTORY ATTENTION MODEL ======================
print("Loading / Building User History Attention Model...")

def build_history_attention_model(num_users, num_items, max_len=MAX_HISTORY_LEN, num_factors=32):
    user_input = layers.Input(shape=(1,), dtype=tf.int32, name='user')
    history_input = layers.Input(shape=(max_len,), dtype=tf.int32, name='history')
    target_input = layers.Input(shape=(1,), dtype=tf.int32, name='target')

    user_emb = layers.Embedding(num_users, num_factors)(user_input)
    user_emb = layers.Flatten()(user_emb)

    item_emb = layers.Embedding(num_items + 1, num_factors, mask_zero=True)(history_input)
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=num_factors, dropout=0.1)(item_emb, item_emb)
    history_vec = layers.GlobalAveragePooling1D()(attn_out)

    target_emb = layers.Embedding(num_items + 1, num_factors)(target_input)
    target_emb = layers.Flatten()(target_emb)

    combined = layers.Concatenate()([user_emb, history_vec, target_emb])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[user_input, history_input, target_input], outputs=output)
    return model

# For now we create it (in real project you would load trained weights)
attention_model = build_history_attention_model(len(user_ids), len(item_ids))
# attention_model.load_weights(...)  # Uncomment when you have trained weights

# ====================== 4. HYBRID RECOMMENDATION ENGINE ======================
class HybridRecommender:
    def __init__(self, products, interactions, tfidf_matrix, attention_model):
        self.products = products
        self.interactions = interactions
        self.tfidf_matrix = tfidf_matrix
        self.attention_model = attention_model
        self.user_history = self._build_user_history()

    def _build_user_history(self):
        history = defaultdict(list)
        for _, row in self.interactions.iterrows():
            history[row['user_id']].append(row['product_id'])
        return history

    def get_recommendations(self, user_id, top_n=10):
        if user_id not in self.user_history:
            # Cold-start: pure content-based
            return self._content_only_recs(top_n)

        user_history_ids = self.user_history[user_id][-MAX_HISTORY_LEN:]
        user_history_indices = self.products[self.products['product_id'].isin(user_history_ids)].index.tolist()

        all_items = self.products['product_id'].unique()
        user_rated = set(user_history_ids)

        candidates = []
        user_idx = np.array([user_to_idx.get(user_id, 0)])

        for prod_id in all_items:
            if prod_id in user_rated:
                continue

            try:
                prod_idx = self.products[self.products['product_id'] == prod_id].index[0]
                target_idx = np.array([prod_idx])

                # 1. Content Score
                content_sc = content_score(prod_idx, user_history_indices)

                # 2. History Attention Score
                hist_padded = np.zeros(MAX_HISTORY_LEN, dtype=int)
                hist_ids = [item_to_idx.get(pid, 0) for pid in user_history_ids[-MAX_HISTORY_LEN:]]
                hist_padded[-len(hist_ids):] = hist_ids
                hist_input = np.array([hist_padded])

                attn_score = self.attention_model.predict(
                    [user_idx, hist_input, target_idx], verbose=0
                )[0][0]

                # 3. Simple Collab fallback (average rating)
                collab_sc = self.interactions[self.interactions['product_id'] == prod_id]['rating'].mean()
                collab_sc = collab_sc if not np.isnan(collab_sc) else 3.0

                # Final Hybrid Score
                hybrid_score = (WEIGHT_CONTENT * content_sc +
                               WEIGHT_HISTORY_ATTENTION * attn_score +
                               WEIGHT_COLLAB * (collab_sc / 5.0))

                candidates.append((prod_id, hybrid_score, content_sc, attn_score, collab_sc))

            except:
                continue

        # Sort and return top-N with metadata
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_recs = candidates[:top_n]

        result = []
        for pid, hscore, c, a, col in top_recs:
            info = self.products[self.products['product_id'] == pid].iloc[0]
            result.append({
                'product_id': pid,
                'title': info.get('title', 'N/A'),
                'category': info.get('category', 'N/A'),
                'hybrid_score': round(hscore, 4),
                'content_score': round(c, 4),
                'attention_score': round(a, 4),
                'collab_score': round(col, 2)
            })
        return result

    def _content_only_recs(self, top_n=10):
        # Fallback for new users
        print("New user - using content-based recommendations")
        # Return popular or random high-quality items
        popular = self.products.sample(top_n)
        return [{'product_id': row['product_id'], 'title': row.get('title', 'N/A'), 
                 'category': row.get('category', 'N/A'), 'hybrid_score': 0.8} 
                for _, row in popular.iterrows()]

# ====================== 5. INITIALIZE & TEST ======================
recommender = HybridRecommender(products, interactions, tfidf_matrix, attention_model)

# Test
sample_user = interactions['user_id'].iloc[0] if len(interactions) > 0 else user_ids[0]
print(f"\n🔥 Generating Hybrid Recommendations for User: {sample_user}")

recommendations = recommender.get_recommendations(sample_user, top_n=8)

for i, rec in enumerate(recommendations, 1):
    print(f"{i:2d}. {rec['title'][:60]:60} | Cat: {rec['category'][:25]:25} | "
          f"Score: {rec['hybrid_score']:.4f}")

# ====================== 6. SAVE FINAL HYBRID SYSTEM ======================
with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(recommender, f)

print(f"\n✅ Final Hybrid Recommender saved to {MODEL_SAVE_PATH}")