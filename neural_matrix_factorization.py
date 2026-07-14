import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os

# ========================= CONFIG =========================
PROCESSED_DIR = 'processed_data'
INTERACTIONS_FILE = f'{PROCESSED_DIR}/user_interactions.csv'
MODEL_SAVE_PATH = f'{PROCESSED_DIR}/neumf_model.keras'

NUM_FACTORS = 32          # Embedding dimension
NUM_LAYERS = 3            # MLP layers
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

# ====================== 1. LOAD & PREPARE DATA ======================
print("Loading interactions...")
interactions = pd.read_csv(INTERACTIONS_FILE)

# Map to continuous IDs (important for embeddings)
user_ids = interactions['user_id'].unique()
item_ids = interactions['product_id'].unique()

user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

interactions['user_idx'] = interactions['user_id'].map(user_to_idx)
interactions['item_idx'] = interactions['product_id'].map(item_to_idx)

# Binary or rating target (NeuMF works well with implicit feedback too)
interactions['target'] = (interactions['rating'] >= 4).astype(int)  # 1 = positive interaction

print(f"Users: {len(user_ids)}, Items: {len(item_ids)}, Interactions: {len(interactions)}")

# Train-test split
train, test = train_test_split(interactions, test_size=0.2, random_state=42, stratify=interactions['target'])

# ====================== 2. BUILD NeuMF MODEL ======================
def build_neumf(num_users, num_items, num_factors=32, num_layers=3):
    # GMF Path (Linear)
    user_input_gmf = layers.Input(shape=(1,), dtype=tf.int32, name='user_gmf')
    item_input_gmf = layers.Input(shape=(1,), dtype=tf.int32, name='item_gmf')
    
    user_embedding_gmf = layers.Embedding(num_users, num_factors, name='user_embedding_gmf')(user_input_gmf)
    item_embedding_gmf = layers.Embedding(num_items, num_factors, name='item_embedding_gmf')(item_input_gmf)
    
    gmf_vector = layers.Multiply()([user_embedding_gmf, item_embedding_gmf])
    gmf_vector = layers.Flatten()(gmf_vector)
    
    # MLP Path (Non-linear)
    user_input_mlp = layers.Input(shape=(1,), dtype=tf.int32, name='user_mlp')
    item_input_mlp = layers.Input(shape=(1,), dtype=tf.int32, name='item_mlp')
    
    user_embedding_mlp = layers.Embedding(num_users, num_factors * 2, name='user_embedding_mlp')(user_input_mlp)
    item_embedding_mlp = layers.Embedding(num_items, num_factors * 2, name='item_embedding_mlp')(item_input_mlp)
    
    mlp_vector = layers.Concatenate()([layers.Flatten()(user_embedding_mlp), layers.Flatten()(item_embedding_mlp)])
    
    for i in range(num_layers):
        mlp_vector = layers.Dense(2 ** (num_layers - i) * num_factors, activation='relu')(mlp_vector)
        mlp_vector = layers.Dropout(0.2)(mlp_vector)
    
    # Concatenate GMF + MLP
    neumf_vector = layers.Concatenate()([gmf_vector, mlp_vector])
    output = layers.Dense(1, activation='sigmoid', name='prediction')(neumf_vector)
    
    model = Model(inputs=[user_input_gmf, item_input_gmf, user_input_mlp, item_input_mlp], outputs=output)
    return model

model = build_neumf(len(user_ids), len(item_ids), NUM_FACTORS, NUM_LAYERS)
model.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
model.summary()

# ====================== 3. TRAIN MODEL ======================
print("\nTraining NeuMF...")

X_train = [train['user_idx'].values, train['item_idx'].values, 
           train['user_idx'].values, train['item_idx'].values]
y_train = train['target'].values

X_test = [test['user_idx'].values, test['item_idx'].values, 
          test['user_idx'].values, test['item_idx'].values]
y_test = test['target'].values

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# ====================== 4. EVALUATE & SAVE ======================
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
print(f"\n✅ Test AUC: {test_auc:.4f} | Accuracy: {test_acc:.4f}")

model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# ====================== 5. INFERENCE (Recommendations) ======================
def get_neumf_recommendations(user_id, top_n=10):
    user_idx = user_to_idx[user_id]
    all_items = np.arange(len(item_ids))
    
    # Predict for all items
    user_input = np.full(len(all_items), user_idx)
    predictions = model.predict([user_input, all_items, user_input, all_items], batch_size=1024, verbose=0).flatten()
    
    # Sort and get top-N (exclude already interacted if desired)
    top_indices = predictions.argsort()[-top_n:][::-1]
    recommended_items = [item_ids[i] for i in top_indices]
    
    return recommended_items, predictions[top_indices]

# Example
sample_user = interactions['user_id'].iloc[0]
recs, scores = get_neumf_recommendations(sample_user, top_n=5)
print(f"\nTop recommendations for user {sample_user}:")
for item, score in zip(recs, scores):
    print(f"Product ID: {item} | Score: {score:.4f}")