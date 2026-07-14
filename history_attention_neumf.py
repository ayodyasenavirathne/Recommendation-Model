import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from sklearn.model_selection import train_test_split
import os

# ========================= CONFIG =========================
PROCESSED_DIR = 'processed_data'
INTERACTIONS_FILE = f'{PROCESSED_DIR}/user_interactions.csv'
MODEL_SAVE_PATH = f'{PROCESSED_DIR}/user_history_attention_neumf.keras'

MAX_HISTORY_LEN = 20   # Max past items per user
NUM_FACTORS = 32
NUM_HEADS = 4
BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 0.001

# ====================== 1. LOAD & CREATE USER HISTORIES ======================
interactions = pd.read_csv(INTERACTIONS_FILE)

# Sort by user (add pseudo-timestamp if needed)
interactions = interactions.sort_values(by=['user_id'])

user_ids = interactions['user_id'].unique()
item_ids = interactions['product_id'].unique()

user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

interactions['user_idx'] = interactions['user_id'].map(user_to_idx)
interactions['item_idx'] = interactions['product_id'].map(item_to_idx)

# Create sequences per user (last item as target, previous as history)
def create_sequences(df, max_len=MAX_HISTORY_LEN):
    sequences = []
    for user, group in df.groupby('user_idx'):
        items = group['item_idx'].values
        for i in range(1, len(items)):
            history = items[:i][-max_len:]  # Last max_len items
            target = items[i]
            pad_len = max_len - len(history)
            if pad_len > 0:
                history = np.pad(history, (pad_len, 0), constant_values=0)
            sequences.append({
                'user_idx': user,
                'history': history,
                'target_item': target,
                'label': 1
            })
    return pd.DataFrame(sequences)

seq_df = create_sequences(interactions)

# Negative sampling (same as before)
def add_negatives(seq_df, neg_ratio=3):
    negatives = []
    for _, row in seq_df.iterrows():
        for _ in range(neg_ratio):
            neg_item = np.random.choice(len(item_ids))
            negatives.append({
                'user_idx': row['user_idx'],
                'history': row['history'],
                'target_item': neg_item,
                'label': 0
            })
    return pd.concat([seq_df, pd.DataFrame(negatives)])

data = add_negatives(seq_df)

train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

# ====================== 2. MODEL WITH USER HISTORY ATTENTION ======================
def build_user_history_attention_model(num_users, num_items, max_len=MAX_HISTORY_LEN, 
                                       num_factors=32, num_heads=4):
    # User ID embedding (static)
    user_input = layers.Input(shape=(1,), dtype=tf.int32, name='user')
    user_emb = layers.Embedding(num_users, num_factors)(user_input)
    user_emb = layers.Flatten()(user_emb)
    
    # History sequence
    history_input = layers.Input(shape=(max_len,), dtype=tf.int32, name='history')
    item_emb = layers.Embedding(num_items + 1, num_factors, mask_zero=True)(history_input)  # +1 for padding
    
    # Self-Attention over history
    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=num_factors,
        dropout=0.1
    )(item_emb, item_emb)
    
    # Pool attended history
    history_vec = layers.GlobalAveragePooling1D()(attn_out)
    
    # Target item
    target_input = layers.Input(shape=(1,), dtype=tf.int32, name='target_item')
    target_emb = layers.Embedding(num_items + 1, num_factors)(target_input)
    target_emb = layers.Flatten()(target_emb)
    
    # Fusion
    combined = layers.Concatenate()([user_emb, history_vec, target_emb])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[user_input, history_input, target_input], outputs=output)
    return model

model = build_user_history_attention_model(len(user_ids), len(item_ids), MAX_HISTORY_LEN, NUM_FACTORS, NUM_HEADS)
model.compile(optimizer=optimizers.Adam(LEARNING_RATE), 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.summary()

# ====================== 3. TRAINING ======================
X_train = [
    train['user_idx'].values, 
    np.stack(train['history'].values), 
    train['target_item'].values
]
y_train = train['label'].values

X_test = [
    test['user_idx'].values, 
    np.stack(test['history'].values), 
    test['target_item'].values
]
y_test = test['label'].values

early_stop = callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(X_test, y_test), callbacks=[early_stop])

model.save(MODEL_SAVE_PATH)
print(f"✅ User History Attention model saved to {MODEL_SAVE_PATH}")