import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from sklearn.model_selection import train_test_split
import os

# ========================= CONFIG =========================
PROCESSED_DIR = 'processed_data'
INTERACTIONS_FILE = f'{PROCESSED_DIR}/user_interactions.csv'
MODEL_SAVE_PATH = f'{PROCESSED_DIR}/neumf_model.keras'

NUM_FACTORS = 32
MLP_LAYERS = [64, 32, 16, 8]
BATCH_SIZE = 256
EPOCHS = 15
LEARNING_RATE = 0.001
NEGATIVE_SAMPLES = 4  # per positive interaction

# Load & Prepare
interactions = pd.read_csv(INTERACTIONS_FILE)

user_ids = interactions['user_id'].unique()
item_ids = interactions['product_id'].unique()
user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
item_to_idx = {iid: i for i, iid in enumerate(item_ids)}

interactions['user_idx'] = interactions['user_id'].map(user_to_idx)
interactions['item_idx'] = interactions['product_id'].map(item_to_idx)

# Positive interactions
positives = interactions.copy()
positives['label'] = 1

# Simple negative sampling
def sample_negatives(df, neg_ratio=NEGATIVE_SAMPLES):
    negatives = []
    for _, row in df.iterrows():
        for _ in range(neg_ratio):
            neg_item = np.random.choice(item_ids)
            negatives.append({'user_idx': row['user_idx'], 'item_idx': item_to_idx[neg_item], 'label': 0})
    return pd.DataFrame(negatives)

neg_df = sample_negatives(positives)
data = pd.concat([positives[['user_idx', 'item_idx', 'label']], neg_df])

# Train-test split
train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

# Build Model (NeuMF)
def build_neumf(num_users, num_items, num_factors=32, mlp_layers=[64,32,16,8]):
    # GMF branch
    user_gmf = layers.Input(shape=(1,), dtype=tf.int32, name='user_gmf')
    item_gmf = layers.Input(shape=(1,), dtype=tf.int32, name='item_gmf')
    u_emb_gmf = layers.Embedding(num_users, num_factors)(user_gmf)
    i_emb_gmf = layers.Embedding(num_items, num_factors)(item_gmf)
    gmf = layers.Multiply()([layers.Flatten()(u_emb_gmf), layers.Flatten()(i_emb_gmf)])
    
    # MLP branch
    user_mlp = layers.Input(shape=(1,), dtype=tf.int32, name='user_mlp')
    item_mlp = layers.Input(shape=(1,), dtype=tf.int32, name='item_mlp')
    u_emb_mlp = layers.Embedding(num_users, num_factors*2)(user_mlp)
    i_emb_mlp = layers.Embedding(num_items, num_factors*2)(item_mlp)
    mlp_vec = layers.Concatenate()([layers.Flatten()(u_emb_mlp), layers.Flatten()(i_emb_mlp)])
    
    for units in mlp_layers:
        mlp_vec = layers.Dense(units, activation='relu')(mlp_vec)
        mlp_vec = layers.Dropout(0.2)(mlp_vec)
    
    # Fusion
    neumf_vec = layers.Concatenate()([gmf, mlp_vec])
    output = layers.Dense(1, activation='sigmoid')(neumf_vec)
    
    model = Model(inputs=[user_gmf, item_gmf, user_mlp, item_mlp], outputs=output)
    return model

model = build_neumf(len(user_ids), len(item_ids), NUM_FACTORS, MLP_LAYERS)
model.compile(optimizer=optimizers.Adam(LEARNING_RATE), 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.summary()

# Training
X_train = [train['user_idx'].values, train['item_idx'].values, 
           train['user_idx'].values, train['item_idx'].values]
y_train = train['label'].values

X_test = [test['user_idx'].values, test['item_idx'].values, 
          test['user_idx'].values, test['item_idx'].values]
y_test = test['label'].values

early_stop = callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(X_test, y_test), callbacks=[early_stop])

# Save & Inference (same as before)
model.save(MODEL_SAVE_PATH)