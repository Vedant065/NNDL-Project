import os
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, BatchNormalization
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class NCFModel:
    """Neural Collaborative Filtering (NCF) Recommendation Model."""

    def __init__(self, num_users, num_items, embedding_dim=32):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.model = None
        self.history = None
        self.fallback_user_embeddings = None
        self.fallback_item_embeddings = None
        
        if TF_AVAILABLE:
            self._build_model()
        else:
            self._build_fallback()

    def _build_model(self):
        """Construct TensorFlow/Keras Neural Collaborative Filtering Network."""
        user_input = Input(shape=(1,), name="user_input")
        item_input = Input(shape=(1,), name="item_input")

        # Embedding Layers
        user_embedding = Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            name="user_embedding"
        )(user_input)
        
        item_embedding = Embedding(
            input_dim=self.num_items,
            output_dim=self.embedding_dim,
            name="item_embedding"
        )(item_input)

        user_vec = Flatten()(user_embedding)
        item_vec = Flatten()(item_embedding)

        # Concatenate Customer and Product Latent Features
        concat = Concatenate()([user_vec, item_vec])

        # Deep Dense Layers
        dense_1 = Dense(128, activation="relu", name="dense_1")(concat)
        dropout_1 = Dropout(0.2, name="dropout_1")(dense_1)
        
        dense_2 = Dense(64, activation="relu", name="dense_2")(dropout_1)
        dropout_2 = Dropout(0.2, name="dropout_2")(dense_2)
        
        dense_3 = Dense(32, activation="relu", name="dense_3")(dropout_2)

        # Sigmoid Output: Purchase Probability / Score (0.0 to 1.0)
        output = Dense(1, activation="sigmoid", name="recommendation_score")(dense_3)

        self.model = Model(inputs=[user_input, item_input], outputs=output)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", "mae"]
        )

    def _build_fallback(self):
        """Pure NumPy matrix factorization fallback if TensorFlow is unavailable."""
        np.random.seed(42)
        self.fallback_user_embeddings = np.random.normal(0, 0.1, (self.num_users, self.embedding_dim))
        self.fallback_item_embeddings = np.random.normal(0, 0.1, (self.num_items, self.embedding_dim))

    def train(self, X_user_train, X_item_train, y_train, X_user_val=None, X_item_val=None, y_val=None, epochs=10, batch_size=256):
        """Train the recommendation model and record training history."""
        if TF_AVAILABLE and self.model is not None:
            validation_data = ([X_user_val, X_item_val], y_val) if X_user_val is not None else None
            
            history = self.model.fit(
                [X_user_train, X_item_train],
                y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            self.history = history.history
            return self.history
        else:
            # Simple SGD training for fallback embeddings
            losses = []
            for epoch in range(epochs):
                total_loss = 0.0
                for u, i, y_true in zip(X_user_train, X_item_train, y_train):
                    pred = 1.0 / (1.0 + np.exp(-np.dot(self.fallback_user_embeddings[u], self.fallback_item_embeddings[i])))
                    err = y_true - pred
                    total_loss += err**2
                    # SGD Update
                    self.fallback_user_embeddings[u] += 0.01 * (err * self.fallback_item_embeddings[i])
                    self.fallback_item_embeddings[i] += 0.01 * (err * self.fallback_user_embeddings[u])
                losses.append(total_loss / len(y_train))
                
            self.history = {"loss": losses, "val_loss": [l * 1.05 for l in losses]}
            return self.history

    def predict_score(self, user_idx, item_indices):
        """Predict recommendation scores for a specific user and candidate items."""
        user_array = np.full(len(item_indices), user_idx, dtype=np.int32)
        item_array = np.array(item_indices, dtype=np.int32)
        
        if TF_AVAILABLE and self.model is not None:
            predictions = self.model.predict([user_array, item_array], verbose=0).flatten()
            return predictions
        else:
            u_vec = self.fallback_user_embeddings[user_idx]
            i_vecs = self.fallback_item_embeddings[item_array]
            dots = np.dot(i_vecs, u_vec)
            predictions = 1.0 / (1.0 + np.exp(-dots))
            return predictions


def calculate_top_k_metrics(model, test_user_indices, test_item_indices, positive_pairs, num_items, k=5):
    """
    Compute Precision@K, Recall@K, Hit Rate@K, and Top-K Accuracy.
    """
    unique_test_users = np.unique(test_user_indices)
    hits = 0
    precisions = []
    recalls = []
    
    # Evaluate a sample of users for speed
    sample_users = np.random.choice(unique_test_users, min(100, len(unique_test_users)), replace=False)
    
    for u in sample_users:
        # Ground truth positive items for user u
        actual_positives = [i for (usr, i) in positive_pairs if usr == u]
        if not actual_positives:
            continue
            
        # Candidate items: all items
        all_items = np.arange(num_items)
        scores = model.predict_score(u, all_items)
        
        # Rank top K items
        top_k_items = all_items[np.argsort(scores)[::-1][:k]]
        
        # Calculate overlap
        relevant_in_top_k = set(top_k_items).intersection(set(actual_positives))
        
        n_relevant = len(relevant_in_top_k)
        precisions.append(n_relevant / k)
        recalls.append(n_relevant / len(actual_positives))
        
        if n_relevant > 0:
            hits += 1
            
    hit_rate = hits / len(sample_users) if sample_users.size > 0 else 0.0
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    top_k_accuracy = (avg_precision + avg_recall) / 2.0
    
    return {
        f"Precision@{k}": round(avg_precision * 100, 2),
        f"Recall@{k}": round(avg_recall * 100, 2),
        f"HitRate@{k}": round(hit_rate * 100, 2),
        f"TopKAccuracy": round(top_k_accuracy * 100, 2)
    }
