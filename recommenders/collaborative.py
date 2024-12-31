from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
from typing import List, Dict

class CollaborativeRecommender:
    def __init__(self, num_factors=50):
        self.num_factors = num_factors
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}

    def create_interaction_matrix(self, interactions: List[Dict]):
        # Create user and item mappings
        unique_users = list(set(inter['user_id'] for inter in interactions))
        unique_items = list(set(inter['article_id'] for inter in interactions))
        
        self.user_map = {user: idx for idx, user in enumerate(unique_users)}
        self.item_map = {item: idx for idx, item in enumerate(unique_items)}
        
        # Create interaction matrix
        matrix = np.zeros((len(unique_users), len(unique_items)))
        for inter in interactions:
            user_idx = self.user_map[inter['user_id']]
            item_idx = self.item_map[inter['article_id']]
            matrix[user_idx, item_idx] = inter['rating']
        
        return matrix

    def train(self, interactions: List[Dict]):
        matrix = self.create_interaction_matrix(interactions)
        
        # Perform SVD
        U, sigma, Vt = svds(matrix, k=self.num_factors)
        
        # Convert to latent factors
        self.user_factors = U
        self.item_factors = Vt.T
        
    def recommend(self, user_id: str, n=5) -> List[str]:
        if user_id not in self.user_map:
            return []
            
        user_idx = self.user_map[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Calculate predicted ratings
        predictions = np.dot(user_vector, self.item_factors.T)
        
        # Get top N recommendations
        top_indices = np.argsort(predictions)[-n:][::-1]
        
        # Convert back to article IDs
        reverse_item_map = {v: k for k, v in self.item_map.items()}
        return [reverse_item_map[idx] for idx in top_indices]