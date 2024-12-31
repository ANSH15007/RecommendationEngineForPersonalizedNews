from typing import List, Dict
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender

class HybridRecommender:
    def __init__(self, content_weight=0.5):
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeRecommender()
        self.content_weight = content_weight

    def train(self, articles: List[dict], interactions: List[Dict]):
        # Train content-based recommender
        self.content_recommender.train_word2vec(articles)
        self.content_recommender.train_lda(articles)
        
        # Train collaborative recommender
        self.collaborative_recommender.train(interactions)

    def recommend(self, user_profile: dict, articles: List[dict], n=5) -> List[str]:
        # Get recommendations from both systems
        content_recs = self.content_recommender.recommend(user_profile, articles, n)
        collab_recs = self.collaborative_recommender.recommend(user_profile['user_id'], n)
        
        # Combine recommendations with weights
        combined_recs = {}
        
        for i, rec in enumerate(content_recs):
            combined_recs[rec] = self.content_weight * (n - i)
            
        for i, rec in enumerate(collab_recs):
            if rec in combined_recs:
                combined_recs[rec] += (1 - self.content_weight) * (n - i)
            else:
                combined_recs[rec] = (1 - self.content_weight) * (n - i)
        
        # Sort and return top N recommendations
        sorted_recs = sorted(combined_recs.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
        return [rec[0] for rec in sorted_recs[:n]]