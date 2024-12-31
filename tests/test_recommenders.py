import pytest
from recommenders.content_based import ContentBasedRecommender
from recommenders.collaborative import CollaborativeRecommender
from recommenders.hybrid import HybridRecommender
from datetime import datetime

@pytest.fixture
def sample_articles():
    return [
        {
            "article_id": "1",
            "title": "Python Programming",
            "content": "Python is a popular programming language",
            "category": "Technology",
            "tags": ["programming", "python"],
            "published_date": datetime.now(),
            "author": "John Doe"
        },
        {
            "article_id": "2",
            "title": "Machine Learning",
            "content": "Introduction to machine learning concepts",
            "category": "Technology",
            "tags": ["ML", "AI"],
            "published_date": datetime.now(),
            "author": "Jane Smith"
        }
    ]

@pytest.fixture
def sample_interactions():
    return [
        {"user_id": "user1", "article_id": "1", "rating": 5},
        {"user_id": "user1", "article_id": "2", "rating": 4},
        {"user_id": "user2", "article_id": "1", "rating": 3}
    ]

def test_content_based_recommender(sample_articles):
    recommender = ContentBasedRecommender()
    recommender.train_word2vec(sample_articles)
    recommender.train_lda(sample_articles)
    
    user_profile = {
        "user_id": "user1",
        "preferences": {"Technology": 0.8, "Programming": 0.6}
    }
    
    recommendations = recommender.recommend(user_profile, sample_articles, n=1)
    assert len(recommendations) == 1

def test_collaborative_recommender(sample_interactions):
    recommender = CollaborativeRecommender()
    recommender.train(sample_interactions)
    
    recommendations = recommender.recommend("user1", n=1)
    assert len(recommendations) == 1

def test_hybrid_recommender(sample_articles, sample_interactions):
    recommender = HybridRecommender()
    recommender.train(sample_articles, sample_interactions)
    
    user_profile = {
        "user_id": "user1",
        "preferences": {"Technology": 0.8, "Programming": 0.6}
    }
    
    recommendations = recommender.recommend(user_profile, sample_articles, n=1)
    assert len(recommendations) == 1