#News Recommendation Engine

A hybrid recommendation system that combines collaborative and content-based filtering with NLP techniques for personalized news recommendations.

## Features

- Hybrid recommendation system combining:
  - Content-based filtering using Word2Vec and LDA
  - Collaborative filtering using matrix factorization
- Real-time processing with PySpark integration
- REST API for easy integration
- Comprehensive test suite

## Requirements

- Python 3.9+
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the API server:
   ```bash
   uvicorn api.main:app --reload
   ```

2. Train the recommender:
   ```bash
   POST /train
   {
     "articles": [...],
     "interactions": [...]
   }
   ```

3. Get recommendations:
   ```bash
   POST /recommend/{user_id}
   {
     "user_profile": {...},
     "articles": [...]
   }
   ```

## Architecture

- FastAPI for REST API
- Word2Vec and LDA for content analysis
- Matrix factorization for collaborative filtering
- PySpark for scalable processing
- Pytest for testing

## Performance Features

- Efficient hybrid recommendation algorithm
- Optimized matrix operations
- Caching of model predictions
- Asynchronous API endpoints
