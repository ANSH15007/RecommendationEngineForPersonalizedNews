from fastapi import FastAPI, HTTPException
from models.article import Article
from models.user import UserProfile
from recommenders.hybrid import HybridRecommender
from typing import List
import uvicorn

app = FastAPI()
recommender = HybridRecommender()

@app.post("/train")
async def train_recommender(articles: List[Article], interactions: List[dict]):
    try:
        recommender.train([article.dict() for article in articles], interactions)
        return {"message": "Training completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/{user_id}")
async def get_recommendations(
    user_id: str,
    user_profile: UserProfile,
    articles: List[Article],
    n: int = 5
):
    try:
        recommendations = recommender.recommend(
            user_profile.dict(),
            [article.dict() for article in articles],
            n
        )
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)