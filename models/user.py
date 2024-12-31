from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict

class UserProfile(BaseModel):
    user_id: str
    preferences: Dict[str, float]
    read_articles: List[str]
    created_at: datetime
    last_active: datetime