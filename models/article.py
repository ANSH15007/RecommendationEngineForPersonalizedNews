from pydantic import BaseModel
from datetime import datetime
from typing import List

class Article(BaseModel):
    article_id: str
    title: str
    content: str
    category: str
    tags: List[str]
    published_date: datetime
    author: str