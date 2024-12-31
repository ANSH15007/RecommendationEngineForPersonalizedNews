from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Coordinates(BaseModel):
    latitude: float
    longitude: float

class RouteRequest(BaseModel):
    start: Coordinates
    end: Coordinates
    departure_time: datetime
    user_id: str
    preferences: Optional[dict] = None

class RouteResponse(BaseModel):
    route_id: str
    path: List[Coordinates]
    distance: float
    estimated_time: float
    traffic_level: str
    updated_at: datetime