from typing import Tuple, List
from haversine import haversine
import numpy as np

def calculate_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """Calculate distance between two points in kilometers."""
    return haversine(point1, point2)

def get_bounding_box(
    points: List[Tuple[float, float]],
    padding: float = 0.01
) -> Tuple[float, float, float, float]:
    """Get the bounding box for a set of points with padding."""
    lats, lons = zip(*points)
    min_lat, max_lat = min(lats) - padding, max(lats) + padding
    min_lon, max_lon = min(lons) - padding, max(lons) + padding
    return min_lat, min_lon, max_lat, max_lon

def interpolate_points(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_points: int = 10
) -> List[Tuple[float, float]]:
    """Interpolate points between start and end coordinates."""
    lats = np.linspace(start[0], end[0], num_points)
    lons = np.linspace(start[1], end[1], num_points)
    return list(zip(lats, lons))