from algorithms.path_finding import PathFinder
from services.traffic_predictor import TrafficPredictor
from datetime import datetime
from typing import List, Tuple

class RouteOptimizer:
    def __init__(self, path_finder: PathFinder, traffic_predictor: TrafficPredictor):
        self.path_finder = path_finder
        self.traffic_predictor = traffic_predictor

    def optimize_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        departure_time: datetime,
        preferences: dict = None
    ) -> List[Tuple[float, float]]:
        # Get initial path using A*
        base_path = self.path_finder.a_star(start, end)
        
        # Apply traffic predictions
        optimized_path = self._apply_traffic_optimization(
            base_path,
            departure_time
        )
        
        # Apply user preferences if provided
        if preferences:
            optimized_path = self._apply_preferences(
                optimized_path,
                preferences
            )
            
        return optimized_path
    
    def _apply_traffic_optimization(
        self,
        path: List[Tuple[float, float]],
        departure_time: datetime
    ) -> List[Tuple[float, float]]:
        optimized_path = []
        current_time = departure_time
        
        for i in range(len(path) - 1):
            current_point = path[i]
            next_point = path[i + 1]
            
            # Get traffic prediction for current segment
            traffic_level = self.traffic_predictor.predict_traffic(
                current_point,
                current_time
            )
            
            # If heavy traffic, try alternative routes
            if traffic_level > 0.7:  # 70% congestion threshold
                alternative_path = self.path_finder.dijkstra(
                    current_point,
                    next_point
                )
                optimized_path.extend(alternative_path[:-1])
            else:
                optimized_path.append(current_point)
                
        optimized_path.append(path[-1])
        return optimized_path
    
    def _apply_preferences(
        self,
        path: List[Tuple[float, float]],
        preferences: dict
    ) -> List[Tuple[float, float]]:
        # Apply user preferences (e.g., avoid highways, prefer scenic routes)
        # This is a simplified version - real implementation would be more complex
        if preferences.get('avoid_highways'):
            return self._avoid_highways(path)
        return path
    
    def _avoid_highways(
        self,
        path: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        # Implementation to avoid highways
        # This would use road classification data to avoid highway segments
        return path