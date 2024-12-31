import networkx as nx
from typing import List, Tuple
import numpy as np
from haversine import haversine

class PathFinder:
    def __init__(self, graph):
        self.graph = graph

    def a_star(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        def heuristic(node1, node2):
            return haversine(node1, node2)

        path = nx.astar_path(
            self.graph,
            start,
            end,
            heuristic=heuristic,
            weight='weight'
        )
        return path

    def dijkstra(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        return nx.dijkstra_path(
            self.graph,
            start,
            end,
            weight='weight'
        )