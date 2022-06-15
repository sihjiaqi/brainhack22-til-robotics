from queue import PriorityQueue
from re import X
from typing import List, Tuple, TypeVar, Dict
from tilsdk.localization import *
import heapq

T = TypeVar('T')

class NoPathFoundException(Exception):
    pass


class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []

    def is_empty(self) -> bool:
        return not self.elements

    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> T:
        return heapq.heappop(self.elements)


class Planner:
    def __init__(self, map_:SignedDistanceGrid=None, sdf_weight:float=0.0):
        '''
        Parameters
        ----------
        map : SignedDistanceGrid
            Distance grid map
        sdf_weight: float
            Relative weight of distance in cost function.
        '''
        self.map = map_
        self.coord_grid = [] 

        for row in range(self.map.height):
            coord_row = []
            for col in range(self.map.width):
                coord_row.append(GridLocation(col, row))
        
            self.coord_grid.append(coord_row)
        
        self.sdf_weight = sdf_weight

    def update_map(self, map:SignedDistanceGrid):
        '''Update planner with new map.'''
        self.map = map

    def heuristic(self, a:GridLocation, b:GridLocation) -> float:
        '''Planning heuristic function.
        
        Parameters
        ----------
        a: GridLocation
            Starting location.
        b: GridLocation
            Goal location.
        '''
        return euclidean_distance(a, b)

    def plan(self, start:RealLocation, goal:RealLocation) -> List[RealLocation]:
        '''Plan in real coordinates.
        
        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: RealLocation
            Starting location.
        goal: RealLocation
            Goal location.
        
        Returns
        -------
        path
            List of RealLocation from start to goal.
        '''

        path = self.plan_grid(self.map.real_to_grid(start), self.map.real_to_grid(goal))
        #print(path)
        return [self.map.grid_to_real(wp) for wp in path]

    def plan_grid(self, start:GridLocation, goal:GridLocation) -> List[GridLocation]:
        '''Plan in grid coordinates.
        
        Raises NoPathFileException path is not found.

        Parameters
        ----------
        start: GridLocation
            Starting location.
        goal: GridLocation
            Goal location.
        
        Returns
        -------
        path
            List of GridLocation from start to goal.
        '''
        count = 0
        # print(self.map)
        # print(self.map.grid)
        # print(self.map.width)
        # print(self.map.height)
        # print(min(self.map.grid[0]))
        if not self.map:
            raise RuntimeError('Planner map is not initialized.')
    
        frontier = PriorityQueue()
        frontier.put(0, (0, start))

        came_from: Dict[GridLocation, GridLocation] = {}
        #cost_so_far: Dict[GridLocation, float] = {}
        came_from[start] = None
        #print(goal)
        #cost_so_far[start] = 0

        g_score = {node: float('inf') for row in self.coord_grid for node in row}
        g_score[start] = 0
        f_score = {node: float('inf') for row in self.coord_grid for node in row}
        f_score[start] = self.heuristic(start, goal)
        open_set_hash = {start}
        #print('hash',open_set_hash)
        while not frontier.is_empty():
            
            # TODO: Participant to complete.
            current = frontier.get()[0][1]
            #print('now', current)
            #print(g_score[current])
            #print('Current:', current)
            #print('Goal:', goal)

            neightbors = self.map.neighbours(current)
            for neighbor in neightbors:
                temp_g_score = g_score[current] + 1
                
                if temp_g_score < g_score[neighbor[0]]:
                    came_from[neighbor[0]] = current
                    g_score[neighbor[0]] = temp_g_score
                    f_score[neighbor[0]] = temp_g_score + self.heuristic(neighbor[0], goal)
                    if neighbor not in open_set_hash:
                        count += 1
                        frontier.put(count, (f_score[neighbor[0]], neighbor[0]))
                        #print('Inserted', count, (f_score[neighbor[0]], neighbor[0]))
                        open_set_hash.add(neighbor[0])


        if goal not in came_from:
            raise NoPathFoundException
        
        print('done')
        
        return self.reconstruct_path(came_from, start, goal)

    def reconstruct_path(self,
                         came_from:Dict[GridLocation, GridLocation],
                         start:GridLocation, goal:GridLocation) -> List[GridLocation]:
        '''Traces traversed locations to reconstruct path.
        
        Parameters
        ----------
        came_from: dict
            Dictionary mapping location to location the planner came from.
        start: GridLocation
            Start location for path.
        goal: GridLocation
            Goal location for path.

        Returns
        -------
        path
            List of GridLocation from start to goal.
        '''
        
        current: GridLocation = goal
        path: List[GridLocation] = []
        
        while current != start:
            path.append(current)
            current = came_from[current]
            
        # path.append(start)
        path.reverse()
        return path

