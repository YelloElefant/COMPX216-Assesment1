from time import time
from search import *
import numpy as np
from assignment1aux import *


# from assignment1aux import *
def read_initial_state_from_file(filename):
    """
    Reads the initial state from a file and constructs the game map.
    Args:
        filename (str): Path to the configuration file.
    Returns:
        tuple: A tuple containing the game map, initial position, and direction.
    """
    try:
        with open(filename, 'r') as file:
            height = int(file.readline().strip())
            width = int(file.readline().strip())
            game_map = [['' for _ in range(width)] for _ in range(height)]
            rocks = set()
            for line in file:
                x, y = map(int, line.strip().split(','))
                rocks.add((x, y))
                game_map[x][y] = 'rock'
        return tuple(tuple(row) for row in game_map), None, None
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")
    except ValueError:
        raise ValueError("Invalid file format. Ensure the file contains height, width, and rock positions.")




class ZenPuzzleGarden(Problem):
    def __init__(self, initial):
        if type(initial) is str:
            super().__init__(read_initial_state_from_file(initial))
        else:
            super().__init__(initial)

    def actions(self, state):
        map = state[0]
        position = state[1]
        direction = state[2]
        height = len(map)
        width = len(map[0])
        action_list = []
        if position:
            if direction in ['up', 'down']:
                if position[1] == 0 or not map[position[0]][position[1] - 1]:
                    action_list.append((position, 'left'))
                if position[1] == width - 1 or not map[position[0]][position[1] + 1]:
                    action_list.append((position, 'right'))
            if direction in ['left', 'right']:
                if position[0] == 0 or not map[position[0] - 1][position[1]]:
                    action_list.append((position, 'up'))
                if position[0] == height - 1 or not map[position[0] + 1][position[1]]:
                    action_list.append((position, 'down'))
        else:
            for i in range(height):
                if not map[i][0]:
                    action_list.append(((i, 0), 'right'))
                if not map[i][width - 1]:
                    action_list.append(((i, width - 1), 'left'))
            for i in range(width):
                if not map[0][i]:
                    action_list.append(((0, i), 'down'))
                if not map[height - 1][i]:
                    action_list.append(((height - 1, i), 'up'))
        return action_list

    def result(self, state, action):
        map = [list(row) for row in state[0]]
        position = action[0]
        direction = action[1]
        height = len(map)
        width = len(map[0])
        while True:
            row_i = position[0]
            column_i = position[1]
            if direction == 'left':
                new_position = (row_i, column_i - 1)
            if direction == 'up':
                new_position = (row_i - 1, column_i)
            if direction == 'right':
                new_position = (row_i, column_i + 1)
            if direction == 'down':
                new_position = (row_i + 1, column_i)
            if new_position[0] < 0 or new_position[0] >= height or new_position[1] < 0 or new_position[1] >= width:
                map[row_i][column_i] = direction
                return tuple(tuple(row) for row in map), None, None
            if map[new_position[0]][new_position[1]]:
                return tuple(tuple(row) for row in map), position, direction
            map[row_i][column_i] = direction
            position = new_position

    def goal_test(self, state):
        # Task 2
        # Return a boolean value indicating if a given state is solved.
        # Replace the line below with your code.
        map = state[0]
        height = len(map)
        width = len(map[0])
        for i in range(height):
            for j in range(width):
                if not map[i][j]:
                    return False
        return True
        raise NotImplementedError

# Task 3
# Implement an A* heuristic cost function and assign it to the variable below.
from collections import deque

def astar_heuristic_cost(node):
    """
    Heuristic function for A* search.
    Estimates the total cost to move all rocks to their nearest goal positions.
    Moves are restricted to sliding in one direction until a rock or edge is encountered.
    """
    map = node.state[0]
    height = len(map)
    width = len(map[0])
    rock_positions = []
    goal_positions = []

    # Find all rock and goal positions
    for i in range(height):
        for j in range(width):
            if map[i][j] == 'rock':
                rock_positions.append((i, j))
            elif map[i][j] == 'goal':
                goal_positions.append((i, j))

    # If there are no rocks or no goals, the heuristic cost is 0
    if not rock_positions or not goal_positions:
        return 0

    # Function to calculate the minimum sliding moves from a given position to a goal
    def sliding_distance(start, goal):
        """
        Calculate the sliding distance from the start position to the goal position.
        Movement is done in a straight line until a rock or edge is encountered.
        """
        # Check horizontal (left-right) movement
        x, y = start
        # Try moving right
        if y < goal[1]:
            while y < width and map[x][y] != 'rock' and (x, y) != goal:
                y += 1
            if (x, y) == goal:
                return abs(y - start[1])
        # Try moving left
        elif y > goal[1]:
            while y >= 0 and map[x][y] != 'rock' and (x, y) != goal:
                y -= 1
            if (x, y) == goal:
                return abs(y - start[1])

        # Check vertical (up-down) movement
        x, y = start
        # Try moving down
        if x < goal[0]:
            while x < height and map[x][y] != 'rock' and (x, y) != goal:
                x += 1
            if (x, y) == goal:
                return abs(x - start[0])
        # Try moving up
        elif x > goal[0]:
            while x >= 0 and map[x][y] != 'rock' and (x, y) != goal:
                x -= 1
            if (x, y) == goal:
                return abs(x - start[0])

        return float('inf')  # In case the goal is unreachable

    # Calculate the total cost as the sum of the minimum sliding distances from each rock
    total_cost = 0
    for rock in rock_positions:
        min_distance = float('inf')
        for goal in goal_positions:
            # Find the sliding distance from the rock to the goal
            distance = sliding_distance(rock, goal)
            min_distance = min(min_distance, distance)
        total_cost += min_distance

    return total_cost



def beam_search(problem, f, beam_width):
    if beam_width <= 0:
        raise ValueError("Beam width must be a positive integer.")
    
    node = Node(problem.initial)
    frontier = [node]
    visited = set()
    
    while frontier:
        new_frontier = []
        for node in frontier:
            if problem.goal_test(node.state):
                return node
            visited.add(node.state)
            for action in problem.actions(node.state):
                child = node.child_node(problem, action)
                if child.state not in visited:
                    new_frontier.append(child)
        
        # Use a heap to efficiently keep the top beam_width nodes
        if new_frontier:
            heapq.heapify(new_frontier)
            frontier = heapq.nsmallest(beam_width, new_frontier, key=f)
        else:
            frontier = []
    
    return None  # No solution found





if __name__ == "__main__":

    # Task 1 test code
    print('The loaded initial state is visualised below.')
    visualise(read_initial_state_from_file('assignment1config.txt'))

    # Task 2 test code
    garden = ZenPuzzleGarden('assignment1config.txt')
    print('Running breadth-first graph search.')
    before_time = time()
    node = breadth_first_graph_search(garden)
    after_time = time()
    print(f'Breadth-first graph search took {after_time - before_time} seconds.')
    if node:
        print(f'Its solution with a cost of {node.path_cost} is animated below.')
        animate(node)
    else:
        print('No solution was found.')

    # Task 3 test code
    print('Running A* search.')
    before_time = time()
    node = astar_search(garden, astar_heuristic_cost)
    after_time = time()
    print(f'A* search took {after_time - before_time} seconds.')
    if node:
        print(f'Its solution with a cost of {node.path_cost} is animated below.')
        animate(node)
    else:
        print('No solution was found.')

    # Task 4 test code
    print('Running beam search.')
    before_time = time()
    node = beam_search(garden, lambda n: n.path_cost + astar_heuristic_cost(n), 50)
    after_time = time()
    print(f'Beam search took {after_time - before_time} seconds.')
    if node:
        print(f'Its solution with a cost of {node.path_cost} is animated below.')
        animate(node)
    else:
        print('No solution was found.')
