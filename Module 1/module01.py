###
# Edwin Flores
###

import queue
import operator
import math
import copy

INFINITE = 999999

full_world = [
  ['.', '.', '.', '.', '.', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], 
  ['.', '.', '.', '.', '.', '.', '.', '*', '*', '*', '*', '*', '*', '*', '*', '*', '.', '.', 'x', 'x', 'x', 'x', 'x', 'x', 'x', '.', '.'], 
  ['.', '.', '.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', '#', '#', '#', 'x', 'x', '#', '#'], 
  ['.', '.', '.', '.', '#', 'x', 'x', 'x', '*', '*', '*', '*', '~', '~', '*', '*', '*', '*', '*', '.', '.', '#', '#', 'x', 'x', '#', '.'], 
  ['.', '.', '.', '#', '#', 'x', 'x', '*', '*', '.', '.', '~', '~', '~', '~', '*', '*', '*', '.', '.', '.', '#', 'x', 'x', 'x', '#', '.'], 
  ['.', '#', '#', '#', 'x', 'x', '#', '#', '.', '.', '.', '.', '~', '~', '~', '~', '~', '.', '.', '.', '.', '.', '#', 'x', '#', '.', '.'], 
  ['.', '#', '#', 'x', 'x', '#', '#', '.', '.', '.', '.', '#', 'x', 'x', 'x', '~', '~', '~', '.', '.', '.', '.', '.', '#', '.', '.', '.'], 
  ['.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.', '.', '.', '#', 'x', 'x', 'x', '~', '~', '~', '.', '.', '#', '#', '#', '.', '.'], 
  ['.', '.', '.', '#', '#', '#', '.', '.', '.', '.', '.', '.', '#', '#', 'x', 'x', '.', '~', '~', '.', '.', '#', '#', '#', '.', '.', '.'], 
  ['.', '.', '.', '~', '~', '~', '.', '.', '#', '#', '#', 'x', 'x', 'x', 'x', '.', '.', '.', '~', '.', '#', '#', '#', '.', '.', '.', '.'], 
  ['.', '.', '~', '~', '~', '~', '~', '.', '#', '#', 'x', 'x', 'x', '#', '.', '.', '.', '.', '.', '#', 'x', 'x', 'x', '#', '.', '.', '.'], 
  ['.', '~', '~', '~', '~', '~', '.', '.', '#', 'x', 'x', '#', '.', '.', '.', '.', '~', '~', '.', '.', '#', 'x', 'x', '#', '.', '.', '.'], 
  ['~', '~', '~', '~', '~', '.', '.', '#', '#', 'x', 'x', '#', '.', '~', '~', '~', '~', '.', '.', '.', '#', 'x', '#', '.', '.', '.', '.'], 
  ['.', '~', '~', '~', '~', '.', '.', '#', '*', '*', '#', '.', '.', '.', '.', '~', '~', '~', '~', '.', '.', '#', '.', '.', '.', '.', '.'], 
  ['.', '.', '.', '.', 'x', '.', '.', '*', '*', '*', '*', '#', '#', '#', '#', '.', '~', '~', '~', '.', '.', '#', 'x', '#', '.', '.', '.'], 
  ['.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', '#', '#', '.', '~', '.', '#', 'x', 'x', '#', '.', '.', '.'], 
  ['.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', '.', '.', 'x', 'x', 'x', '.', '.', '.', '.', '.'], 
  ['.', '.', '.', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x', 'x', 'x', 'x', '.', '.', '.', '.', '.', '.', '.'], 
  ['.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', '*', '*', '.', '.', '.', '#', '#', '.', '.', '.', '.', '.', '.', '.', '.'], 
  ['.', '.', '.', '.', 'x', 'x', 'x', '*', '*', '*', '*', '*', '*', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '~', '~', '~', '~'], 
  ['.', '.', '#', '#', '#', '#', 'x', 'x', '*', '*', '*', '*', '*', '.', 'x', '.', '.', '.', '.', '.', '~', '~', '~', '~', '~', '~', '~'], 
  ['.', '.', '.', '.', '#', '#', '#', 'x', 'x', 'x', '*', '*', 'x', 'x', '.', '.', '.', '.', '.', '.', '~', '~', '~', '~', '~', '~', '~'], 
  ['.', '.', '.', '.', '.', '.', '#', '#', '#', 'x', 'x', 'x', 'x', '.', '.', '.', '.', '#', '#', '.', '.', '~', '~', '~', '~', '~', '~'], 
  ['.', '#', '#', '.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.', '.', '#', '#', 'x', 'x', '#', '#', '.', '~', '~', '~', '~', '~'], 
  ['#', 'x', '#', '#', '#', '#', '.', '.', '.', '.', '.', 'x', 'x', 'x', '#', '#', 'x', 'x', '.', 'x', 'x', '#', '#', '~', '~', '~', '~'], 
  ['#', 'x', 'x', 'x', '#', '.', '.', '.', '.', '.', '#', '#', 'x', 'x', 'x', 'x', '#', '#', '#', '#', 'x', 'x', 'x', '~', '~', '~', '~'], 
  ['#', '#', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '#', '#', '#', '#', '#', '.', '.', '.', '.', '#', '#', '#', '.', '.', '.']]

test_world = [
  ['.', '*', '*', '*', '*', '*', '*'],
  ['.', '*', '*', '*', '*', '*', '*'],
  ['.', '*', '*', '*', '*', '*', '*'],
  ['.', '.', '.', '.', '.', '.', '.'],
  ['*', '*', '*', '*', '*', '*', '.'],
  ['*', '*', '*', '*', '*', '*', '.'],
  ['*', '*', '*', '*', '*', '*', '.'],
]

cardinal_moves = [(0,-1), (1,0), (0,1), (-1,0)]

costs = { '.': 1, '*': 3, '#': 5, '~': 7}

# heuristic function
def heuristic(position, world, costs, goal):
    x, y = position
    g = 0
    try:
      # Raise an exception for negative indices
      if x < 0 or y < 0:
        raise Exception
      # Return the cost of the move
      g = costs[world[y][x]]
    except:
      # Return a very larg enumber
      g = INFINITE
    # Calculate the euclidean distance between the
    # position and the goal
    h = math.sqrt(sum([(a - b) ** 2 for a, b in zip(position,goal)]))
    # Return the sum of h and g
    return g, h

# Determines whether the state is in the frontier
def in_frontier(frontier, state):
    for item in frontier.queue:
      if item[1] == state:
        return True
    return False

# Build the path found by the search
def build_path(start, goal, path):
    constructed = []
    current = goal
    constructed.append(path[current][1])
    # Iterate through the path dictionary starting from the
    # goal until the key for start is not found
    while current != start:
      try:
        current = path[current][0]
        constructed.append(path[current][1])
      except:
        # Do nothing if there is a key error
        pass
    # Return the reversed list
    return constructed[::-1]

# Gets a state from the frontier
def get_state(frontier, state):
    data = []
    for x in range(len(frontier.queue)):
      if frontier.queue[x][1] == state[1]:
        data = copy.deepcopy(frontier.queue[x])
        del frontier.queue[x]
        return data

# Determines whether the state has been explored
def in_explored(explored, state):
    for item in explored:
      if item == state:
        return True
    return False

# Obtain the successor nodes of the given stat
def get_successors(state, moves, world, costs, goal):
    # Obtain possible next positions
    next_positions = [tuple(map(operator.add, state, x)) for x in moves]
    # This successors list will contain tuples of
    # ([g,h], position, action)
    successors = []
    # Iterate over the positions. If there is an
    # exception do nothing. This means the 
    # position is invalid
    for x in range(len(next_positions)):
      try:
        # Calculate the hueristic and if it's less
        # than the infinite value then add it to the list
        g, h = heuristic(next_positions[x], world, costs, goal)
        successors.append(([g, h], next_positions[x], moves[x]))
      except IndexError:
        # Do nothing if there is an index error
        # the move is out of bounds
        pass
    return successors

# Performs a star seach
def a_star_search( world, start, goal, costs, moves, heuristic):
    # The data stored in the frontier will be:
    # ( Priority, ( State, Action, Cost/g(n) ))
    # Initialize the frontier and explored list
    frontier = queue.PriorityQueue()
    explored = []
    path = {}
    # Put the starting state on the frontier
    frontier.put((0, start, None, 0))
    # Iterate over the frontier
    while not frontier.empty():
      # Get the state with the highest priority
      current_state = frontier.get() 
      # Return the path list is the current
      # state is the goal state
      if current_state[1] == goal:
        return build_path(start, goal, path)
      # Obtain the successor states of the current state
      successors = get_successors(current_state[1], cardinal_moves, world, costs, goal)
      # Iterate over the successors
      for successor in successors:
        # If the successor is in the explored list
        # dont do anything
        if in_explored(explored, successor[1]):
          continue
        # Calculate f(n) and if greater than INFINITE
        # just skip this successor since it won't likley
        # be used
        f = current_state[3] + sum(successor[0])
        if f >= INFINITE:
          continue
        # Update the path dictionary
        path[successor[1]] = (current_state[1], successor[2])
        # If the successor is not in the frontier then add it
        if not in_frontier(frontier, successor[1]):
          frontier.put((f, successor[1], successor[2], successor[0][0]))
        # Otherwise, compare the f values
        else:
          # Get the states previously calculated data
          other = get_state(frontier, successor)
          # if the newly calculate f value is >= the prvious then put the other back
          # on the frontier
          if f >= other[0]:
            frontier.put(other)
          # Otherwise, put the new data on the frontier
          else:
            frontier.put((f, successor[1], successor[2], successor[0][0]))
      # Add the current state to the explored list
      explored.append(current_state[1])
    # If we happen not to reach the goal still return the path
    return path

def pretty_print_solution( world, path, start):
    direction = { (0, 1): 'v', (1, 0): '>', (0, -1): '^', (-1, 0): '<'}
    position = start
    # Iterate over the path and update the world
    # with the move and get the new position
    for move in path:
        x,y = position
        world[y][x] = direction[move]
        position = tuple(map(operator.add, position, move))
    # Update the goal position    
    x,y = position
    world[y][x] = 'G'
    # Print out the world
    for row in world:
      print(*row)

if __name__ == "__main__":
    
    print("A* solution for test world")
    test_path = a_star_search(test_world, (0, 0), (6, 6), costs, cardinal_moves, heuristic)
    print(test_path)
    pretty_print_solution( test_world, test_path, (0, 0))

    print("A* solution for full world")
    full_path = a_star_search(full_world, (0, 0), (26, 26), costs, cardinal_moves, heuristic)
    print(full_path)
    pretty_print_solution( full_world, full_path, (0, 0))