###
# Edwin Flores
###

import queue
import operator
import math

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
      g =  costs[world[y][x]]
    except:
      # Return a very larg enumber
      g = INFINITE
    # Calculate the euclidean distance between the
    # position and the goal
    h = math.sqrt(sum([(a - b) ** 2 for a, b in zip(position,goal)]))
    # Return the sum of h and g
    return (h + g)

# Determines whether the state is in the frontier
def in_frontier(frontier, state):
    for item in frontier.queue:
      if item[1] == state:
        return True
    return False

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
    # (cost, position, action)
    successors = []
    # Iterate over the positions. If there is an
    # exception do nothing. This means the 
    # position is invalid
    for x in range(len(next_positions)):
      try:
        # Calculate the hueristic and if it's less
        # than the infinite value then add it to the list
        f = heuristic(next_positions[x], world, costs, goal)
        if f < INFINITE:
          successors.append((f, next_positions[x], moves[x]))
      except IndexError:
        # Do nothing if there is an index error
        # the move is out of bounds
        pass
    return successors

# Performs a star seach
def a_star_search( world, start, goal, costs, moves, heuristic):
    # Initialize the frontier and explored list
    frontier = queue.PriorityQueue()
    explored = []
    path = []
    # Put the starting state on the frontier
    frontier.put((0, start, start))
    # Iterate over the frontier
    while not frontier.empty():
      # Get the state with the highest priority
      current_state = frontier.get()
      # Return the path list is the current
      # state is the goal state
      if current_state[1] == goal:
        return path
      # Add the action of the current state to
      # the path list
      path.append(current_state[2])
      # Obtain the successor states of the current state
      successors = get_successors(current_state[1], cardinal_moves, world, costs, goal)
      # Iterate over the successors
      for successor in successors:
        # If the successor is not in the frontier or has been explored
        # add it to the frontier
        if not in_frontier(frontier, successor[1]) and not in_explored(explored, successor[1]):
          frontier.put(successor)
    
      print("CURRENT", current_state[1])
      # Add the current state to the explored list
      explored.append(current_state[1])
      #print("SUCCESSORS: ", len(successors))
     # print("FRONTIER")
     # for item in frontier.queue:
      #  print(item)
     # print("EXPLORED")
      #print(*explored)
    # If we happen not to reach the goal
    # return the path list
    return path

def pretty_print_solution( world, path, start):
    direction = { (0, 1): 'v', (1, 0): '>', (0, -1): '^', (-1, 0): '<'}
    position = start
    x,y = position
    world[y][x] = 'S'
    print(position)
    for move in path[1:]:
      position = tuple(map(operator.add, position, move))
      x,y = position
      print(position)
      world[y][x] = direction[move]
    x,y = tuple(map(operator.add, position, path[-1]))
    world[y][x] = 'G'
    
    for item in world:
      print(*item)
    return None

if __name__ == "__main__":
    
    #print("A* solution for test world")
    #test_path = a_star_search(test_world, (0, 0), (6, 6), costs, cardinal_moves, heuristic)
    #print(test_path)
    #pretty_print_solution( test_world, test_path, (0, 0))

    print("A* solution for full world")
    full_path = a_star_search(full_world, (0, 0), (26, 26), costs, cardinal_moves, heuristic)
    #full_path = a_star_search(full_world, (0, 0), (7, 7), costs, cardinal_moves, heuristic)
    print(full_path)
    #pretty_print_solution(full_world, full_path, (0, 0))