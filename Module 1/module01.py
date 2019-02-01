###
# Edwin Flores
###

import queue
import operator

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

frontier = []
explored = []

# heuristic function
def heuristic(start, goal):
    # calculate the manhattan distance
    x = tuple(map(operator.sub, start, goal))
    return sum([abs(y) for y in x ])

def get_costs(position, world, costs):
    x, y = position
    try:
      # Raise an exception for negative indices
      if x < 0 or y < 0:
        raise Exception
      # Return the cost of the move
      return costs[world[y][x]]
    except:
      # Return a very larg enumber
      return INFINITE

def in_frontier(frontier, state):
    for item in frontier.queue:
      if item == state:
        return True
    return False

def in_explored(explored, state):
    for item in explored:
      if item == state:
        return True
    return False
# obtain the successor nodes of the state
def get_successors(state, moves, world, costs, goal):
    next_positions = [tuple(map(operator.add, state, x)) for x in moves]
    # This list will contain tuples of
    # (cost, position, action)
    successors = []
    for x in range(len(next_positions)):
      try:
        g = get_costs(next_positions[x], world, costs)
        if g < INFINITE:
          f = g + heuristic(next_positions[x], goal)
          successors.append((f, next_positions[x], moves[x]))
      except IndexError:
        # Do nothing if there is an index error
        # the move is out of bounds
        pass
    return successors

def a_star_search( world, start, goal, costs, moves, heuristic):
    # Initialize the frontier and explored list
    frontier = queue.PriorityQueue()
    explored = []
    path = []
    # Put the starting state on the frontier
    frontier.put((0, start, start))

    while not frontier.empty():
      
      current_state = frontier.get()
      
      print(current_state, "~~~~~~~~~~", in_frontier(frontier, current_state[1]), in_explored(explored, current_state[1]))
      if current_state[1] == goal:
        print("MATCH")
        return path
      
      path.append(current_state[2])
      successors = get_successors(current_state[1], cardinal_moves, world, costs, goal)

      for successor in successors:
        if not in_frontier(frontier, successor[1]) and not in_explored(explored, successor[1]):
          frontier.put(successor)
    
      explored.append(current_state[1])
    return path

def pretty_print_solution( world, path, start):
    ### YOUR SOLUTION HERE ###
    ### YOUR SOLUTION HERE ###
    return None

if __name__ == "__main__":
    
    #print("A* solution for test world")
    #test_path = a_star_search(test_world, (0, 0), (6, 6), costs, cardinal_moves, heuristic)
    #print(test_path)
    #pretty_print_solution( test_world, test_path, (0, 0))

    print("A* solution for full world")
    #full_path = a_star_search(full_world, (0, 0), (26, 26), costs, cardinal_moves, heuristic)
    full_path = a_star_search(full_world, (0, 0), (7, 7), costs, cardinal_moves, heuristic)
    print(full_path)
    #pretty_print_solution(full_world, full_path, (0, 0))