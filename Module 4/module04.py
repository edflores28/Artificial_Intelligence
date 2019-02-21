from copy import deepcopy
from operator import add

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

costs = { '.': -1, '*': -3, '#': -5, '~': -7}

def initialize_q_table(x_size, y_size, actions):
  '''
  This routine builds and initializes a Q Table
  based on the x, y size of the world
  Args:
        x_size - The width of the world
        y_size - The height of the world
        actions - The actions within the world
  Returns:
        Q_Table - [[{(0, 0): 0.0, (0, 1): 0.0, ..}], [..], ..]
  '''
  all_actions = {key: 0.0 for key in actions}
  minus_down = {key: 0.0 for key in actions[:2]+actions[3:]}
  minus_right = {key: 0.0 for key in actions[:1]+actions[2:]}
  minus_up = {key: 0.0 for key in actions[1:]}
  minus_left = {key: 0.0 for key in actions[:-1]}
  # Build the inner world
  q_table = [[deepcopy(all_actions) for x in range(x_size-2)] for y in range(y_size-2)]
  # Build the top most and bottom most rows
  q_table.append([])
  q_table[len(q_table)-1] = [deepcopy(minus_down) for x in range(x_size-2)]
  q_table.reverse()
  q_table.append([])
  q_table[len(q_table)-1] = [deepcopy(minus_up) for x in range(x_size-2)]
  q_table.reverse()
  # Build the left most and right most portions of the world
  q_table = list(zip(*q_table))
  q_table.append([])
  q_table[len(q_table)-1] = [deepcopy(minus_right) for y in range(len(q_table[0]))]
  q_table.reverse()
  q_table.append([])
  q_table[len(q_table)-1] = [deepcopy(minus_left) for y in range(len(q_table[0]))]
  q_table.reverse()
  q_table = list(zip(*q_table))
  # Remove the actions not needed on the corners of the world
  q_table[0][0].pop((0,-1))
  q_table[0][x_size-1].pop((0,-1))
  q_table[y_size-1][0].pop((0,1))
  q_table[y_size-1][x_size-1].pop((0,1))
  return q_table

def get_action(state, q_table):
  x, y = state
  return max(q_table[y][x].keys(), key=lambda k: q_table[y][x][k])

def calculate_q_value(alpha, gamma, state, action, reward, next_state, q_table):
  '''
  This routine updates the Q value for the state and action
  '''
  x, y = state
  x1, y1 = next_state
  max_q_next = get_action(next_state, q_table)
  q_table[y][x][action] = (1 - alpha) * q_table[y][x][action] + alpha * (reward + gamma*q_table[y1][x1][max_q_next])

def get_rewards(state, rewards, world):
  x, y = state
  try:
    return rewards[world[y][x]]
  except:
    return -100

def q_learning(world, costs, goal, reward, actions, gamma, alpha):
  # Create the Q table initialied to 0.0
  q_table = initialize_q_table(len(world[0]), len(world), actions)
  # Initialize the state
  state = (0,0)
  # Obtain an action
  action = get_action(state, q_table)
  # Determine the next state
  next_state = tuple(map(add, state, action))
  calculate_q_value(alpha, gamma, state, action, get_rewards(state,reward,world), next_state, q_table)
  print(q_table[0][0][action])
  return

def pretty_print_policy(rows, cols, policy):
    pass

if __name__ == "__main__":
    goal = (5, 6)
    gamma = 0.9  # FILL ME IN
    alpha = 0.5  # FILL ME IN
    reward = costs #0.0  # FILL ME IN
    test_policy = q_learning(test_world, costs, goal, reward, cardinal_moves, gamma, alpha)
    # rows = 0  # FILL ME IN
    # cols = 0  # FILL ME IN
    # pretty_print_policy(rows, cols, test_policy)
    # print()

    # goal = (26, 26)
    # gamma = 0.0  # FILL ME IN
    # alpha = 0.0  # FILL ME IN
    # reward = 0.0  # FILL ME IN
    # full_policy = q_learning(full_world, costs, goal, reward, cardinal_moves, gamma, alpha)
    # rows = 0  # FILL ME IN
    # cols = 0  # FILL ME IN
    # pretty_print_policy(rows, cols, full_policy)
    # print()