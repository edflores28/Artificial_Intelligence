import random
import datetime
from copy import deepcopy
from operator import add

sysrand = random.SystemRandom()
EPISODES = 500

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
  return [[deepcopy(all_actions) for x in range(x_size)] for y in range(y_size)]

def get_best_q_action(state, q_table):
  '''
  This routine returns the best action

  Args:
      state - The state
      q_table - The q tables
  Returns:
      The best action based on Q
  '''
  x,y = state
  return max(q_table[y][x].keys(), key=lambda k: q_table[y][x][k])

def get_action(state, q_table):
  '''
  This routine gets an action on the given state. The best
  Q action is returned if a random number is <= 0.7. Otherwise
  a random action is returned that is not the best q action

  Args:
      state - The state
      q_table - The q table
  Returns:
      An action
  '''
  x, y = state
  actions = list(q_table[y][x].keys())
  selected_action = max(q_table[y][x].keys(), key=lambda k: q_table[y][x][k])
  # Generate a random value and if <= 0.70 then
  # the selected action is used. Otherwise pick an 
  # action that is not the best
  if sysrand.random() <= 0.70:
    return selected_action, selected_action
  else:
    actions.pop(actions.index(selected_action))
    return selected_action, sysrand.choice(actions)

def determine_next_state_and_reward(state, action, rewards, world, goal):
  '''
  This routine determines the next state and assocated reward

  Args:
      state - The state
      action - The action
      rewards - The rewards
      world - The world
      goal - The goal state
  Returns:
      The next state and reward
  '''
  IMPASSABLE_REWARD = -100
  GOAL_REWARD = 100
  # Determine the next state
  next_state = tuple(map(add, state, action))
  # Determine whether next state is the goal state
  # If so return the next state and goal reward
  if next_state == goal:
    return next_state, GOAL_REWARD
  # For all non goal states return the next state
  # and it's corresponding reward
  try:
    x, y = next_state
    if x < 0 or y < 0:
      raise Exception
    return next_state, rewards[world[y][x]]
  # If an exception ocurred then that means the next state is
  # out of bounds or the next state is an 'x' which is impassable.
  # Return the current state and the impassable reward
  except:
    return state, IMPASSABLE_REWARD

def calculate_q_value(alpha, gamma, state, action, reward, next_state, q_table):
  '''
  This routine updates the Q value for the state and action

  Args:
      alpha - The learning rate
      gamma - The discount factor
      state - The state
      action - The action
      reward - The reward
      next_state - The next state
      q_table - The Q table
  Returns:
      Nothing
  '''
  x, y = state
  x1, y1 = next_state
  max_q_next = get_best_q_action(next_state, q_table)
  q_table[y][x][action] = (1 - alpha) * q_table[y][x][action] + alpha * (reward + gamma*q_table[y1][x1][max_q_next])

def q_learning(world, costs, goal, reward, actions, gamma, alpha):
  '''
  The Q learning algorithm

  Args:
      world - The worled
      costs - The costs for each state
      goal  - The goal state
      actions - The actions allowed within the world
      gamma - The discount factor
      alpha - The learning rate
  Returns:
      The policy
  '''
  print("Total episodes", EPISODES)
  # Create the Q table initialied to 0.0
  q_table = initialize_q_table(len(world[0]), len(world), actions)
  # Iterate over all the episodes
  for episode in range(EPISODES):
    if len(world) > 10 and episode % 100 == 0:
      print("Episode:", episode+1)
    # Initialize the state
    state = (0,0)
    while state != goal:
      # Obtain an action
      selected_action, actual_action = get_action(state, q_table)
      # Determine the next state and it's reward
      next_state, next_reward = determine_next_state_and_reward(state, actual_action, reward, world, goal)
      # Calculate the q value
      calculate_q_value(alpha, gamma, state, selected_action, next_reward, next_state, q_table)
      state = next_state
  policy = {}
  # Iterate over the Q table and selected the
  # best action to build the policy
  for y in range(len(q_table)):
    for x in range(len(q_table[y])):
      if (x, y) != goal:
        policy[(x,y)] = get_best_q_action((x,y), q_table)
      else:
        policy[(x,y)] = (0,0)
  return policy

def pretty_print_policy(rows, cols, policy):
  '''
  This routine prints out the policy for the size of the world

  Args:
      rows - The total rows in the world
      cols - The total columns in the world
  Returns:
      Nothing
  '''
  # Dictionary for each actions text representation
  directions = {(0, 1): 'v', (0, -1): '^', (1, 0): '>', (-1, 0): '<', (0, 0): 'G'}
  # Iterate over the rows and cols and print the
  # corresponding action
  for y in range(rows):
    for x in range(cols):
      print(directions[policy[(x,y)]], end='' if x < cols-1 else '\n')

if __name__ == "__main__":
    goal = (5, 6)
    gamma = 0.9
    alpha = 0.3
    reward = costs
    print("Test World Start")
    test_policy = q_learning(test_world, costs, goal, reward, cardinal_moves, gamma, alpha)
    rows = len(test_world)
    cols = len(test_world[0])
    pretty_print_policy(rows, cols, test_policy)
    print()

    goal = (26, 26)
    gamma = 0.7
    alpha = 0.15
    reward = costs
    print("Full World Start")
    full_policy = q_learning(full_world, costs, goal, reward, cardinal_moves, gamma, alpha)
    rows = len(full_world)
    cols = len(full_world[0])
    pretty_print_policy(rows, cols, full_policy)
    print()