import push
import random
import math

# The edges of the game
edges = ['T', 'B', 'L', 'R']

# Constant values
MAX_VALUE = 1
NO_VALUE = 0
MIN_VALUE = -1
MAXIMIZER = 'X'
MINIMIZER = 'O'
MINIMAX_PLY = 3
ALPHA_BETA_PLY = 6

def print_board(board):
    '''
    This routine prints out the board
    '''
    for x in board:
        print(*x)
    print("\n")

def find_winner(board):
    ''' 
    This routine determines which player has
    the greater amount of straights. If neither
    have any then None is returned

    Args:
        board - The current state of the board
    Returns:
        The winning player
    '''
    results = push.straights(board)
    if results['X'] > results['O']:
        return 'X'
    elif results['O'] > results['X']:
        return 'O'
    return None

def get_value(player):
    '''
    Obtain the value of the corresponding
    player

    Returns:
        The value of of the players winning state
    '''
    if player is 'X':
        return MAX_VALUE
    elif player is 'O':
        return MIN_VALUE
    return NO_VALUE

def duplicate_exists(board, board_states):
    '''
    This routine determines if the board
    has been in it's current state previously 
 
    Args:
        board - The current state of the board
    Returns:
        True if a previous state, otherwise, False
    '''
    for state in board_states:
        if push.equal(board, state):
            return True,
    return False

def is_end_game (board, board_states, player):
    '''
    Determines wheter the game is ended

    Args:
        board - The board after the players turn
        player - The player
    Returns:
        Boolean value indicating if the game
        has ended and the winning player
    '''
    # Check to see if the board has been
    # duplicated, if so return True and
    # the opposite player as the winner
    if duplicate_exists(board, board_states):
        return True, 'X' if player is 'O' else 'O'
    # Check to see who has the most amount
    # of straights
    winner = find_winner(board)
    # If a winner has been found return
    # True and the winner
    if winner is not None:
        return True, winner
    # Return False and None
    return False, None

def generate_random_move(player, board_size):
    '''
    This routine generates a random move
    for the given player

    Args:
        player - The player that is generating 
        board  - The size of the board
    Returns:
        The players random move
    '''
    edge = random.choice(['T', 'B', 'L', 'R'])
    index = random.choice(range(1, board_size+1))
    return (player, edge, index)

def get_moves(player, board_size):
    '''
    This routine generates the move list for
    a given player

    Args:
        player - The player that is generating 
        board  - The size of the board
    Returns:
        A list of the players move randomized
    '''
    moves = [(player, x, y+1) for x in edges for y in range(board_size)]
    random.shuffle(moves)
    return moves

def max_values(board, alpha, beta, ply, current_ply):
    '''
    This routine performs the maximizing portion of
    the minimax algorithm. Alpha Beta pruning is
    performed when alpha and beta set to valid values

    Args:
        board - The board
        alpha - The alpha value
        beta  - The beta value
        ply   - Total amount of plys to search
        current_ply - The current ply
    Returns:
        The best value and move
    ''' 
    # Determine if the board has reached a terminal state
    winner = find_winner(board)
    if winner != None or current_ply == ply:
        return get_value(winner), None
    # Set the best value and best move
    best_value = -math.inf
    best_move = None
    # Iterate over all the available moves
    for next_move in get_moves(MAXIMIZER, len(board)):
        # Apply the move to the board
        next_board = push.move(board, next_move)
        # Minimize the next board
        value, move = min_values(next_board, alpha, beta, ply, current_ply+1)
        # If the value is greater than the best value
        # update the best value and best move
        if value > best_value:
            best_value = value
            best_move = next_move
        # When alpha and beta are != None then
        # alpha beta pruning is performed
        if alpha != None and beta != None:
            # Determine if next nodes don't need to
            # be checked and return the value and move    
            if best_value >= beta:
                return best_value, best_move
            # Update the value for alpha
            alpha = max(alpha, best_value)
    return best_value, best_move


def min_values(board, alpha, beta, ply, current_ply):
    '''
    This routine performs the minimizing portion of
    the minimax algorithm. Alpha Beta pruning is
    performed when alpha and beta set to valid values

    Args:
        board - The board
        alpha - The alpha value
        beta  - The beta value
        ply   - Total amount of plys to search
        current_ply - The current ply
    Returns:
        The best value and move
    '''     
    # Determine if the board has reached a terminal state
    winner = find_winner(board)
    if winner != None or current_ply == ply:
        return get_value(winner), None
    # Set the best value and best move
    best_value = math.inf
    best_move = None
    # Iterate over all the available moves
    for next_move in get_moves(MINIMIZER, len(board)):
        # Apply the move to the board
        next_board = push.move(board, next_move)
        # Maximize the next board
        value, move = max_values(next_board, alpha, beta, ply, current_ply+1)
        # If the value is less than the best value
        # update the best value and best move
        if value < best_value:
            best_value = value
            best_move = next_move
        # When alpha and beta are != None then
        # alpha beta pruning is performed
        if alpha != None and beta != None:
            # Determine if next nodes don't need to
            # be checked and return the value and move
            if best_value <= alpha:
                return best_value, best_move
            # Update the value for beta
            beta = max(beta, best_value)
    return best_value, best_move

def minimax(board, is_maximizer, ply, alpha, beta):
    '''
    This routine initiates the minimax/alpha beta pruning algorithm

    Args:
        board - The board
        is_maximizer - Indicates whether player is maximizing
        alpha - The alpha value
        beta  - The beta value
        ply   - Total amount of plys to search
    Returns:
        The best value and move
    ''' 
    if is_maximizer:
        return max_values(board, alpha, beta, ply, 0)
    else:
        return min_values(board, alpha, beta, ply, 0)

def random_minimax_game(ply, is_random_start):
    '''
    This routine plays a whole push game between
    the random and minimax player. There is no alpha
    beta pruning.

    Args:
        ply - The amount of plys to search
        is_random_start - Random player starts
    '''
    # Create the board
    board = push.create()
    # List that contains all the previous states
    board_states = []
    while True:
        # If random player is starting start with a random move
        # as the maximizer. Otherwise start with minimax as
        # the maximizer
        if is_random_start:
            board = push.move(board, generate_random_move(MAXIMIZER, len(board)))
        else:
            board = push.move(board, minimax(board, True, ply, None, None)[1])
        # Determine if there is a winner
        is_end, winner = is_end_game(board, board_states, MAXIMIZER)
        if is_end:
            return winner, board
        # Append the board to the states
        board_states.append(board)
        # If the random player is starting then the minimax player is the
        # minimizing player. Otherwise the random player is minimizing
        if is_random_start:
            board = push.move(board, minimax(board, False, ply, None, None)[1])
        else:
            board = push.move(board, generate_random_move(MINIMIZER, len(board)))
        # Determine if there is a winner
        is_end, winner = is_end_game(board, board_states, MINIMIZER)
        if is_end:
            return winner, board
        # Append the board to the states
        board_states.append(board)

def alpha_beta_minimax_game(is_alpha_beta_start, minimax_ply, alpha_beta_ply):
    '''
    This routine plays a whole push game between
    the alpha beta and minimax player. 
    
    Args:
        is_alpha_beta_start - Alpha Beta player starts
        minimax_ply - The amount of plys to for minimax player
        alpha_beta_ply - The amount of plys to for alpha beta player
    '''
    # Create the board
    board = push.create()
    # List that contains all the previous states
    board_states = []
    while True:
        # If alpha beta player is starting then it is the maximizer
        # Otherwise start with minimax as the maximizer
        if is_alpha_beta_start:
            board = push.move(board, minimax(board, True, alpha_beta_ply, -math.inf, math.inf)[1])
        else:
            board = push.move(board, minimax(board, True, minimax_ply, None, None)[1])
        # Determine if there is a winner
        is_end, winner = is_end_game(board, board_states, MAXIMIZER)
        if is_end:
            return winner, board
        # Append the board to the states
        board_states.append(board)
        # If the alph beta player is starting then the minimax player is the
        # minimizing player. Otherwise the alpha beta player is minimizing
        if is_alpha_beta_start:
            board = push.move(board, minimax(board, False, minimax_ply, None, None)[1])
        else:
            board = push.move(board, minimax(board, False, alpha_beta_ply, -math.inf, math.inf)[1])
        # Determine if there is a winner
        is_end, winner = is_end_game(board, board_states, MINIMIZER)
        if is_end:
            return winner, board
        # Append the board to the states
        board_states.append(board)            

def minimax_versus_random():
    '''
    This routine plays the push game between the
    minimax and random player
    '''
    minimax_maximizer = [True, False, True, False, True]
    minimax_wins = 0
    random_wins = 0
    board = None
    print("Minimax Player is searching", MINIMAX_PLY, "ply.\n")
    # Iteterate through each game alternating between
    # who starts
    for game in range(len(minimax_maximizer)):
        # Start the game with minimax as the starting player
        if minimax_maximizer[game]:
            print("Starting game", game+1, "Minimax Start.")
            winner, board = random_minimax_game(MINIMAX_PLY, False)
            # Determine who gets the win
            if winner is 'X':
                minimax_wins += 1
            else:
                random_wins += 1
        # Start the game with random as the starting player
        else:
            print("Starting game", game+1, "Random Start.")
            winner,board = random_minimax_game(MINIMAX_PLY, True)
            # Determine who gets the win
            if winner is 'O':
                minimax_wins += 1
            else:
                random_wins += 1
        print("The ending state of the board:")
        print_board(board)
    print("Minimax won", minimax_wins, "games.")
    print("Random won", random_wins, "games.")


def minimax_versus_alphabeta():
    '''
    This routine plays the push game between the
    minimax and alpha beta players
    '''
    minimax_maximizer = [True, False, True, False, True]
    minimax_wins = 0
    alpha_beta_wins = 0
    board = None
    print("Minimax Player is searching", MINIMAX_PLY, "ply.")
    print("Alpha Beta Player is searching", ALPHA_BETA_PLY, "ply.\n")

    # Iteterate through each game alternating between
    # who starts
    for game in range(len(minimax_maximizer)):
        # Start the game with minimax as the starting player
        if minimax_maximizer[game]:
            print("Starting game", game+1, "Minimax Start.")
            winner, board = alpha_beta_minimax_game(False, MINIMAX_PLY, ALPHA_BETA_PLY)
            # Determine who gets the win
            if winner is 'X':
                minimax_wins += 1
            else:
                alpha_beta_wins += 1
        # Start the game with alpha beta as the starting player
        else:
            print("Starting game", game+1, "Alpha Beta Start.")
            winner,board = alpha_beta_minimax_game(True, MINIMAX_PLY, ALPHA_BETA_PLY)
            # Determine who gets the win
            if winner is 'O':
                minimax_wins += 1
            else:
                alpha_beta_wins += 1
        print("The ending state of the board:")
        print_board(board)
    print("Minimax won", minimax_wins, "games.")
    print("Alph Beta won", alpha_beta_wins, "games.")


if __name__ == "__main__":
    print("Random v. Minimax")
    minimax_versus_random()
    print("\nMinimax v. Alpha Beta")
    minimax_versus_alphabeta()