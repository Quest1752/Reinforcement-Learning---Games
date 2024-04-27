import random
from typing import Dict, Tuple
from tqdm import tqdm

import numpy as np

# from my_tools import (
# get_win_percentage,
# choose_left,
# choose_middle)

from check_submission import check_submission
from game_mechanics import (
    Connect4Env,
    choose_move_randomly,
    get_empty_board,
    get_piece_longest_line_length,
    get_top_piece_row_index,
    has_won,
    is_column_full,
    load_dictionary,
    place_piece,
    play_connect_4_game,
    reward_function,
    save_dictionary,
    human_player,
)

TEAM_NAME = "The Curry-Cheese Allaince"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

def to_feature_vector(state: np.ndarray) -> Tuple:
    """
    TODO: Write this function to convert the state to a
           feature vector.

    We suggest you use functions in game_mechanics.py to
     make a handcrafted feature vector based on the
     state of the board.

    Args:
        state: board state as a np array. 1's are your
               pieces. -1's your opponent's pieces & 0's
               are empty.

    Returns:
        the feature vector for this state, as
         designed by you.
    """
    feature_list = []
    #Checking for largest line length in each column
    for col_idx in range(8):
      row_idx = get_top_piece_row_index(state, col_idx)
      #print(row_idx,col_idx)
      if row_idx == None:
        feature_list.append(get_piece_longest_line_length(state,(5, col_idx)))
      else:
        feature_list.append(get_piece_longest_line_length(state,(row_idx-1, col_idx)))

    return tuple(feature_list)

def train() -> Dict:
    """
    TODO: Write this function to train your algorithm.

    Returns:
        Value function dictionary used by your agent.
         You can structure this how you like, but
         choose_move() expects {feature_vector: value}.
    """
    alpha = 0.2
    epsilon = 0.3
    env = Connect4Env()

    # Expected format: {feature_vector: value}, where feature_vector is a tuple and value is a float
    value_fn = {}

    # Requires fewer episodes of experience because there are fewer states to learn
    for _ in range(1000):
        state, reward, done, _ = env.reset()
        #print(state)
        while not done:
            # Epsilon-greedy policy
            #if random.random() < epsilon:
          action = choose_move_randomly(state)
            #else:
               # action = choose_move(state, value_fn) #Issue
          #print(action)
          prev_state = state
          state, reward, done, _ = env.step(action)
        
          prev_features = to_feature_vector(prev_state)
          features = to_feature_vector(state)

          if not done:
            # TD update rule
            value_fn[prev_features] = (1 - alpha) * value_fn.get(prev_features, 0) + alpha * (reward + value_fn.get(features, 0))
        else:
            # Slightly different update since we're setting 
            #  terminal state == reward
            value_fn[prev_features] = (1 - alpha) * value_fn.get(prev_features, 0) + alpha * reward
    
    # Update terminal state
    value_fn[features] = reward
  
    return value_fn
    raise NotImplementedError("You need to implement the train() function!")


def choose_move(state: np.ndarray, value_function: Dict, verbose: bool = False) -> int:
    """
    Called during competitive play. It acts greedily given
    current state of the board and value function dictionary.
    It returns a single move to play.

    Args:
        state: State of the board as a np array. Your pieces are
                1's, the opponent's are -1's and empty are 0's.
        value_function: The dictionary output by train().

    Returns:
        position (int): The column you want to place your counter
                        into (an integer 0 -> 7), where 0 is the
                        far left column and 7 is the far right
                        column.
    """
    values = []
    not_full_cols = [col for col in range(state.shape[1]) if not is_column_full(state, col)]

    for not_full_col in not_full_cols:
        # Do 1-step lookahead and compare values of successor states
        state_copy = state.copy()
        place_piece(board=state_copy, column_idx=not_full_col, player=1)

        # Get the feature vector associated with the successor state
        features = to_feature_vector(state_copy)
        if verbose:
            print(
                "Column index:",
                not_full_col,
                "Feature vector:",
                features,
                "Value:",
                value_function.get(features, 0),
            )

        # Add the action value to the values list
        action_value = value_function.get(features, 0) + reward_function(not_full_col, state_copy)
        values.append(action_value)

    # Pick randomly between max value actions
    max_value = max(values)
    value_indices = [index for index, value in enumerate(values) if value == max_value]
    value_index = random.choice(value_indices)
    return not_full_cols[value_index]

if __name__ == "__main__":
    # Example workflow, feel free to edit this! ###
    my_value_fn = train()
    #print(my_value_fn)
    save_dictionary(my_value_fn, TEAM_NAME)

    check_submission(
        TEAM_NAME
    )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    my_value_fn = load_dictionary(TEAM_NAME)

    # Code below plays a single game of Connect 4 against a random
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_value_fn(state: np.ndarray) -> int:
        """
        The arguments to play_connect_4_game() require functions that 
            only take the state as input.
        """
        return choose_move(state, my_value_fn)
    results = []
    rounds = 1000
    # Play a game against your bot! Click a column to
    # place a counter!
    for _ in range(rounds):
      results.append(play_connect_4_game(
        your_choose_move=choose_move_no_value_fn,
        opponent_choose_move=choose_move_no_value_fn,
        game_speed_multiplier=10,
        render=False,
        verbose=False,
    ))
    print((results.count(1)/rounds)*100)
