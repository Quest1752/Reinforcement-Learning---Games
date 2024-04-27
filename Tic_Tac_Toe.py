import random
from typing import Dict, List, Tuple
import numpy as np

from check_submission import check_submission
from game_mechanics import (
    Cell,
    WildTictactoeEnv,
    choose_move_randomly,
    human_player,
    load_dictionary,
    play_wild_ttt_game,
    render,
    save_dictionary,
)

TEAM_NAME = "The Curry-Cheese Alliance"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

      

# for _ in range(10):
#   state, reward, done, _ = env.reset()
#   while not done:
#     #print(state,reward,done)
#     position = random.randint(0,8)
#     while not (state[position] == " "):
#       position = random.randint(0,8)
#     action = (position,random.choice('XO'))
#     state, reward, done, _ = env.step(action)
#   print(state,reward,done)

def train() -> Dict:
    """Write this function to train your algorithm.
    
    Returns:
         Value function dictionary used by your agent. You can
         structure this how you like, however your choose_move must
         be able to use it.
    """
    
    env = WildTictactoeEnv(opponent_choose_move=choose_move_randomly)
    value_fn = {}
    alpha = 0.2
    # For 1000 episodes
    for _ in range(100000):
        # Each iteration is one episode. So we need to call reset() every time
        state, reward, done, _ = env.reset()
        # This while loop runs the episode
        while not done:
        #print(state,reward,done)
            position = random.randint(0,8)
            while not (state[position] == " "):
                position = random.randint(0,8)
            action = (position,random.choice('XO'))        
            # Before calling the transition function, keep the prev state for the TD update step
            prev_state = state

            # Transition the env object using the .step() function
            state, reward, done, _ = env.step(action)
  
            # TD update step
            if value_fn.get(str(prev_state)) == None:
                value_fn[str(prev_state)] = 0
            if value_fn.get(str(state)) == None:
                value_fn[str(state)] = 0
          
            value_fn[str(prev_state)] += alpha * (reward + value_fn[str(state)] - value_fn[str(prev_state)])
        
    #print(value_fn)
    return value_fn    
    raise NotImplementedError("You need to implement this function!")

def choose_move(board: List[str], value_function: Dict) -> Tuple[int, str]:
    """
    TODO: WRITE THIS FUNCTION

    This is what will be called during competitive play.
    It takes the current state of the board as input.
    It returns a single move to play.

    Args:
        board: list representing the board.
                (see README Technical Details for more info)

        value_function: The dictionary output by train().

    Returns:
        position (int): The position to place your piece
                        (an integer 0 -> 8), where 0 is
                        top left and 8 is bottom right.
        counter (str): The counter to place. "X" or "O".

    It's important that you think about exactly what this
     function does when you submit, as it will be called
     in order to take your turn!
    # """
    # max_value = -np.Inf
    # best_actions = []
    # possible_actions = [count for count, item in enumerate(board) if item == Cell.EMPTY]
    # print(possible_actions)
    # for poss_action in possible_actions:
    #     poss_new_state = transition_function(current_state, poss_action)
    #     if value_fn[poss_new_state] > max_value:
    #         best_actions = [poss_action]
    #         max_value = value_fn[poss_new_state]
    #     elif math.isclose(value_fn[poss_new_state], max_value, abs_tol=1e-4):
    #         best_actions.append(poss_action)
    # return random.choice(best_actions)
    #We provide an example here that chooses a random position on the board and
    #and places a random counter there.
    position = random.choice([count for count, item in enumerate(board) if item == Cell.EMPTY])
    counter = random.choice([Cell.O, Cell.X])
    return position, counter
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    ## Example workflow, feel free to edit this! ###
    my_value_fn = train()
    save_dictionary(my_value_fn, TEAM_NAME)

    # Make sure I do not error, or your submission will
    # not work in the competition
    check_submission(TEAM_NAME)

    my_value_fn = load_dictionary(TEAM_NAME)

    def choose_move_no_value_fn(board: List[str]) -> Tuple[int, str]:
        """The arguments in play_wild_ttt_game() require 
        functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(board, my_value_fn)

    # Play against your bot!!
    # Click on the board to take a move.
        # Left click to place an `O`.
        # Right click to place an `X`.
    play_wild_ttt_game(
        your_choose_move=human_player,
        opponent_choose_move=choose_move_no_value_fn,
        game_speed_multiplier=100,
        verbose=True,
        render=True,
    )
