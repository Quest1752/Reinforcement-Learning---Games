import random

import numpy as np
import torch
from torch import nn
from typing import Tuple

from check_submission import check_submission
from game_mechanics import (
    PongEnv,
    human_player,
    load_network,
    play_pong,
    robot_choose_move,
    save_network,
)

#Useful Tools:
class ExperienceReplayMemory:
    def __init__(self, max_length: int):
        self.states = []
        self.actions = []
        self.rewards = []
        self.successor_states = []
        self.is_terminal = []
        self.max_length = max_length
        self._all_lists = [
            self.states,
            self.actions,
            self.rewards,
            self.successor_states,
            self.is_terminal,
        ]

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        successor_state: np.ndarray,
        is_terminal: bool,
    ):
        for l, item in zip(
            self._all_lists, [state, action, reward, successor_state, is_terminal]
        ):
            dtype = (
                bool
                if isinstance(item, bool)
                else int
                if isinstance(item, int)
                else torch.float32
            )
            l.append(torch.tensor(item, dtype=dtype))
            # If max length, remove the oldest states
            if len(l) > self.max_length:
                l.pop(0)

    def sample(self, size: int) -> Tuple[torch.Tensor, ...]:
        size = min(size, len(self.rewards))
        idxs = np.random.choice(
            list(range(len(self.rewards))), replace=False, size=size
        )
        return tuple(torch.stack([l[idx] for idx in idxs]) for l in self._all_lists)


TEAM_NAME = "Curry Cheese Alliance"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"

def train() -> nn.Module:
    """
    TODO: Write this function to train your algorithm.

    Returns:
        A pytorch network to be used by choose_move. You can architect
        this however you like but your choose_move function must be able
        to use it.
    """
    env = PongEnv(steps_per_state=50)

    # Hyperparameters - given you some to narrow the search space
    gamma = 0.9
    epsilon = 0.1
    batch_size = 128  # Can be lowered if this is slow to run
    lr = 0.01
    max_num_episodes = 10000
    max_memory_size = 100000  # This shouldn't matter

    n_neurons = 64

    Q = nn.Sequential(
        nn.Linear(6, n_neurons),
        nn.ReLU(),
        nn.Linear(n_neurons, 1),
    )

    optim = torch.optim.Adam(Q.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    memory = ExperienceReplayMemory(max_length=max_memory_size)

    #num_wins, num_losses, num_steps, ep_losses = 0, 0, [], []
    #moving_avg = 0
    episodes = 0
    for episode_num in range(max_num_episodes):
        # This moving average is an early stopping condition: aim is 
        #  200 steps, so average of 200 with epsilon-greedy will solve
        #if moving_avg > 200:
            #break
        # Counts number of steps this episode
        #n_steps = 0
        episodes = episodes + 1
        print(episodes)
        successor_state, reward, done, _ = env.reset()
        #print(successor_state, reward, done)
        while not done:
            prev_state = successor_state.copy()
            if np.random.random() < epsilon:
              action = np.random.randint(-1,1)
            else:
              with torch.no_grad():
                action = torch.argmax(Q(torch.tensor(successor_state, dtype=torch.float32))).item()
                
            # try:
            successor_state, reward, done, _ = env.step(action)
            # except:
            #   successor_state, reward, done, _ = env.reset()
            # else:
            #   successor_state, reward, done, _ = env.step(action)

                        
              
            memory.append(prev_state, action, reward, successor_state, done)

            if len(memory.rewards) >= batch_size:
                s1, a1, r1, s2, is_terminals = memory.sample(batch_size)
    
                # Update steps
                q_values_chosen_1 = Q(s1)[range(len(s1)), a1]
                with torch.no_grad():
                    chosen_successor_action = torch.argmax(Q(s2), dim=1)
                    max_q_successor = (
                        Q(s2)[range(len(s2)), chosen_successor_action] * ~is_terminals
                    )
    
                loss = loss_fn(q_values_chosen_1, r1 + gamma * max_q_successor)
    
                # Updates the parameters!
                optim.zero_grad()
                loss.backward()
                optim.step()
    return Q
    raise NotImplementedError("You need to implement this function")


def choose_move(
    state: np.ndarray,
    neural_network: nn.Module,
) -> int:  # <--------------- Please do not change these arguments!
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        state: State of the game as a np array, length = 6.
        network: The pytorch network output by train().

    Returns:
        move (int): The move you want to given the state of the game.
                    Should be in {-1,0,1}
    """
    Q_values = neural_network(torch.tensor(state, dtype=torch.float32))
    action = torch.argmax(Q_values).item()
    return action
    
    raise NotImplementedError("You need to implement this function")


if __name__ == "__main__":
    # Example workflow, feel free to edit this! ###
    my_network = train()
    save_network(my_network, TEAM_NAME)

    # # Make sure this does not error! Or your
    # submission will not work in the tournament!
    check_submission(TEAM_NAME)

    my_network = load_network(TEAM_NAME)

    #  Code below plays a single game of pong against a basic robot player
    #  opponent, think about how you might want to adapt this to
    #  test the performance of your algorithm.
    def choose_move_no_network(state) -> int:
        """The arguments in play_pong_game() require functions that only take the state as input.

        This converts choose_move() to that format.
        """
        return choose_move(state, neural_network=my_network)

    # # Play against your bot!!
    play_pong(
        your_choose_move=choose_move_no_network,
        opponent_choose_move=robot_choose_move,
        game_speed_multiplier=100,
        verbose=True,
        render=False,
    )
  
