import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = 0.7 #discount factor
ALPHA = 0.9
bigtable_on = 1


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    
    #ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
    if self_action == "UP":
        self_action = 0
    elif self_action == "RIGHT":
        self_action = 1
    elif self_action == "DOWN":
        self_action = 2
    elif self_action == "LEFT":
        self_action = 3
    elif self_action == "BOMB":
        self_action = 4
    elif self_action == "WAIT":
        self_action = 5
        
    # get own new position
    _, score, bombs_left, (x, y) = new_game_state["self"]
    
    if new_game_state["step"] != 1:
        # get old q-value for old position
        _, score_old, bombs_left_old, (x_old, y_old) = old_game_state["self"]
        
        old_q_value = self.qtable[y_old][x_old][self_action]
        temp_diff = reward_from_events(self, events) + GAMMA * np.max(self.qtable[(y, x)][0:6]) - old_q_value
        new_q_value = old_q_value + (ALPHA * temp_diff)
        self.qtable[y_old][x_old][self_action] = new_q_value


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    #ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
    if last_action == "UP":
        last_action = 0
    elif last_action == "RIGHT":
        last_action = 1
    elif last_action == "DOWN":
        last_action = 2
    elif last_action == "LEFT":
        last_action = 3
    elif last_action == "BOMB":
        last_action = 4
    elif last_action == "WAIT":
        last_action = 5
        
    # get own new position
    _, score, bombs_left, (x, y) = last_game_state["self"]
    
    if last_game_state["step"] != 1:
        # get old q-value for old position
        _, score_old, bombs_left_old, (x_old, y_old) = last_game_state["self"]
        
        old_q_value = self.qtable[y_old][x_old][last_action]
        temp_diff = reward_from_events(self, events) + GAMMA * np.max(self.qtable[(y, x)][0:6]) - old_q_value
        new_q_value = old_q_value + (ALPHA * temp_diff)
        self.qtable[y_old][x_old][last_action] = new_q_value

    # Store the q-table
    with open("my-saved-qtable.pt", "wb") as file:
        pickle.dump(self.qtable, file)
    
    if(bigtable_on == 1):
        self.coin1table = build_big_qtable(self.qtable[1:16,1:16])
    
        with open("my-saved-coin1table.pt", "wb") as file:
            pickle.dump(self.coin1table, file)
        
    


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 200,
        # idea: end game as fast as possible
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.BOMB_DROPPED: -3,
        e.WAITED: -2,
        # invalid action is bad
        e.INVALID_ACTION: -10000
        }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def build_big_qtable(small_qtable):
    # initialize big q table
    small_qtable[0:15,0,3] = small_qtable[0:15,0,1]
    small_qtable[0,0:15,0] = small_qtable[0,0:15,2]
    big_qtable = np.zeros((29,29,6))
    # copy small q table in bottom right corner
    big_qtable[14:29, 14:29, :] = small_qtable
    
    # flip and change right <-> left for botton left corner
    flip = np.fliplr(small_qtable)
    big_qtable[14:29, 0:14, :] = flip[0:15, 0:14, [0,3,2,1,4,5]]
    # flip and change up <-> down for upper half
    flip2 = np.flipud(big_qtable[14:29, 0:29])
    big_qtable[0:14, 0:29, :] = flip2[0:14, 0:29, [2,1,0,3,4,5]]
    return big_qtable