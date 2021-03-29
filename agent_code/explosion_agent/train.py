import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
# explosion agent
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # not used
RECORD_ENEMY_TRANSITIONS = 1.0  # not used
GAMMA = 0.1
ALPHA = 0.9

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
    
    # get old q-value for old position
    if new_game_state["step"] != 1:
        _, score_old, bombs_left_old, (x_old, y_old) = old_game_state["self"]
        old_q_value = self.explosiontable[y_old][x_old][self_action]
        temp_diff = reward_from_events(self, events) + GAMMA * np.max(self.explosiontable[y][x][[0,1,2,3,4,5]]) - old_q_value
        new_q_value = old_q_value + (ALPHA * temp_diff)
        self.explosiontable[y_old][x_old][self_action] = new_q_value
        


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    
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
        
        old_q_value = self.explosiontable[y_old][x_old][last_action]
        temp_diff = reward_from_events(self, events) + GAMMA * np.max(self.explosiontable[y][x][0:6]) - old_q_value
        new_q_value = old_q_value + (ALPHA * temp_diff)
        self.explosiontable[y_old][x_old][last_action] = new_q_value
    # Store the q-table
    with open("my-saved-explosiontable.pt", "wb") as file:
        pickle.dump(self.explosiontable, file)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 0,
        e.KILLED_OPPONENT: 0,
        # idea: end game as fast as possible
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.WAITED: -2,
        e.BOMB_DROPPED: -3,
        # invalid action is bad
        e.INVALID_ACTION: -1000,
        #e.KILLED_SELF: -2000
        }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum