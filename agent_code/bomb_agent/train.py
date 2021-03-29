import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
# bomb agent

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # not used
RECORD_ENEMY_TRANSITIONS = 1.0  # not used
GAMMA = 0.7
ALPHA = 0.9

# Events
BOMB_0 = 'BOMB_0'
BOMB_1 = 'BOMB_1'
BOMB_2 = 'BOMB_2'
BOMB_3 = 'BOMB_3'
BOMB_4 = 'BOMB_4'


def setup_training(self):
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # get own new position
    _, score, bombs_left, (x, y) = new_game_state["self"]
    safe_spots=[(3,7),(5,6),(5,8),(6,5),(6,9),(7,3),(7,11),(8,5),(8,9),(9,6),(9,8),(11,7)]
    end_spots=[(2,7),(3,6),(3,8),(5,5),(5,9),(6,3),(6,11),(7,2),(7,12),(8,3),(8,11),(9,5),(9,9),(11,6),(11,8),(12,7)]
    # Idea: Add your own events to hand out rewards -> auxially rewards
    if (x,y) == (7,7):
        events.append(BOMB_0)
    elif (x,y) == (7,6) or (x,y) == (7,8) or (x,y) == (6,7) or (x,y) == (8,7):
        events.append(BOMB_1)
    elif (x,y) == (7,5) or (x,y) == (7,9) or (x,y) == (5,7) or (x,y) == (9,7):
        events.append(BOMB_2)
    elif (x,y) == (7,4) or (x,y) == (7,10) or (x,y) == (4,7) or (x,y) == (10,7):
        events.append(BOMB_3)
    elif (x,y) in safe_spots:
        events.append(BOMB_4)
        
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
    
    # get old q-value for old position
    if new_game_state["step"] != 1:
        _, score_old, bombs_left_old, (x_old, y_old) = old_game_state["self"]
        if self_action != 4:
            old_q_value = self.bombtable[y_old][x_old][self_action]
            temp_diff = reward_from_events(self, events) + GAMMA * np.max(self.bombtable[y,x,[0,1,2,3,5]]) - old_q_value
            new_q_value = old_q_value + (ALPHA * temp_diff)
            self.bombtable[y_old][x_old][self_action] = new_q_value
        


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the q-table
    with open("my-saved-bombtable.pt", "wb") as file:
        pickle.dump(self.bombtable, file)
        

def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 0,
        # idea: end game as fast as possible
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        e.MOVED_UP: -1,
        e.WAITED: -2,
        BOMB_0: -900,
        BOMB_1: -800,
        BOMB_2: -700,
        BOMB_3: -600,
        BOMB_4: 900,
        e.BOMB_DROPPED: -3,
        # invalid action is bad
        e.INVALID_ACTION: -1000
        #e.KILLED_SELF: -2000
        }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum