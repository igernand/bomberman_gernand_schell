import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
# bomb crate 2 agent
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # not used
RECORD_ENEMY_TRANSITIONS = 1.0  # not used
GAMMA = 0.7
ALPHA = 0.9
bigtable_on = 0

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
BOMB_0 = 'BOMB_0'
BOMB_1 = 'BOMB_1'
BOMB_2 = 'BOMB_2'
BOMB_3 = 'BOMB_3'
BOMB_4 = 'BOMB_4'


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # get own new position
    _, score, bombs_left, (x, y) = new_game_state["self"]
    
    # Idea: Add your own events to hand out rewards -> auxially rewards
    if (x,y) == (8,7):
        events.append(BOMB_0)
    elif (x,y) == (7,7) or (x,y) == (9,7):
        events.append(BOMB_1)
    elif (x,y) == (6,7) or (x,y) == (10,7) or (x,y) == (7,8) or (x,y) == (7,6) or (x,y) == (9,6) or (x,y) == (9,8):
        events.append(BOMB_2)
    elif (x,y) == (5,7) or (x,y) == (11,7) or (x,y) == (7,9) or (x,y) == (7,5) or (x,y) == (9,5) or (x,y) == (9,9):
        events.append(BOMB_3)
    elif (x,y) == (4,7) or (x,y) == (12,7) or (x,y) == (7,10) or (x,y) == (7,4) or (x,y) == (9,4) or (x,y) == (9,10):
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
        
    

    
    if new_game_state["step"] != 1:
        _, score_old, bombs_left_old, (x_old, y_old) = old_game_state["self"]
        old_q_value = self.bomb2table[y_old][x_old][self_action]
        temp_diff = reward_from_events(self, events) + GAMMA * np.max(self.bomb2table[y,x,[0,1,2,3,5]]) - old_q_value
        new_q_value = old_q_value + (ALPHA * temp_diff)
        self.bomb2table[y_old][x_old][self_action] = new_q_value
        


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    self.bomb3table = build_qtable3(self.bomb2table)
    
    # Store the q-table
    with open("my-saved-bomb2table.pt", "wb") as file:
        pickle.dump(self.bomb2table, file)
    with open("my-saved-bomb3table.pt", "wb") as file:
        pickle.dump(self.bomb3table, file)

        


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.KILLED_OPPONENT: 0,
        PLACEHOLDER_EVENT: 0,  # idea: the custom event is bad
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
        BOMB_4: -500,
        e.BOMB_DROPPED: 0,
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


def build_qtable3(small_qtable): 
    # transpose small_qtable
    small_qtable = small_qtable.transpose(1, 0, 2)
    # change walking directions
    small_qtable = small_qtable[0:17, 0:17, np.array([3,2,1,0,4,5])]
    return(small_qtable)