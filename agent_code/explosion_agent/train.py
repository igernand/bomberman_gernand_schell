import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
#coin1 agent
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = 0.7
ALPHA = 0.9
bigtable_on = 0

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


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

    # Idea: Add your own events to hand out rewards -> auxially rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

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
        temp_diff = reward_from_events(self, events) + GAMMA * np.max(self.explosiontable[y][x][[0,1,2,3,5]]) - old_q_value
        new_q_value = old_q_value + (ALPHA * temp_diff)
        self.explosiontable[y_old][x_old][self_action] = new_q_value
        


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

    # Store the model
    """with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)"""
    
    # Store the q-table
    with open("my-saved-explosiontable.pt", "wb") as file:
        pickle.dump(self.explosiontable, file)
        
    if(bigtable_on == 1):
        self.crate1table = build_big_qtable(self.qtable[1:16,1:16])
    
        with open("my-saved-crate1table.pt", "wb") as file:
            pickle.dump(self.crate1table, file)
    

        


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
        
        e.BOMB_DROPPED: 0,
        # invalid action is bad
        e.INVALID_ACTION: -1000,
        e.KILLED_SELF: -2000
        }
    
    reward_sum = 0
    """for ev in events:
        if ev == e.INVALID_ACTION:
            print("invalid")"""
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