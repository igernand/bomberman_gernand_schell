import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features
# crate 2agent
# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # not used
RECORD_ENEMY_TRANSITIONS = 1.0  # not used
GAMMA = 0.4
ALPHA = 0.9
bigtable_on = 1

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
BOMB_NEXT_TO_CRATE = "BOMB_NEXT_TO_CRATE"
FALSE_BOMB = "FALSE_BOMB"


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
    
     #ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB']
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
        
        # compute temporal difference        
        temp_diff = reward_from_events(self, events) + GAMMA * np.max(self.qtable[y][x][0:6]) - old_q_value
            
        # update q-value 
        new_q_value = old_q_value + (ALPHA * temp_diff)
        self.qtable[y_old][x_old][self_action] = new_q_value


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

    for event in events:
        if event == e.BOMB_DROPPED:
            old_board =  last_game_state["field"]
            _, score_old, bombs_left_old, (x_old, y_old) = last_game_state["self"]
            
            if old_board[x_old,y_old+1]==1 or old_board[x_old+1,y_old]==1 or old_board[x_old,y_old-1]==1 or old_board[x_old-1,y_old]==1:
                events.append(BOMB_NEXT_TO_CRATE)
            else:
                events.append(FALSE_BOMB)
    

     #ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB']
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
        
        # compute temporal difference
        temp_diff = reward_from_events(self, events) + GAMMA * np.max(self.qtable[y][x][0:6]) - old_q_value
            
        # update q-value 
        new_q_value = old_q_value + (ALPHA * temp_diff)
        self.qtable[y_old][x_old][last_action] = new_q_value
    
    
    
    # Store the q-table
    with open("my-saved-qtable.pt", "wb") as file:
        pickle.dump(self.qtable, file)
        
    if(bigtable_on == 1):
        self.crate2table = build_big_qtable2(self.qtable[1:16,2:16])
        self.crate3table = build_big_qtable3(self.qtable[1:16,2:16])
    
        with open("my-saved-crate2table.pt", "wb") as file:
            pickle.dump(self.crate2table, file)
        with open("my-saved-crate3table.pt", "wb") as file:
            pickle.dump(self.crate3table, file)
    

        


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
        e.WAITED: -200,
        BOMB_NEXT_TO_CRATE: 300,
        FALSE_BOMB: -3,
        #e.BOMB_DROPPED: 2,
        # invalid action is bad
        e.INVALID_ACTION: -1000
        }
    
    #print(events)
    reward_sum = 0
    
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


# for building crate2-table
def build_big_qtable2(small_qtable):
    small_qtable[0:15,0,3] = small_qtable[0:15,0,1]
    small_qtable[0,0:14,0] = small_qtable[0,0:14,2]
    # initialize big q table
    big_qtable = np.zeros((29,27,6))
    # copy small q table in bottom right corner
    big_qtable[14:29, 13:27,:] = small_qtable
    # flip and change right <-> left for botton left corner
    big_qtable[14:29, 0:13] = np.fliplr(small_qtable)[0:15,0:13, np.array([0,3,2,1,4,5])]
    # flip and change up <-> down for upper half
    big_qtable[0:14, 0:27] = np.flipud(big_qtable[14:29, 0:27])[0:14, 0:27, np.array([2,1,0,3,4,5])]
    return big_qtable

# for building crate3-table
def build_big_qtable3(small_qtable):
    small_qtable[0:15,0,3] = small_qtable[0:15,0,1]
    small_qtable[0,0:14,0] = small_qtable[0,0:14,2]
    # transpose small_qtable
    small_qtable = small_qtable.transpose(1, 0, 2)
    # change walking directions
    small_qtable = small_qtable[0:14, 0:15, np.array([3,2,1,0,4,5])]
    # initialize big q table
    big_qtable = np.zeros((27,29,6))
    # copy small q table in bottom right corner
    big_qtable[13:27, 14:29,:] = small_qtable
    # flip and change right <-> left for botton left corner
    big_qtable[13:27, 0:14,:] = np.fliplr(small_qtable)[0:14,0:14, np.array([0,3,2,1,4,5])]
    # flip and change up <-> down for upper half
    big_qtable[0:13, 0:29,:] = np.flipud(big_qtable[14:27, 0:29])[0:13, 0:29, np.array([2,1,0,3,4,5])]
    return big_qtable