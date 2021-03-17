import os
import pickle
import random

import numpy as np


#ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        
        self.logger.info("Setting up q-table from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        self.qtable = np.zeros((17,17,4))
        
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            
        self.logger.info("Loading q-table from saved state.")
        with open("my-saved-qtable.pt", "rb") as file:
            self.qtable = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .2
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25])
    
    self.logger.debug("Querying model for action.")
    
    ### TODO: take values from q-table and transform to probabilities -> argmax
    # return action with highest probability
    # return np.argmax(q_values[current_row_index, current_column_index])
    # features = state_to_features(game_state)
    # get current q-values for action
    #q_val_actions = self.qtable[features]
    _, score, bombs_left, (x, y) = game_state['self']
    q_val_actions = self.qtable[x,y]
    # determine maximal q-value indices
    max_actions = np.argwhere(q_val_actions==np.max(q_val_actions))
    # choose random action out of the list of actions with max q-value 
    #(if multiple exist, otherwise return action with maximal q-value)
    return ACTIONS[random.choice(max_actions)[0]]
    
    
    #return ACTIONS[np.argmax(self.qtable[x_self, y_self])]
    #return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    _, score, bombs_left, (x, y) = game_state['self']
    arena = game_state['field']
    coins = game_state['coins']
     
    # find occupied fields in neighbourhood 
    # 0: free
    # 1: crate
    # -1: stone wall
    neighbour_fields = np.zeros(4)
    neighbour_fields[0] = arena[x,y+1] #UP
    neighbour_fields[1] = arena[x+1,y] #RIGHT
    neighbour_fields[2] = arena[x,y-1] #DOWN
    neighbour_fields[3] = arena[x-1,y] #LEFT
     
    # find closest coin (if all actions were valid)
    distances = np.zeros(len(coins))
    for i in range(len(coins)):
        c = coins[i]
        distances[i] = np.abs(c[0]-x) + np.abs(c[1]-y)
    closest_coin = coins[np.argmin(distances)]
    coin_position = np.zeros(2)
    coin_position[0] = x-closest_coin[0] # coin x coordinate relative to own position
    coin_position[1] = y-closest_coin[1] # coin y coordinate relative to own position
    
    return np.array([x,y,neighbour_fields[0],neighbour_fields[1],neighbour_fields[2],neighbour_fields[3],coin_position[0],coin_position[1]])

"""
    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)"""
