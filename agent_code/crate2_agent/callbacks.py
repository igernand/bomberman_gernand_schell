import os
import pickle
import random

import numpy as np
#crate agent

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']

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
    
    #qtable
    if not os.path.isfile("my-saved-qtable.pt"):
        self.qtable = np.zeros((17, 17, 6))
    
    else:
        with open("my-saved-qtable.pt", "rb") as file:
            self.qtable = pickle.load(file)
            
    #bigqtable
    bigtable_on = 1
    if(bigtable_on == 1):
        if not os.path.isfile("crate2table.pt"):
            self.logger.info("Setting up crate2-table from scratch.")
            self.crate2table = np.zeros((29, 27, 6))
    
        else:
            self.logger.info("Loading crate2-table from saved state.")
            with open("my-saved-crate2table.pt", "rb") as file:
                self.crate2table = pickle.load(file)
            
        if not os.path.isfile("crate3table.pt"):
            self.logger.info("Setting up crate3-table from scratch.")
            self.crate3table = np.zeros((27, 29, 6))
    
        else:
            self.logger.info("Loading crate3-table from saved state.")
            with open("my-saved-crate3table.pt", "rb") as file:
                self.crate3table = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    
    # todo Exploration vs exploitation
    random_prob = .7
    _, score, bombs_left, (x_self, y_self) = game_state["self"]
    board = game_state["field"]
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.24, .24, .24, .24, .04, 0])
    
    self.logger.debug("Querying model for action.")
    # take values from q-table and transform to probabilities -> argmax
    # return action with highest probability
    return ACTIONS[np.argmax(self.qtable[y_self,x_self,0:5])]

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

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
