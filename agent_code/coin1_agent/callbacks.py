import os
import pickle
import random

import numpy as np
# coin1 agent

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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
        self.logger.info("Setting up q-table from scratch.")
        self.qtable = np.zeros((17, 17, 6))
    
    else:
        self.logger.info("Loading q-table from saved state.")
        with open("my-saved-qtable.pt", "rb") as file:
            self.qtable = pickle.load(file)
            
    #bigqtable
    bigtable_on = 1
    if(bigtable_on == 1):
        if not os.path.isfile("my-saved-coin1table.pt"):
            self.coin1table = np.zeros((29, 29, 6))
    
        else:
            with open("my-saved-coin1table.pt", "rb") as file:
                self.coin1table = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
 
    # todo Exploration vs exploitation
    random_prob = 0.7
    _, score, bombs_left, (x_self, y_self) = game_state["self"]
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 100%: walk in any direction. 0% wait. 0% bomb.
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25, 0, 0])
    
    self.logger.debug("Querying model for action.")
    # take values from q-table and transform to probabilities -> argmax
    # return action with highest probability
    return ACTIONS[np.argmax(self.qtable[y_self,x_self,0:4])]

