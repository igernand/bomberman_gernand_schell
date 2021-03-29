import os
import pickle
import random

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']


def setup(self):
    # load qtable
    if not os.path.isfile("my-saved-qtable.pt"):
        self.logger.info("Setting up q-table from scratch.")
        self.qtable = np.zeros((17, 17, 6))
    
    else:
        self.logger.info("Loading q-table from saved state.")
        with open("my-saved-qtable.pt", "rb") as file:
            self.qtable = pickle.load(file)
            
    # load bigqtables
    bigtable_on = 1
    
    if(bigtable_on == 1):
        if not os.path.isfile("coin2table.pt"):
            self.logger.info("Setting up coin2-table from scratch.")
            self.coin2table = np.zeros((29, 27, 6))
    
        else:
            self.logger.info("Loading coin2-table from saved state.")
            with open("my-saved-coin2table.pt", "rb") as file:
                self.coin2table = pickle.load(file)
            
        if not os.path.isfile("coin3table.pt"):
            self.logger.info("Setting up coin3-table from scratch.")
            self.coin3table = np.zeros((27, 29, 6))
    
        else:
            self.logger.info("Loading coin3-table from saved state.")
            with open("my-saved-coin3table.pt", "rb") as file:
                self.coin3table = pickle.load(file)


def act(self, game_state: dict) -> str:
    # todo Exploration vs exploitation
    random_prob = .7
    _, score, bombs_left, (x_self, y_self) = game_state["self"]
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 22,5%: walk in any direction. 2% bomb, 8% wait.
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .02, .08])
    
    self.logger.debug("Querying model for action.")
    # return action with highest q-value
    return ACTIONS[np.argmax(self.qtable[y_self,x_self,0:6])]

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None
    return None