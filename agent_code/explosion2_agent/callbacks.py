import os
import pickle
import random

import numpy as np
#explosion 2agent

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']

def setup(self):
    #qtable
            
    if not os.path.isfile("my-saved-explosion2table.pt"):
        self.explosion2table = np.zeros((17, 17, 6))
    
    else:
        with open("my-saved-explosion2table.pt", "rb") as file:
            self.explosion2table = pickle.load(file)
            
    if not os.path.isfile("my-saved-explosion3table.pt"):
        self.explosion3table = np.zeros((17, 17, 6))
    
    else:
        with open("my-saved-explosion3table.pt", "rb") as file:
            self.explosion3table = pickle.load(file)


def act(self, game_state: dict) -> str:
    
    # todo Exploration vs exploitation
    random_prob = .7
    _, score, bombs_left, (x_self, y_self) = game_state["self"]
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .05, .05])
        
    
    self.logger.debug("Querying model for action.")
  
    return ACTIONS[np.argmax(self.explosion2table[y_self,x_self,[0,1,2,3,4,5]])]


def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None
    return None
