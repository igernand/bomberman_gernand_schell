import os
import pickle
import random

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']

def setup(self):
    
    #qtable
    if not os.path.isfile("my-saved-timertable.pt"):
        self.timertable = np.zeros((17, 17, 6))
    
    else:
        with open("my-saved-timertable.pt", "rb") as file:
            self.timertable = pickle.load(file)
            

def act(self, game_state: dict) -> str:    
    #Exploration vs exploitation
    random_prob = 0.9
    _, score, bombs_left, (x_self, y_self) = game_state["self"]
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .02, .08])
        
    
    self.logger.debug("Querying model for action.")
  
    return ACTIONS[np.argmax(self.timertable[y_self,x_self,0:6])]


def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None
    return None