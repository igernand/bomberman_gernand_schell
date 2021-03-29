import os
import pickle
import random

import numpy as np
# bomb2 agent


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']

def setup(self):
            
    if not os.path.isfile("my-saved-bomb2table.pt"):
        self.bomb2table = np.zeros((17, 17, 6))
    
    else:
        with open("my-saved-bomb2table.pt", "rb") as file:
            self.bomb2table = pickle.load(file)
            
    if not os.path.isfile("my-saved-bomb3table.pt"):
        self.bomb3table = np.zeros((17, 17, 6))
    
    else:
        with open("my-saved-bomb3table.pt", "rb") as file:
            self.bomb3table = pickle.load(file)
            


def act(self, game_state: dict) -> str:
    
    # Exploration vs exploitation
    random_prob = 1
    _, score, bombs_left, (x_self, y_self) = game_state["self"]
    safe_spots=[(5,6),(7,6),(9,6),(11,6),(4,7),(12,7),(5,8),(7,8),(9,8),(11,8)]
    end_spots=[(5,5),(7,5),(9,5),(11,5),(3,7),(13,7),(5,9),(7,9),(9,9),(11,9)]
    if (x_self,y_self) in end_spots:
        return 'BOMB'
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, 0, .1])
        
    
    self.logger.debug("Querying model for action.")
    output = np.argmax(self.bomb2table[y_self,x_self,[0,1,2,3,5]])
    if output == 4:
        output = 5
    return ACTIONS[output]


def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None
    return None
