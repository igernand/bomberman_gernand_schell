import os
import pickle
import random

import numpy as np
# bomb agent


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']

def setup(self):
            
    if not os.path.isfile("my-saved-bombtable.pt"):
        self.bombtable = np.zeros((17, 17, 6))
    
    else:
        with open("my-saved-bombtable.pt", "rb") as file:
            self.bombtable = pickle.load(file)
            
    #bigqtable
    bigtable_on = 0
    if(bigtable_on == 1):
        if not os.path.isfile("my-saved-crate1table.pt"):
            self.crate1table = np.zeros((29, 29, 6))
    
        else:
            with open("my-saved-crate1table.pt", "rb") as file:
                self.crate1table = pickle.load(file)


def act(self, game_state: dict) -> str:
    # Exploration vs exploitation
    random_prob = 1
    _, score, bombs_left, (x_self, y_self) = game_state["self"]
    safe_spots=[(7,2),(6,3),(8,3),(6,5),(8,5),(3,6),(5,6),(9,6),(11,6),(2,7),(12,7),(3,8),(5,8),(9,8),(11,8),(6,9),(8,9),(6,11),(8,11),(7,12)]
    end_spots=[(2,7),(3,6),(3,8),(5,5),(5,9),(6,3),(6,11),(7,2),(7,12),(8,3),(8,11),(9,5),(9,9),(11,6),(11,8),(12,7)]
    if (x_self,y_self) in end_spots:
        return 'BOMB'
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, 0, .1])
        
    
    self.logger.debug("Querying model for action.")
    output = np.argmax(self.bombtable[y_self,x_self,[0,1,2,3,5]])
    if output == 4:
        output = 5
    return ACTIONS[output]


def state_to_features(game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    return None
