import os
import pickle
import random

import numpy as np
#master agent

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
#ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

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

    #TODO table anlegen im ersten fall abstellen
            
    # load coin tables
    if not os.path.isfile("my-saved-coin1table.pt"):
        self.logger.info("Setting up coin1-table from scratch.")
        self.coin1table = np.zeros((29,29,6))
        self.coin1qtable = np.zeros((29, 29, 6))
    
    else:
        self.logger.info("Loading coin1-table from saved state.")
        with open("my-saved-coin1table.pt", "rb") as file:
            self.coin1table = pickle.load(file)
            
    if not os.path.isfile("my-saved-coin2table.pt"):
        self.logger.info("Setting up coin2-table from scratch.")
        self.coin2table = np.zeros((29, 27, 6))
    
    else:
        self.logger.info("Loading coin2-table from saved state.")
        with open("my-saved-coin2table.pt", "rb") as file:
            self.coin2table = pickle.load(file)
    
    if not os.path.isfile("my-saved-coin3table.pt"):
        self.logger.info("Setting up coin3-table from scratch.")
        self.coin3table = np.zeros((27, 29, 6))
    
    else:
        self.logger.info("Loading coin3-table from saved state.")
        with open("my-saved-coin3table.pt", "rb") as file:
            self.coin3table = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    #todo
    #open cratetable, cointable1, cointable2, bombtable, explosiontable
    
    coins = game_state["coins"]
    #crates = ?
    #bombs = ?
    #explosions = ?
    #others = treat as crates or stonewalls
    
    #TODO: maybe load old qtable and only change a few entries
    self.qtable = np.zeros((15, 15, 6))
    for coin in coins:
        x, y = coin
        if x%2==1 and y%2==1: 
            self.qtable = self.qtable + self.coin1table[(15-y):(30-y), (15-x):(30-x)]
            
        elif x%2==0 and y%2==1:
            self.qtable = self.qtable + self.coin2table[(15-y):(30-y), (14-x):(29-x)]
            
        elif x%2==1 and y%2==0:
            self.qtable = self.qtable + self.coin3table[(14-y):(29-y), (15-x):(30-x)]
        
            
            
    
    # Exploration vs exploitation
    random_prob = .7
    _, score, bombs_left, (x_self, y_self) = game_state["self"]
    #TODO train mode unnecessary
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25,.0,.0])
    
    self.logger.debug("Querying model for action.")
    # take values from q-table and transform to probabilities -> argmax
    # return action with highest probability
    return ACTIONS[np.argmax(self.qtable[(y_self-1),(x_self-1)][0:4])]

