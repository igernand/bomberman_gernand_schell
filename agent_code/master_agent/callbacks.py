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
    '''if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
        
        self.logger.info("Setting up q-table from scratch.")
        #weights = np.random.rand(len(ACTIONS))
        self.qtable = np.zeros((17, 17, 4))
        
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            
        self.logger.info("Loading q-table from saved state.")
        with open("my-saved-qtable.pt", "rb") as file:
            self.qtable = pickle.load(file)'''
            
    # ÜBERHAUPT LADEN? ODER LOKAL ANLEGEN?
    if not os.path.isfile("my-saved-qtable.pt"):
        self.qtable = np.zeros((15, 15, 6))
    
    else:
        with open("my-saved-bigqtable.pt", "rb") as file:
            self.qtable = pickle.load(file)
            
    if not os.path.isfile("my-saved-bigqtable.pt"):
        self.bigqtable = np.zeros((29, 29, 6))
    
    else:
        with open("my-saved-bigqtable.pt", "rb") as file:
            self.bigqtable = pickle.load(file)


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
    #qtable = np.zeros((17, 17, 6))
    #
    coins = game_state["coins"]
    #crates = ?
    #bombs = ?
    #explosions = ?
    #others = treat as crates or stonewalls
    for coin in coins:
        x, y = coin
        if x%2==1 and y%2==1: #cointable1: MOMENTAN bigqtable (TODO)
            # TODO: oder x und y vertauschen?
            self.qtable = self.bigqtable[(15-x):(30-x), (15-y):(30-y)]
        # TODO richtigen Ausschnitt finden
        #elif x%2==0 and y%2==1:
        #    self.qtable = coin2table[(14-x):(29-x), (14-y):(29-y)]
        #elif x%2==1 and y%2==0:
        #    self.qtable = coin3table[(14-x):(29-x), (14-y):(29-y)]
        
            
        
            
            
         
                
        #2. Richtung Mittelpunkt Werte von cointable1/2 übernehmen
        #3. auf andere Quadranten spiegeln
    #wiederholen für crates, bombs, explosions
    
    
    # todo Exploration vs exploitation
    random_prob = .7
    _, score, bombs_left, (x_self, y_self) = game_state["self"]
    #TODO train mode unnecessary
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        #return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return np.random.choice(ACTIONS, p=[.25, .25, .25, .25,.0,.0])
    
    self.logger.debug("Querying model for action.")
    ### TODO: take values from q-table and transform to probabilities -> argmax
    # return action with highest probability
    #return np.random.choice(ACTIONS, p=self.model)
    return ACTIONS[np.argmax(self.qtable[(x_self-1),(y_self-1)][0:4])]


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
