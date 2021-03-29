import os
import pickle
import random

import numpy as np
#master agent

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

    #TODO table anlegen im ersten fall abstellen
            
    # load coin tables
    if not os.path.isfile("my-saved-coin1table.pt"):
        self.logger.info("Setting up coin1-table from scratch.")
        self.coin1table = np.zeros((29,29,6))
    
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

    # load crate tables
    if not os.path.isfile("my-saved-crate1table.pt"):
        self.logger.info("Setting up crate1-table from scratch.")
        self.crate1table = np.zeros((29,29,6))
    
    else:
        self.logger.info("Loading crate1-table from saved state.")
        with open("my-saved-crate1table.pt", "rb") as file:
            self.crate1table = pickle.load(file)
            
    if not os.path.isfile("my-saved-crate2table.pt"):
        self.logger.info("Setting up crate2-table from scratch.")
        self.crate2table = np.zeros((29,27,6))
    
    else:
        self.logger.info("Loading crate2-table from saved state.")
        with open("my-saved-crate2table.pt", "rb") as file:
            self.crate2table = pickle.load(file)
            
    if not os.path.isfile("my-saved-crate3table.pt"):
        self.logger.info("Setting up crate3-table from scratch.")
        self.crate3table = np.zeros((27,29,6))
    
    else:
        self.logger.info("Loading crate3-table from saved state.")
        with open("my-saved-crate3table.pt", "rb") as file:
            self.crate3table = pickle.load(file)
     
    # load bomb and explosion tables
    if not os.path.isfile("my-saved-bombtable.pt"):
        self.logger.info("Setting up bomb-table from scratch.")
        self.bombtable = np.zeros((17,17,6))
    
    else:
        self.logger.info("Loading bombtable from saved state.")
        with open("my-saved-bombtable.pt", "rb") as file:
            self.bombtable = pickle.load(file)
            
    if not os.path.isfile("my-saved-explosiontable.pt"):
        self.logger.info("Setting up explosiontable from scratch.")
        self.explosiontable = np.zeros((17,17,6))
    
    else:
        self.logger.info("Loading explosiontable from saved state.")
        with open("my-saved-explosiontable.pt", "rb") as file:
            self.explosiontable = pickle.load(file)
        
            
    if not os.path.isfile("my-saved-bomb2table.pt"):
        self.logger.info("Setting up bomb2-table from scratch.")
        self.bomb2table = np.zeros((17,17,6))
    
    else:
        self.logger.info("Loading bomb2table from saved state.")
        with open("my-saved-bomb2table.pt", "rb") as file:
            self.bomb2table = pickle.load(file)
            
    if not os.path.isfile("my-saved-explosion2table.pt"):
        self.logger.info("Setting up explosion2table from scratch.")
        self.explosion2table = np.zeros((17,17,6))
    
    else:
        self.logger.info("Loading explosion2table from saved state.")
        with open("my-saved-explosion2table.pt", "rb") as file:
            self.explosion2table = pickle.load(file)
            
    if not os.path.isfile("my-saved-bomb3table.pt"):
        self.logger.info("Setting up bomb3-table from scratch.")
        self.bomb3table = np.zeros((17,17,6))
    
    else:
        self.logger.info("Loading bomb3table from saved state.")
        with open("my-saved-bomb3table.pt", "rb") as file:
            self.bomb3table = pickle.load(file)
            
    if not os.path.isfile("my-saved-explosion3table.pt"):
        self.logger.info("Setting up explosion1table from scratch.")
        self.explosion3table = np.zeros((17,17,6))
    
    else:
        self.logger.info("Loading explosion3table from saved state.")
        with open("my-saved-explosion3table.pt", "rb") as file:
            self.explosion3table = pickle.load(file)
            
    # timer
    if not os.path.isfile("my-saved-walking1table.pt"):
        self.logger.info("Setting up walking1table from scratch.")
        self.walking1table = np.zeros((17,17,6))
    
    else:
        self.logger.info("Loading walking1table from saved state.")
        with open("my-saved-walking1table.pt", "rb") as file:
            self.walking1table = pickle.load(file)
            
    if not os.path.isfile("my-saved-walking2table.pt"):
        self.logger.info("Setting up walking2table from scratch.")
        self.walking2table = np.zeros((17,17,6))
    
    else:
        self.logger.info("Loading walking2table from saved state.")
        with open("my-saved-walking2table.pt", "rb") as file:
            self.walking2table = pickle.load(file)
    

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

        
    coins = game_state["coins"]
    board = game_state["field"]
    crates_x, crates_y = np.where(board==1)
    bomb_info = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    explosions_x, explosions_y = np.where(explosion_map>0)
    _, score, bombs_left, (x_self, y_self) = game_state["self"]
    bomb_off = 1
    #others = treat as crates or stonewalls
    
    #TODO: maybe load old qtable and only change a few entries
    self.qtable = np.zeros([15,15,6])
    
    self.coinqtable = self.walking1table[1:16, 1:16]
    self.coinqtable[:,:,:]=-1000
    # add coin tables
    for coin in coins:
        x, y = coin
        if x%2==1 and y%2==1: 
            self.coinqtable =  np.fmax(self.coinqtable, self.coin1table[(15-y):(30-y), (15-x):(30-x)])
            
        elif x%2==0 and y%2==1:
            self.coinqtable = np.fmax(self.coinqtable, self.coin2table[(15-y):(30-y), (14-x):(29-x)])
            
        elif x%2==1 and y%2==0:
            self.coinqtable = np.fmax(self.coinqtable, self.coin3table[(14-y):(29-y), (15-x):(30-x)])
    
    self.qtable = self.qtable + self.coinqtable
    
    # add bomb tables
    for k in range(len(bomb_info)):
        (x,y), t = bomb_info[k]
        if x == x_self and y==y_self:
            bomb_off = 0
            x_bomb = x
            y_bomb = y
        if x%2==1 and y%2==1: 
            bombt = self.bombtable[1:16,1:16]
            lim1 = np.maximum(0,y-1-6)
            dist1 = y-1 - lim1
            lim2 = np.minimum(15, y-1+7) 
            dist2 = lim2-y+1
            lim3 = np.maximum(0, x-1-6)
            dist3 = x-1-lim3
            lim4 = np.minimum(15, x-1+7)
            dist4 = lim4-x+1
            
            self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] = self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] + bombt[(6-dist1):(6+dist2), (6-dist3):(6+dist4)]
                            
        elif x%2==0 and y%2==1:
            bombt = self.bomb2table[1:16,1:16]
            lim1 = np.maximum(0,y-1-6)
            dist1 = y-1 - lim1
            lim2 = np.minimum(15, y-1+7) 
            dist2 = lim2-y+1
            lim3 = np.maximum(0, x-1-6)
            dist3 = x-1-lim3
            lim4 = np.minimum(15, x-1+7)
            dist4 = lim4-x+1
            
            self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] = self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] + bombt[(6-dist1):(6+dist2), (7-dist3):(7+dist4)]
        
        elif x%2==1 and y%2==0:
            bombt = self.bomb3table[1:16,1:16]
            lim1 = np.maximum(0,y-1-6)
            dist1 = y-1 - lim1
            lim2 = np.minimum(15, y-1+7) 
            dist2 = lim2-y+1
            lim3 = np.maximum(0, x-1-6)
            dist3 = x-1-lim3
            lim4 = np.minimum(15, x-1+7)
            dist4 = lim4-x+1
            
            self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] = self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] + bombt[(7-dist1):(7+dist2), (6-dist3):(6+dist4)]
    
    
    # add crate tables
    self.crateqtable = self.walking1table[1:16, 1:16]
    self.crateqtable[:,:,:] = -1000
    for k in range(len(crates_x)):
        x = crates_x[k]
        y = crates_y[k]
        #if bomb_off == 1:
        if x%2==1 and y%2==1: 
            self.crateqtable = np.fmax(self.crateqtable, self.crate1table[(15-y):(30-y), (15-x):(30-x)])
        
        elif x%2==0 and y%2==1:
            self.crateqtable = np.fmax(self.crateqtable, self.crate2table[(15-y):(30-y), (14-x):(29-x)])
        
        elif x%2==1 and y%2==0:
            self.crateqtable = np.fmax(self.crateqtable, self.crate3table[(14-y):(29-y), (15-x):(30-x)])
    
    self.qtable = self.qtable + self.crateqtable
                        
     
    # add explosion tables 
    for k in range(len(explosions_x)):
        x = explosions_x[k]
        y = explosions_y[k]
        if x%2==1 and y%2==1: 
            explot = self.explosiontable[1:16,1:16]
            lim1 = np.maximum(0,y-1-6)
            dist1 = y-1 - lim1
            lim2 = np.minimum(15, y-1+7) 
            dist2 = lim2-y+1
            lim3 = np.maximum(0, x-1-6)
            dist3 = x-1-lim3
            lim4 = np.minimum(15, x-1+7)
            dist4 = lim4-x+1
            
            self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] = self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] + explot[(6-dist1):(6+dist2), (6-dist3):(6+dist4)]
                        
        elif x%2==0 and y%2==1:
            explot = self.explosion2table[1:16,1:16]
            lim1 = np.maximum(0,y-1-6)
            dist1 = y-1 - lim1
            lim2 = np.minimum(15, y-1+7) 
            dist2 = lim2-y+1
            lim3 = np.maximum(0, x-1-6)
            dist3 = x-1-lim3
            lim4 = np.minimum(15, x-1+7)
            dist4 = lim4-x+1
            
            self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] = self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] + explot[(6-dist1):(6+dist2), (7-dist3):(7+dist4)]
        
        elif x%2==1 and y%2==0:
            explot = self.explosion3table[1:16,1:16]
            lim1 = np.maximum(0,y-1-6)
            dist1 = y-1 - lim1
            lim2 = np.minimum(15, y-1+7) 
            dist2 = lim2-y+1
            lim3 = np.maximum(0, x-1-6)
            dist3 = x-1-lim3
            lim4 = np.minimum(15, x-1+7)
            dist4 = lim4-x+1
            
            self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] = self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] + explot[(7-dist1):(7+dist2), (6-dist3):(6+dist4)]
     
    
    # bomb timer
    if bombs_left ==0:
        self.qtable[:,:,4]=-10000
        #self.qtable[:,:,4] = 100*self.timertable[1:16,1:16,4]
        #self.qtable[:,:,4] = 100* self.walking2table[1:16, 1:16, 4]
        
    # stones boundary
    #self.qtable[0,0:15, 0]= self.qtable[0,0:15,0] -1000
    #self.qtable[0:15, 0, 3] = self.qtable[0:15,0,3] -1000
    #self.qtable[14, 0:15, 2] = self.qtable[14,0:15,2] -1000
    #self.qtable[0:15, 14, 1] = self.qtable[0:15,14,1] -1000

    
    
    # Exploration vs exploitation
    random_prob = .7
    _, score, bombs_left, (x_self, y_self) = game_state["self"]
    
    ####
   
    self.logger.debug("Querying model for action.")
    # take values from q-table and transform to probabilities -> argmax
    # return action with highest probability
    #print(ACTIONS[np.argmax(self.qtable[(y_self-1),(x_self-1)][0:6])])
    #print(self.qtable[(y_self-1),(x_self-1)])
    #print(self.coin1table[4,5])
    return ACTIONS[np.argmax(self.qtable[(y_self-1),(x_self-1)][0:6])]

def add_bombtable(x,y, self):
    if x%2==1 and y%2==1: 
        bombt = self.bombtable[1:16,1:16]
        lim1 = np.maximum(0,y-1-6)
        dist1 = y-1 - lim1
        lim2 = np.minimum(15, y-1+7) 
        dist2 = lim2-y+1
        lim3 = np.maximum(0, x-1-6)
        dist3 = x-1-lim3
        lim4 = np.minimum(15, x-1+7)
        dist4 = lim4-x+1
            
        self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] = self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] + bombt[(6-dist1):(6+dist2), (6-dist3):(6+dist4)]
        #return(self.qtable)
            
    elif x%2==0 and y%2==1:
        bombt = self.bomb2table[1:16,1:16]
        lim1 = np.maximum(0,y-1-6)
        dist1 = y-1 - lim1
        lim2 = np.minimum(15, y-1+7) 
        dist2 = lim2-y+1
        lim3 = np.maximum(0, x-1-6)
        dist3 = x-1-lim3
        lim4 = np.minimum(15, x-1+7)
        dist4 = lim4-x+1
        
        self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] = self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] + bombt[(6-dist1):(6+dist2), (7-dist3):(7+dist4)]
        #return(qtable)
            

                
    elif x%2==1 and y%2==0:
        bombt = self.bomb3table[1:16,1:16]
        lim1 = np.maximum(0,y-1-6)
        dist1 = y-1 - lim1
        lim2 = np.minimum(15, y-1+7) 
        dist2 = lim2-y+1
        lim3 = np.maximum(0, x-1-6)
        dist3 = x-1-lim3
        lim4 = np.minimum(15, x-1+7)
        dist4 = lim4-x+1
        
        self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] = self.qtable[(y-1-dist1):(y-1+dist2), (x-1-dist3):(x-1+dist4)] + bombt[(7-dist1):(7+dist2), (6-dist3):(6+dist4)]
        #return(qtable)
            