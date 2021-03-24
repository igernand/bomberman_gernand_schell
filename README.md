# bomberman_rl
FML Project. 

Reinforcement learning for Bomberman: Temporal Difference Q-learning with data augmentation


Note on the different agents:

coin1_agent trains collecting the coin at position (1,1)
coin2_agent trains collecting the coin at position (2,1) and indirectly at (1,2)
crate_agent trains placing bombs next to crate at (1,1)
crate2_agent trains placing bombs next to crate at (2,1) and indirectly at (1,2)
bomb_agent trains walking away from a bomb placed at (7,7)
bomb2_agent trains walking away from a bomb at (8,7) and indirectly at (7,8)
explosion_agent trains walking away from a bomb placed at (7,7)
explosion2_agent trains walking away from a bomb at (8,7) and indirectly at (7,8)

master_agent: all generated big-q-tables and q-tables of the above agents have to be handed to this agent. By data augmetation the given data can be used to generate a q-table holding all the information of the current game_state:
For example: If a coin is placed at (x,y) where x,y are odd, coin1_table gives the needed data for this coin by mirroring, rotating,.. entries of the (1,1)-q-table.
All coin, crate, bomb, explosion q-tables are then summed up and used for the decision taking.

user_agent, random_agent, rules_based_agent, tpl_agent and peaceful_agent are the predefined agents. 
