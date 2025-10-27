import torch
from constants import *

def set_state(state:torch.Tensor,x:int,y:int,target:Chess) ->None:
    state[x,y,:] = 0
    state[x,y,target] = 1

def create_test_state():
    # create empty state
    state = torch.zeros([WIDTH, HEIGHT, CHESS_NUM])
    state[:, :, Chess.EMPTY] = 1
    # create latest action
    latest_action = [7, 7]
    set_state(state,latest_action[0],latest_action[1],Chess.BLACK)
    return state
