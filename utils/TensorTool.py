import torch
from constants import *

def set_state(state:torch.Tensor,x:int,y:int,target:Chess) ->None:
    state[x,y,:] = 0
    state[x,y,target] = 1

def create_test_state():
    # create empty state
    state = torch.zeros([BOARD_SIZE, BOARD_SIZE, CHESS_NUM])
    state[:, :, Chess.EMPTY] = 1
    return state
