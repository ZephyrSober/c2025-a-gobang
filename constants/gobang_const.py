from typing import Final
from enum import Enum,IntEnum

WIDTH: Final = 15
HEIGHT: Final = 15
CHESS_NUM: Final = 3

class Chess(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    def __invert__(self):
        if self == Chess.BLACK:
            return Chess.WHITE
        elif self == Chess.WHITE:
            return Chess.BLACK
        return Chess.EMPTY

    @classmethod
    def from_onehot(cls, onehot):
        if onehot[Chess.BLACK] == 1:
            return Chess.BLACK
        elif onehot[Chess.WHITE] == 1:
            return Chess.WHITE
        return Chess.EMPTY