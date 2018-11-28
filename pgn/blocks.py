import torch.nn as nn

class Block(nn.module):
    def __init__(self):
        self._projectors = {}
        self._aggregators = {}
        self._updaters = {}

class GlobalBlock(Block):
    pass

class NodeBlock(Block):
    pass

class EdgeBlock(Block):
    pass
