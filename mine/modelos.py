import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .mineTools import actFunc


class StandardModel:

    def __init__(self, nLayers: int, in_dimension: int, neurons: int,
                 activationFunc: str) -> None:
        # seed initialization
        # torch.cuda.manual_seed_all(0)
        # torch.manual_seed(0)
        # np.random.seed(0)
        # random.seed(0)

        input = in_dimension
        output = neurons
        layers = OrderedDict()

        for i in range(nLayers):

            # Add a fully-connected layer
            if i != 0:
                input = neurons
            if i == nLayers - 1:
                output = 1
            layers[f"linear_{i+1}"] = nn.Linear(input, output)

            # Add an activation function
            if i != nLayers - 1:
                layers[f"activation_{i+1}"] = actFunc(activationFunc)()

        self.T = nn.Sequential(layers)

    def get_model(self):
        return self.T
