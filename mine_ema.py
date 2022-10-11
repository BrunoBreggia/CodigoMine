import torch
# import torch.nn as nn
from Implementacion_de_MINE.mine import Mine
import math
# from collections import OrderedDict
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import random
# from Implementacion_de_MINE.mineTools import generate_batches, actFunc

# Implementacion de red neuronal multicapas 
# para estimacion de red neuronal profunda

"""
Modelo para señales unidimensionales

Hiperparametros del modelo:
    * Cantidad de capas (capas de perceptrones simples)
    * Cantidad de neuronas por capa
    * Dimension de datos de entrada
  [ * Funcion de activacion  ]

Otros hiperparametros:
    * Tamaño del batch

"""

class MineEMA(Mine):

    def __init__(self, model=None, alpha=0.1, cuda=None):
        super().__init__(model, cuda)
        self.running_mean = 0
        self.alpha = alpha

    def forward(self, x, z):
        """
        Calcula estimacion de informacion mutua (limite inferior), segun la representacion de
        Donsker-Varadhan de la Divergencia de Kullback-Leibler.
        Usa exponential moving average para el calculo backward.
        """

        batch_size = x.shape[0]  # cantidad de muestras (tensor de 1 elemento)
        z_hat = z[torch.randperm(batch_size)]  # permuto z

        # Ingresan al modelo los pares x,z
        eval = self.T(torch.cat((x, z), dim=1))
        primer_termino = eval.mean()

        # Ingresan al modelo los pares x, z permutado
        eval_hat = self.T(torch.cat((x, z_hat), dim=1))

        # Calculo de EMA
        t_exp = torch.exp(torch.logsumexp(eval_hat, 0) - math.log(eval_hat.shape[0])).detach()
        if self.running_mean == 0:
            self.running_mean = t_exp
        else:
            self.running_mean = self.alpha * t_exp + (1 - self.alpha) * self.running_mean
        segundo_termino = MineEMA.EMALoss.apply(eval_hat, self.running_mean)

        # Evaluo limite inferior de informacion mutua
        nu = primer_termino - segundo_termino

        # invertido de signo para poder maximizar con algoritmos convencionales de minimizacion
        return -nu

    class EMALoss(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor, running_ema):
            # ctx is a context object that can be used to stash information
            # for backward computation
            ctx.tensor = tensor
            ctx.running_ema = running_ema

            return tensor.exp().mean().log()

        @staticmethod
        def backward(ctx, grad_output):
            grad = grad_output * ctx.tensor.exp().detach() / (ctx.running_ema + 1e-6) / ctx.tensor.shape[0]
            # We return as many input gradients as there were arguments.
            # Gradients of non-Tensor arguments to forward must be None.
            return grad, None



