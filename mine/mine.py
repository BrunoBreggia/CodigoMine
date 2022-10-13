import torch
import torch.nn as nn
# from collections import OrderedDict
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from mineTools import generate_batches, actFunc

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

class Mine(nn.Module):

    def __init__(self, model=None, cuda=None):
        """
        Inicializacion de la red neuronal con los 
        hiperparametros brindados.

        Parametros:
            nLayers (int) Cantidad de capas de la red
            in_dimension (int) Cantidad de datos del vector
                de entrada a la red
            neurons (int)  Cantidad de neuronas por capa
        """
        super().__init__()
        self.cuda = cuda
        if self.cuda is None:
            self.cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if model is None:
            pass
        self.T = model.get_model().to(self.cuda)

    def forward(self, x, z):
        """
        Calcula estimacion de informacion mutua
        (limite inferior), segun la representacion de
        Donsker-Varadhan de la Divergencia de
        Kullback-Leibler.
        """

        batch_size = x.shape[0]  # cantidad de muestras (tensor de 1 elemento)
        z_hat = z[torch.randperm(batch_size)]  # permuto z

        # Ingresan al modelo los pares x,z
        eval = self.T(torch.cat((x, z), dim=1))
        primer_termino = eval.mean()

        # Ingresan al modelo los pares x, z permutado
        eval_hat = self.T(torch.cat((x, z_hat), dim=1))
        segundo_termino = torch.logsumexp(eval_hat, 0) - np.log(batch_size)

        # Evaluo limite inferior de informacion mutua
        nu = primer_termino - segundo_termino

        # invertido de signo para poder maximizar con algoritmos convencionales de minimizacion
        return -nu

    def train_model(self, X, Z, optimizer, batch_size=16, epochs=None, disableTqdm=False):
        """
        Entrena el modelo suministrado con batches formados 
        a partir del total de pares de muestras (x,z)
        suministradas

        Entrega un vector con los datos de informacion mutua
        estimada por cada epoca.
        """ 

        # torch.cuda.manual_seed_all(0)
        # torch.manual_seed(0)
        # np.random.seed(0)  # always using same numpy seed
        # random.seed(0)

        if epochs is None:
            epochs = 100

        if type(epochs) is int:
            estimaciones = np.zeros(epochs)
            epochs = [epochs]
        else:
            estimaciones = np.zeros(epochs[-1])

        output_train = []
        output_test = []

        # Each epoch feeds the model 
        for epoch in tqdm(range(epochs[-1]), disable=disableTqdm):

            # set model to training mode
            self.train()

            # iterate over data
            for x, z in generate_batches(X, Z, batch_size):

                # sample to GPU
                x = x.to(device=self.cuda)
                z = z.to(device=self.cuda)

                # zero out all the gradients (clean the buffer)
                optimizer.zero_grad()

                # enable gradient calculation
                with torch.set_grad_enabled(True):
                    loss = self(x, z)
                    loss.backward()
                    optimizer.step()

            # estimaciones[epoch] = -loss.item()
            if epoch+1 in epochs:
                output_train.append(-loss.item())
                output_test.append(self.evaluate_model(X, Z))

        # devuelve arreglo con la evolucion del calculo de IM
        # en cada una de las epocas solicitadas
        return output_train, output_test

    def evaluate_model(self, X, Z):
        """
        Función de evaluación. Evalúa la salida del modelo 
        suministrado con las entradas dadas. Entrega la salida
        del modelo.
        """

        # set model to testing mode
        self.eval()

        assert len(X) == len(Z), "Input and target data must contain same number of elements"
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(Z, np.ndarray): 
            Z = torch.from_numpy(Z).float()

        X = X.to(device=self.cuda)
        Z = Z.to(device=self.cuda)

        # disable gradient calculation
        with torch.no_grad():
            mutualInfo = -self(X, Z)

        return mutualInfo.cpu().item()

 



