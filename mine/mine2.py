import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import time

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Mine2:

    def __init__(self, hiddenLayers, neurons, lr, minibatches, cuda=None):
        self.neurons = neurons
        self.hiddenLayers = hiddenLayers
        self.minibatches = minibatches
        if cuda is not None:
            self.cuda = cuda
        else:
            self.cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = OrderedDict()
        for i in range(hiddenLayers+1):
            if i == 0:
                model["first layer"] = nn.Linear(2, neurons)
            elif i == hiddenLayers:
                model["last layer"] = nn.Linear(neurons, 1)
            else:
                model["middle layer"] = nn.Linear(neurons, neurons)
            model["activation func"] = nn.ReLU()
        self.model = nn.Sequential(model).to(self.cuda)

        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.entrenamiento = []

    def get_minibatches(self):
        return self.minibatches

    def _estimate_mi(self, entry1: torch.tensor, entry2: torch.tensor):
        minibatch_size = torch.tensor(len(entry1))
        return torch.mean(entry1) - torch.logsumexp(entry2, 0) + torch.log(minibatch_size)

    def single_iteration(self, x_minibatch: np.array, z_minibatch: np.array) -> torch.tensor:
        minibatch_size = len(x_minibatch)
        # First input
        net_input_1 = torch.cat((x_minibatch, z_minibatch), dim=1)
        out1 = self.model(net_input_1)
        # Now the permuted input
        net_input_2 = torch.cat((x_minibatch, z_minibatch[torch.randperm(minibatch_size)]), dim=1)
        out2 = self.model(net_input_2)
        return out1, out2

    def advance_epoch(self, x_batch:torch.tensor, z_batch:torch.tensor, shuffle=True):
        assert len(x_batch) == len(z_batch), "Sizes of input vectors are not the same"
        batch_size = len(x_batch)
        estimacion = None

        if shuffle:
            shuffle_idxs = torch.randperm(batch_size)
            x_batch = x_batch[shuffle_idxs]
            z_batch = z_batch[shuffle_idxs]

        # set model to training mode
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            for i in range(self.minibatches):
                begin = int(i*batch_size/self.minibatches)
                end = int((i+1)*batch_size/self.minibatches)
                out1, out2 = self.single_iteration(x_batch[begin:end], z_batch[begin:end])
                estimacion = self._estimate_mi(out1, out2)
                # self.estimaciones.append(estimacion.item())
                loss = -estimacion
                # self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.entrenamiento.append(estimacion.item())

    def run_epochs(self, x, z, n=1000, viewProgress=True):
        if viewProgress:
            print("Training network...")
        for epoch in tqdm(range(n), disable=not viewProgress):
            self.advance_epoch(x, z, shuffle=True)
        if viewProgress:
            print("Done!")
        return True  # output as a parallelization flag

    def estimate_mi(self, x_batch:torch.tensor, z_batch:torch.tensor):
        assert len(x_batch) == len(z_batch), "Sizes of input vectors are not the same"
        with torch.no_grad():
            out1, out2 = self.single_iteration(x_batch, z_batch)
            estimate = self._estimate_mi(out1, out2).item()
        return estimate

    def plot_training(self, true_mi=None, save=False):
        plt.plot(self.entrenamiento)
        if true_mi is not None:
            plt.axhline(y=true_mi, color='r', linestyle='-')
        plt.ylabel('informacion mutua estimada')
        plt.xlabel('epoca')
        plt.title(f"Progreso de MINE")
        plt.grid()
        if save:
            plt.savefig(f'mine{self.hiddenLayers}-{self.neurons}-{self.lr}-{self.minibatches}.pdf')
        plt.show()


if __name__ == "__main__":
    # media
    mu = np.array([0, 0])
    # covarianza
    rho = 0.5
    cov_matrix = np.array([[1, rho], [rho, 1]])
    # Genero la señal
    samples = 5000
    joint_samples_train = np.random.multivariate_normal(mean=mu, cov=cov_matrix, size=(samples, 1))
    X_samples = joint_samples_train[:, :, 0]
    Z_samples = joint_samples_train[:, :, 1]
    # Convert to tensors
    x = torch.from_numpy(X_samples).float().to(device='cpu')
    z = torch.from_numpy(Z_samples).float().to(device='cpu')
    # Entropia mediante formula
    true_mi = -0.5 * np.log(np.linalg.det(cov_matrix))
    print(f"The real mutual information is {true_mi}")

    tic = time.time()
    MINE = Mine2(1, 10, 0.5, 1)
    MINE.run_epochs(x, z, 1000, viewProgress=False)
    # MINE.plot_training(true_mi)
    # plt.plot([MINE.estimate_mi(x, z) for i in range(1000)])

    MINE2 = Mine2(1, 20, 0.5, 1)
    MINE2.run_epochs(x, z, 5000, viewProgress=False)
    # MINE2.plot_training(true_mi)
    # plt.plot([MINE2.estimate_mi(x, z) for i in range(1000)])

    MINE3 = Mine2(1, 30, 0.5, 1)
    MINE3.run_epochs(x, z, 5000, viewProgress=False)
    toc = time.time()

    print(toc - tic)

    # plt.axhline(y=true_mi, color='r', linestyle='-')
