import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import time

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def generate_minibatches(trainig_set: torch.tensor, size: int):
    """
    Function that returns minibatches from an original trainign dataset,
    of solicited size.

    Parameters
    ----------
    trainig_set: torch.tensor
        The complete dataset used for training, to be splitted into minibatches.
    size: int
        The size of the individual minibatches. If length of dataset is not
        a multiple of the size, then last minibatch may contain less data.

    Return
    ------
    minibatches: torch.tensor
        A tensor of minibatches of demanded size.
    """
    # CHECKOUT Verify if this dimension to split is OK
    minibatches = trainig_set.split(size, dim=1)
    return minibatches


class Mine2(nn.Module):
    """
    Modelo de red neuronal para estimar información mutua

    Modelo basado en el paper de Belghazi para la estimación de informacion
    mutua de señales estocásticas a partir de redes neuronales. Las siglas
    MINE vienen de "Mutual Information Neural Estimator"

    Utiliza la capacidad de minimización del error por backpropagation para
    maximizar la estimación de información mutua, basándose en la representación
    de Donsker-Varadhan de la informacion mutua, con la cual se demuestra que
    la misma posee una cota superior al estimarse con una red neuronal.
    """

    def __init__(self, hidden_layers: int, neurons: int, cuda: str=None):
        """
        Parameters
        ----------
        hidden_layers: int
            Amount of hidden layers for the network. The input layer will
            be of 2 dimensions (for the 2 input signals), and the output will
            be of 1 dimension. Minimum value permitted: 1.
        neurons: int
            Amount of neurons per hidden layer.
        cuda: str
            Specify where the model should run, on cpu or in graphic processor.
            If None, it will run on gpu if found, else on cpu.
        """
        super().__init__()

        # Set inner instance attributes
        self.neurons = neurons
        self.hiddenLayers = hidden_layers
        # self.minibatches = minibatches

        # Set cuda
        if cuda is not None:
            self.cuda = cuda
        else:
            self.cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create model
        model = OrderedDict()
        for i in range(hidden_layers + 1):
            if i == 0:
                model["first layer"] = nn.Linear(2, neurons)
            elif i != hidden_layers:
                model["middle layer"] = nn.Linear(neurons, neurons)
            else:
                model["last layer"] = nn.Linear(neurons, 1)
            model["activation func"] = nn.ReLU()
        self.model = nn.Sequential(model).to(self.cuda)

        # lists for data collection during training
        self.validation_progress = []
        self.training_progress = []

    def forward(self, input: torch.tensor):
        """
        Defines the forward pass of the network.

        It passes the input data through the layers of the model defined in the initializer.
        Overload of forward method is compulsory for models that inherit from nn.Module.

        Parameters
        ----------
        input: torch.tensor
            Batch of input data to pass into the model

        Returns
        -------
        output: torch.tensor
            Tensor with the output of the model
        """

        output = input
        for layer in self.model:
            output = layer(output)
        return output

    def training_step(self, train_dataset: torch.tensor):
        """
        Training step of the MINE neural network.

        The training step returns the loss, to be minimized.
        In this context, the loss is the opposite of the mutual information estimation,
        since we want to maximize the estimation.

        Parameters
        ----------
        train_dataset: torch.tensor
            Input to the model. It consists of the concatenation of the
            two signals to analyze.

        Returns
        -------
        loss: torch.tensor
            Tensor of one dimension which contains the loss of the network.
            The loss is actually the negative of the value we really want
            to maximize, which is the mutual information estimation.
        """

        # Obtain size of minibatch
        minibatch_size = train_dataset.shape[0]

        # First input (x and intact z)
        # net_input_1 = torch.cat((x_minibatch, z_minibatch), dim=1)
        out1 = self(train_dataset)

        # Second input (x and permutted z) -> to break correlation between both signals
        # FIXME: Possible bug at torch.randperm argument, possibly needs int not tensor
        permutation = torch.randperm(minibatch_size)
        permuted_z = train_dataset[:, 1][permutation]
        permuted_input = torch.cat((train_dataset[:, 0], permuted_z), dim=1)
        out2 = self(permuted_input)

        # Mutual information estimation
        mi_estimation = torch.mean(out1) - torch.logsumexp(out2, 0) + torch.log(minibatch_size)

        # Return the loss: the loss is the opposite of the mutual info estimation
        loss = -mi_estimation
        return loss

    def evaluate(self, input_dataset: torch.tensor):
        """
        Evaluates the model output with respect to given input, without
        performing backpropagation and with no gradiente calculation.

        Parameters
        ----------
        input_dataset: torch.tensor
            Input data to the model.

        Return
        ------
        mi_estimation: torch.tensor
            Output of the model, i.e the mutual information estimation.
        """

        with torch.no_grad():

            # Obtain size of minibatch
            batch_size = input_dataset.shape[0]

            # First input (x and intact z)
            # net_input_1 = torch.cat((x_batch, z_batch), dim=1)
            out1 = self(input_dataset)

            # Second input (x and permutted z) -> to break correlation between both signals
            # FIXME: Possible bug at torch.randperm argument, possibly needs int not tensor
            permutation = torch.randperm(batch_size)
            permuted_z = train_dataset[:, 1][permutation]
            permuted_input = torch.cat((train_dataset[:, 0], permuted_z), dim=1)
            # net_input_2 = torch.cat((x_batch, z_batch[torch.randperm(batch_size)]), dim=1)
            out2 = self(permuted_input)

            # Mutual information estimation
            mi_estimation = torch.mean(out1) - torch.logsumexp(out2, 0) + torch.log(batch_size)

        return mi_estimation

    def fit(self, train_loader, val_loader, num_epochs, learning_rate=1e-4):
        """
        Trains the MINE model using the provided training data loader and
        validates it using the provided validation data loader.

        We expect the training and validation datasets to be fractions of the original paired
        signals. For example, 80-20%, i.e. the same pair of signals is split
        into two parts, a bigger one for training and a smaller one for validation.

        Parameters
        ----------
        train_loader: torch.tensor
            A PyTorch DataLoader object representing the training data.
        val_loader: torch.tensor
            A PyTorch DataLoader object representing the validation data.
        num_epochs: int
            An integer specifying the number of epochs to train the model for.
        learning_rate: float
            A float specifying the learning rate to use for the optimizer.

        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        # self.validation_progress.clear()

        for epoch in range(num_epochs):

            # Training loop
            for batch in train_loader:
                loss = self.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Validation
            # with torch.no_grad():
            result_train = self.evaluate(train_loader)
            self.training_progress.append(result_train.item())
            result_val = self.evaluate(val_loader)
            self.validation_progress.append(result_val.item())

            # TODO Process the signal in validation_progress and cut the
            #  training epochs when this signal is decreasing

    def plot_training(self, true_mi: int = None, save=False):
        plt.plot(self.training_progress, color='b')
        plt.plot(self.validation_progress, color='g')
        if true_mi is not None:
            plt.axhline(y=true_mi, color='r', linestyle='-')
        plt.ylabel('Mutual information estimation')
        plt.xlabel('epoch')
        plt.title(f"MINE progress")
        plt.grid()
        if save:
            plt.savefig(f'mine{self.hiddenLayers}-{self.neurons}-{self.lr}-{self.minibatches}.pdf')
        plt.show()

    # def _estimate_mi(self, entry1: torch.tensor, entry2: torch.tensor):
    #     minibatch_size = torch.tensor(len(entry1))
    #     return torch.mean(entry1) - torch.logsumexp(entry2, 0) + torch.log(minibatch_size)

    # def single_iteration(self, x_minibatch: np.array, z_minibatch: np.array) -> torch.tensor:
    #     minibatch_size = len(x_minibatch)
    #     # First input
    #     net_input_1 = torch.cat((x_minibatch, z_minibatch), dim=1)
    #     out1 = self.model(net_input_1)
    #     # Now the permuted input
    #     net_input_2 = torch.cat((x_minibatch, z_minibatch[torch.randperm(minibatch_size)]), dim=1)
    #     out2 = self.model(net_input_2)
    #     return out1, out2

    # def advance_epoch(self, x_batch: torch.tensor, z_batch: torch.tensor, shuffle=True):
    #     assert len(x_batch) == len(z_batch), "Sizes of input vectors are not the same"
    #     batch_size = len(x_batch)
    #     estimacion = None
    #
    #     if shuffle:
    #         shuffle_idxs = torch.randperm(batch_size)
    #         x_batch = x_batch[shuffle_idxs]
    #         z_batch = z_batch[shuffle_idxs]
    #
    #     # set model to training mode
    #     self.optimizer.zero_grad()
    #     with torch.set_grad_enabled(True):
    #         for i in range(self.minibatches):
    #             begin = int(i*batch_size/self.minibatches)
    #             end = int((i+1)*batch_size/self.minibatches)
    #
    #             x_minibatch = x_batch[begin:end]
    #             z_minibatch = z_batch[begin:end]
    #             minibatch_size = len(x_minibatch)
    #             # First input
    #             net_input_1 = torch.cat((x_minibatch, z_minibatch), dim=1)
    #             out1 = self.model(net_input_1)
    #             # Now the permuted input
    #             net_input_2 = torch.cat((x_minibatch, z_minibatch[torch.randperm(minibatch_size)]), dim=1)
    #             out2 = self.model(net_input_2)
    #
    #             estimacion = self._estimate_mi(out1, out2)
    #             # self.estimaciones.append(estimacion.item())
    #             loss = -estimacion
    #             # self.model.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #         # self.entrenamiento.append(estimacion.item())
    #
    # def run_epochs(self, x, z, n=1000, viewProgress=True):
    #     if viewProgress:
    #         print("Training network...")
    #     for epoch in tqdm(range(n), disable=not viewProgress):
    #         self.advance_epoch(x, z, shuffle=True)
    #     if viewProgress:
    #         print("Done!")
    #     return True  # output as a parallelization flag
    #
    # def estimate_mi(self, x_batch:torch.tensor, z_batch:torch.tensor):
    #     assert len(x_batch) == len(z_batch), "Sizes of input vectors are not the same"
    #     with torch.no_grad():
    #         out1, out2 = self.single_iteration(x_batch, z_batch)
    #         estimate = self._estimate_mi(out1, out2).item()
    #     return estimate

    # def plot_training(self, true_mi=None, save=False):
    #     plt.plot(self.entrenamiento)
    #     if true_mi is not None:
    #         plt.axhline(y=true_mi, color='r', linestyle='-')
    #     plt.ylabel('informacion mutua estimada')
    #     plt.xlabel('epoca')
    #     plt.title(f"Progreso de MINE")
    #     plt.grid()
    #     if save:
    #         plt.savefig(f'mine{self.hiddenLayers}-{self.neurons}-{self.lr}-{self.minibatches}.pdf')
    #     plt.show()


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
    MINE = Mine2(1, 10)

    input_dataset = torch.cat((x, z), dim=1)
    TRAIN_PERCENT = 80  # 80 percent of input dataset is saved for training, remaining for validation
    train_size = int(len(input_dataset)*TRAIN_PERCENT/100)
    val_size = len(input_dataset) - train_size
    # CHECKOUT Verify if this dimension to split is OK
    train_dataset, val_dataset = input_dataset.split([train_size, val_size], dim=1)
    training_minibatches = generate_minibatches(train_dataset)

    MINE.fit(training_minibatches, val_dataset, 2000)

    # MINE.plot_training(true_mi)
    # plt.plot([MINE.estimate_mi(x, z) for i in range(1000)])

    # MINE2 = Mine2(1, 20, 0.5, 1)
    # MINE2.run_epochs(x, z, 5000, viewProgress=False)
    # # MINE2.plot_training(true_mi)
    # # plt.plot([MINE2.estimate_mi(x, z) for i in range(1000)])
    #
    # MINE3 = Mine2(1, 30, 0.5, 1)
    # MINE3.run_epochs(x, z, 5000, viewProgress=False)
    toc = time.time()

    MINE.plot_training()

    print(toc - tic)

    # plt.axhline(y=true_mi, color='r', linestyle='-')
