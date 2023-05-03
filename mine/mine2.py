import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def generate_minibatches(trainig_set: torch.tensor, size: int, shuffle=False):
    """
    Function that returns minibatches from an original trainign dataset,
    of solicited size.

    Parameters
    ----------
    trainig_set : torch.tensor
        The complete dataset used for training, to be splitted into minibatches.
    size : int
        The size of the individual minibatches. If length of dataset is not
        a multiple of the size, then last minibatch may contain less data.
    shuffle : bool
        Flag that indicates if dataset should be suffled before splitting
        in minibatches

    Return
    ------
    minibatches : torch.tensor
        A tensor of minibatches of demanded size.
    """

    if shuffle:
        size = trainig_set.shape[0]
        rand_perm = torch.randperm(size)
        trainig_set = trainig_set[rand_perm]

    minibatches = trainig_set.split(size, dim=0)
    return minibatches


def moving_average(raw_data: list, filtered_data: list, k: int):
    """
    Applies moving average to the raw_data signal and saves it onto
    the filtered_data signal

    Parameters
    ----------
    raw_data : list
        Unfiltered data containing the last element of the signal,
        in real time acquisition.
    filtered_data : list
        Filtered data up to one-to-last element, last element to be
         appended in the present function.
    k : int
        Gap to consider the average from, i.e. last k elements of
        raw_data would be averaged out and appended to filtered_data

    """
    last_data = raw_data[-1]
    if len(filtered_data) == 0:
        filtered_data.append(last_data)
    else:
        ref_value = raw_data[(-k) if len(raw_data) > k else 0]
        filtered_data.append(filtered_data[-1] + (last_data - ref_value) / k)


def exponential_moving_average(raw_data: list, filtered_data: list, alfa: float):
    pass


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
        hidden_layers : int
            Amount of hidden layers for the network. The input layer will
            be of 2 dimensions (for the 2 input signals), and the output will
            be of 1 dimension. Minimum value permitted: 1.
        neurons : int
            Amount of neurons per hidden layer.
        cuda : str
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
        self.training_progress = []
        self.training_filtered = []
        self.validation_progress = []
        self.validation_filtered = []
        self.k = 100

    def forward(self, input: torch.tensor):
        """
        Defines the forward pass of the network.

        It passes the input data through the layers of the model defined in the initializer.
        Overload of forward method is compulsory for models that inherit from nn.Module.

        Parameters
        ----------
        input : torch.tensor
            Batch of input data to pass into the model

        Returns
        -------
        output : torch.tensor
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
        train_dataset : torch.tensor
            Input to the model. It consists of the concatenation of the
            two signals to analyze.

        Returns
        -------
        loss : torch.tensor
            Tensor of one dimension which contains the loss of the network.
            The loss is actually the negative of the value we really want
            to maximize, which is the mutual information estimation.
        """

        # Obtain size of minibatch
        minibatch_size = train_dataset.shape[0]

        # First input (x and intact z)
        out1 = self(train_dataset)

        # Second input (x and permutted z) -> to break correlation between both signals
        permutation = torch.randperm(minibatch_size)
        permuted_z = train_dataset[:, 1][permutation]
        # CHECKOUT: better way to do this
        permuted_input = torch.cat((train_dataset[:, 0].unsqueeze(1), permuted_z.unsqueeze(1)), dim=1)
        out2 = self(permuted_input)

        # Mutual information estimation
        mi_estimation = torch.mean(out1) - torch.logsumexp(out2, 0) + torch.log(torch.tensor(minibatch_size))

        # Return the loss: the loss is the opposite of the mutual info estimation
        loss = -mi_estimation
        return loss

    def evaluate(self, input_dataset: torch.tensor):
        """
        Evaluates the model output with respect to given input.

        Does not perform backpropagation nor gradient calculation.

        Parameters
        ----------
        input_dataset : torch.tensor
            Input data to the model.

        Return
        ------
        mi_estimation : torch.tensor
            Output of the model, i.e the mutual information estimation.
        """

        with torch.no_grad():

            # Obtain size of minibatch
            batch_size = input_dataset.shape[0]

            # First input (x and intact z)
            # net_input_1 = torch.cat((x_batch, z_batch), dim=1)
            out1 = self(input_dataset)

            # Second input (x and permutted z) -> to break correlation between both signals
            permutation = torch.randperm(batch_size)
            permuted_z = input_dataset[:, 1][permutation]
            # CHECKOUT: better way to do this
            permuted_input = torch.cat((input_dataset[:, 0].unsqueeze(1), permuted_z.unsqueeze(1)), dim=1)
            out2 = self(permuted_input)

            # Mutual information estimation
            mi_estimation = torch.mean(out1) - torch.logsumexp(out2, 0) + torch.log(torch.tensor(batch_size))

        return mi_estimation

    def fit(self, signal_x: torch.tensor, signal_z: torch.tensor,
            num_epochs: int,
            train_percent: int = 80,
            minibatch_size: int = 1,
            learning_rate: float = 1e-4,
            random_partition: bool=False,
            show_progress: bool = False):
        """
        Trains the MINE model

        Uses the provided training data loader and
        validates it using the provided validation data loader.

        We expect the training and validation datasets to be fractions of the original paired
        signals. For example, 80-20%, i.e. the same pair of signals is split
        into two parts, a bigger one for training and a smaller one for validation.

        Parameters
        ----------
        signal_x : torch.tensor
            First signal to use to find mutual information.
        signal_z : torch.tensor
            Second signal to use to find mutual information.
        num_epochs : int
            An integer specifying the number of epochs to train the model for.
        train_percent : int
            Percentage of dataset to use for training. Remaining will be used
            for validation. Default value is 80.
        minibatch_size : int
            Size of minibatches to be created during training.
        learning_rate : float
            A float specifying the learning rate to use for the optimizer.
        random_partition : bool
            Flag that indicates if you want the validation data to be randomly
            selected from the entire dataset, otherwise it will be taken from
            the end of the signal.
        show_progress : bool
            Flag to turn on the percetage of training progress-bar.
        """

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        input_dataset = torch.cat((signal_x, signal_z), dim=1)

        # Size calculation of training and validation datasets
        train_size = int(len(input_dataset) * train_percent / 100)
        val_size = len(input_dataset) - train_size

        # Random selection of validation data
        if random_partition:
            rand_index = np.random.randint(0, train_size-val_size)
            train_size_1 = rand_index + 1
            train_size_2 = train_size-train_size_1
            train_dataset_1, val_dataset, train_dataset_2 = \
                input_dataset.split([train_size_1, val_size, train_size_2], dim=0)
            train_dataset = torch.cat((train_dataset_1, train_dataset_2), dim=0)
        else:
            train_dataset, val_dataset = input_dataset.split([train_size, val_size], dim=0)

        # Training data in train_dataset
        # Validation data in val_dataset

        # ###################### Loop for epochs ######################
        # In each epoch we train and validate
        for epoch in tqdm(range(num_epochs), disable=not show_progress):

            # ############### Training loop ###############
            for batch in generate_minibatches(train_dataset, size=minibatch_size, shuffle=True):
                loss = self.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # ################# Validation #################
            # TODO Process the signal in validation_progress and cut the
            #  training epochs when this signal is decreasing
            # Raw training signal
            result_train = self.evaluate(train_dataset)
            self.training_progress.append(result_train.item())
            moving_average(self.training_progress, self.training_filtered, self.k)
            # Raw validation signal
            result_val = self.evaluate(val_dataset)
            self.validation_progress.append(result_val.item())
            moving_average(self.validation_progress, self.validation_filtered, self.k)

    def plot_training(self, true_mi: float = None, smooth: bool = True, save: bool = False):
        """
        Plots the progress of training and of validation process during the training epochs.
        The curves are actually smoothed out with moving average filter.

        Parameters
        ----------
        true_mi : float
            True value of mutual information or value of reference we want to
            show in the plot.
        smooth : bool
            True if you want to plot the smoothed versions of the training and
            validation signals.
        save : bool
            Flag that indicates if we want to save the plot to a pdf file.
        """

        if smooth:
            plt.plot(self.training_filtered, color='b', label='Training')
            plt.plot(self.validation_filtered, color='g', label='Validation')
        else:
            plt.plot(self.training_progress, color='b', label='Training')
            plt.plot(self.validation_progress, color='g', label='Validation')

        if true_mi is not None:
            plt.axhline(y=true_mi, color='r', linestyle='-')

        plt.ylabel('Mutual information estimation')
        plt.xlabel('epoch')
        plt.title(f"MINE progress")
        plt.grid()
        plt.legend(loc='best')
        if save:
            plt.savefig(f'mine{self.hiddenLayers}-{self.neurons}.pdf')
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
    # Informacion mutua mediante formula
    true_mi = -0.5 * np.log(np.linalg.det(cov_matrix))
    print(f"The real mutual information is {true_mi}")

    MINE = Mine2(3, 100)

    tic = time.time()
    MINE.fit(x, z, num_epochs=10000, minibatch_size=500, learning_rate=5e-4, random_partition=True, show_progress=True)
    toc = time.time()

    MINE.plot_training(true_mi)

    # print(toc - tic)
