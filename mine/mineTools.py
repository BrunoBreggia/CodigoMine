import torch
import torch.nn as nn
import numpy as np

def actFunc(funcname:str):
    """
    Returns corresponding pytorch activation function.
    Receives string as input.
    """
    funcname = funcname.lower()
    activation_functions = {
            "relu" : nn.ReLU,
            "leakyrelu" : nn.LeakyReLU,
            "gelu" : nn.GELU,
            "elu" : nn.ELU,
            "silu" : nn.SiLU,

            }
    if funcname not in activation_functions:
        raise KeyError("Activation function name not recognized")
    return activation_functions[funcname]


def generate_batches(x, z, batch_size=1, shuffle=True):
    """
    Generador que devuelve batches de tamaño especificado
    a partir del total de las muestras (x,z) que le suministra.
    No disocia los emparejamientos, permite permutar los
    pares (x,z) antes de devolver los batches.
    """
    
    assert len(x) == len(z), "Input vectors must contain the same number of elements"
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z).float()

    samples = len(x)

    if shuffle:
        rand_perm = torch.randperm(samples)
        x = x[rand_perm]
        z = z[rand_perm]

    for i in range(samples//batch_size):
        x_b = x[i*batch_size : (i+1)*batch_size]
        z_b = z[i*batch_size : (i+1)*batch_size]

        yield x_b, z_b



