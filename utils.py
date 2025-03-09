import numpy as np
import torch
from torch import nn, Tensor
import yaml

def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Configuration parameters.
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """
    Initialize a given layer using orthogonal initialization for the weights
    and constant initialization for the biases.

    Args:
        layer (nn.Module): The neural network layer to initialize.
        std (float): The standard deviation (gain) for orthogonal initialization. 
                               Default is np.sqrt(2).
        bias_const (float): The constant value to initialize biases. Default is 0.0.

    Returns:
        nn.Module: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def normalize(vect: Tensor):
    """
    Normalize a tensor.
    A small constant is added to the denominator for numerical stability.

    Args:
        vect (Tensor): The input tensor to normalize.

    Returns:
        Tensor: The normalized tensor.
    """
    return (vect - vect.mean()) / (vect.std() + 1e-8)