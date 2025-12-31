# numpy and pandas
import numpy as np
import pandas as pd

from typing import Tuple

import torch


def sliding_window(dataframe: pd.DataFrame, window_size: int) -> Tuple[np.array, np.array]:
    """Sliding window for input sequences.
    
    Args:
        dataframe (pd.DataFrame): The base model for the input sequences to have the window size implemented on.
        window_size (int): The length of each sequence.
    
    Returns:
        X (np.array): np.array with sliding window applied to features.
        y (np.array): np.array with sliding window applied to targets.
    """
    X, y = [], []
    for i in range(len(dataframe) - window_size):
        Xi = dataframe[i:(i+window_size)]
        yi = dataframe[(i+window_size)]
        X.append(Xi)
        y.append(yi)
    return np.array(X), np.array(y)



def convert_array_to_tensor(array: np.array) -> torch.Tensor:
    """_summary_

    Args:
        array (np.array): _description_

    Returns:
        torch.Tensor: _description_
    """
    return torch.Tensor(array)
    
    