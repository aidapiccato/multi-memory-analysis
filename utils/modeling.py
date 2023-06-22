import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_stim():
    # return np.random.uniform(0,1)
    return 0.488838
def generate_mean(set_size):
    """generate a mean value based on the set size

    Args:
        set_size (int): the amount of objects in the set

    Returns:
        _float_: _the calculated mean value_
    """
    mean = 7 / set_size
    return mean

def sample(mean, std, size):
    """creates a sample set

    Args:
        mean (_float_): mean value of sample
        std (_float_): standard deviation of sample
        size (_float_): how big the sample output will be

    Returns:
        _float_: _sample set_
    """
    return np.random.normal(mean, std, size)

def decision(stim, sample, threshold):
    """Makes a decision about the sample data based on the threshold parameter

    Args:
        samples (_float_): sample set
        threshold (_float_): threshold for different decisions

    Returns:
        _float_: array of decisions based on the threshold
    """
    decision = 0

    if sample >= threshold:
        return stim 
    if sample < threshold:
        return np.random.uniform(0,1)

def accuracy(stim, decision):
    """measure the accuracy of the model with the ground truth data set and the responses from the model

    Args:
        ground_truth (int): the experimental data with which you will be comparing your values to
        responses (int): the simulated data that you will compare against the experimental data

    Returns:
        _float_: accuracy and mean accuracy of comparison
    """
    
    if stim == decision:
        accuracy=1
    if stim != decision:
        accuracy=0

    return accuracy
