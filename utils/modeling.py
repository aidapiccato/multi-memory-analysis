import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_mean(set_size):
    """generate a mean value based on the set size

    Args:
        set_size (_float_): the amount of objects in the set

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

def decision(samples, threshold):
    """Makes a decision about the sample data based on the threshold parameter

    Args:
        samples (_float_): sample set
        threshold (_float_): threshold for different decisions

    Returns:
        _float_: array of decisions based on the threshold
    """
    decision = 0
    for sample in samples:
        if sample >= threshold:
            decision = 1
        if sample < threshold:
            decision = 0
    return decision

def accuracy(ground_truth, decision):
    """measure the accuracy of the model with the ground truth data set and the responses from the model

    Args:
        ground_truth (int): the experimental data with which you will be comparing your values to
        responses (int): the simulated data that you will compare against the experimental data

    Returns:
        _float_: accuracy and mean accuracy of comparison
    """
    
    if ground_truth == decision:
        accuracy=1
    if ground_truth != decision:
        accuracy=0

    return accuracy
