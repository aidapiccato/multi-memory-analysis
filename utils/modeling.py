import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_stim(set_size):
    stimulus = []
    for stim in range(0,set_size):
        stimulus.append(np.random.uniform(0,1))
    
    return stimulus

def generate_mean(set_size):
    """generate a mean value based on the set size

    Args:
        set_size (int[]): the amount of objects in the set

    Returns:
        _float_: _the calculated mean value_
    """
    mean = 7 / set_size
    return mean

def cue(stim):
    return np.random.choice(stim)

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

def decision_random(stim, sample, cue, threshold):
    """Makes a decision about the sample data based on the threshold parameter. This function accounts for random guessing 

    Args:
        samples (_float_): sample set
        threshold (_float_): threshold for different decisions

    Returns:
        _float_: array of decisions based on the threshold
    """

    if sample >= threshold:
        return np.random.normal(cue,1) 
    if sample < threshold:
        return np.random.uniform(0,1)
    
def decision_confused(stim, sample, cue, threshold):
    random_cue = np.random.choice(stim)

    if sample >= threshold:
        return np.random.normal(cue,1) 
    if sample < threshold:
        return np.random.normal(random_cue,1)
    
def find_stim_choice(stim, decision):
    stim.sort()
    midpoints = [[]]
    i=0
    if i != 0 and i != len(stim) -1:
        while i < stim.len()-1:
            upper_limit = (stim[i+1] - stim[i]) / 2
            lower_limit = (stim[i] - stim[i-1]) / 2
            midpoints.append([upper_limit,lower_limit])
    elif i == 0:
        upper_limit = (stim[len(stim)-1] - stim[0]) / 2
        lower_limit = (stim[0] - stim[1]) / 2
        midpoints.append([upper_limit,lower_limit])
    elif i == stim.len():
        upper_limit = (stim[len(stim)+1] - stim[len(stim)]) / 2
        lower_limit = (stim[len(stim)] - stim[len(stim)-1]) / 2
        midpoints.append([upper_limit,lower_limit])
        
    for range in midpoints:
        if decision in range(range[0],range[1]):
            return midpoints.index(range)
        else:
            return None

def accuracy(cue, choice):
    """measure the accuracy of the model with the ground truth data set and the responses from the model

    Args:
        ground_truth (int): the experimental data with which you will be comparing your values to
        responses (int): the simulated data that you will compare against the experimental data

    Returns:
        _float_: accuracy and mean accuracy of comparison
    """
    
    if cue == choice:
        accuracy=1
    if cue != choice:
        accuracy=0

    return accuracy
