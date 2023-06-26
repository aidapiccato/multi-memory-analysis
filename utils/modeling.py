import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from utils import analysis_pipeline

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
    return stim.index(np.random.choice(stim))

def sample(mean, std):
    """creates a sample set

    Args:
        mean (_float_): mean value of sample
        std (_float_): standard deviation of sample
        size (_float_): how big the sample output will be

    Returns:
        _float_: _sample set_
    """
    return np.random.normal(mean, std)

def decision_random(stim, sample, cue, threshold):
    """Makes a decision about the sample data based on the threshold parameter. This function accounts for random guessing 

    Args:
        samples (_float_): sample set
        threshold (_float_): threshold for different decisions

    Returns:
        _float_: array of decisions based on the threshold
    """
    if sample >= threshold:
        return np.random.normal(stim[cue],1) 
    if sample < threshold:
        return np.random.uniform(0,1)
def decision_confused(stim, sample, cue, threshold):
    random_cue = random.randint(len(stim))

    if sample >= threshold:
        return np.random.normal(stim[cue],1) 
    if sample < threshold:
        return np.random.normal(stim[random_cue],1)
    
def normalize_decision(theta):
    if theta > (2 * np.pi):
        return theta % (2 * np.pi)
    else:
        return theta
    
def find_stim_choice(stim, decision):
    distances = []
    for value in stim:
        distances.append(analysis_pipeline.find_angular_dist(value,decision))
    closest_choice = distances.index(min(distances)) + 1
    
    if abs(distances[closest_choice - 1]) > 2:
        return closest_choice
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
