import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from utils import analysis_pipeline
import math

def generate_stim(set_size):
    stimulus = []
    for stim in range(0,int(set_size)):
        stimulus.append(np.random.uniform(0,1))
    
    return stimulus

def generate_stim_dist(set_size,dist):
    points = np.arange(0,1,step=dist)
    return np.random.choice(points,int(set_size),replace=False)

def generate_mean_set(set_size):
    """generate a mean value based on the set size

    Args:
        set_size (int[]): the amount of objects in the set

    Returns:
        _float_: _the calculated mean value_
    """
    mean = 2 * (0.5 ** set_size)
    return mean

def generate_mean_delay(delay_interval):
    """generate a mean value based on the set size

    Args:
        delay_interval (int[]): the amount of objects in the set

    Returns:
        _float_: _the calculated mean value_
    """
    mean = 2 * (0.5 ** delay_interval)
    return mean

def generate_cue(stim):
    cue = np.where(stim == np.random.choice(stim))
    return cue[0][0]

def generate_sample(mean, std):
    """creates a sample set

    Args:
        mean (_float_): mean value of sample
        std (_float_): standard deviation of sample
        size (_float_): how big the sample output will be

    Returns:
        _float_: _sample set_
    """
    return np.random.normal(mean, std)

def decision_random(stim, sample, cue, threshold, std):
    """Makes a decision about the sample data based on the threshold parameter. This function accounts for random guessing 

    Args:
        samples (_float_): sample set
        threshold (_float_): threshold for different decisions

    Returns:
        _float_: array of decisions based on the threshold
    """
    if sample >= threshold:
        return np.random.normal(stim[cue],std) 
    if sample < threshold:
        return np.random.uniform(0,1)
def decision_confused(stim, sample, cue, threshold, std):
    random_cue = random.randint(0,len(stim))

    if sample >= threshold:
        return np.random.normal(stim[cue],std) 
    if sample < threshold:
        return np.random.normal(stim[random_cue-1],std)
    
def normalize_decision(theta):
    if theta > (2 * np.pi):
        return theta % (2 * np.pi)
    elif theta < 0:
        return 2 * np.pi + theta 
    else:
        return theta
    
def find_stim_choice_thresh(stim, decision):
    distances = []
    for value in stim:
        distances.append(analysis_pipeline.find_angular_dist(value,decision))
    distances = np.abs(distances)
    closest_choice = np.argmin(distances)
    
    if distances[closest_choice] < 2:
        return closest_choice
    else:
        return None

def find_stim_choice(stim, decision):
    distances = []
    for value in stim:
        distances.append(analysis_pipeline.find_angular_dist(value,decision))
    distances = np.abs(distances)
    closest_choice = np.argmin(distances)
    
    return closest_choice
         

def generate_accuracy(cue, choice):
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

def set_create_df(trials, std_sample):
    set_sizes = np.random.randint(3,6,trials)
    df = pd.DataFrame(set_sizes, columns=['set_size'])
    stimulus = []
    mean = []
    cue = []
    sample = []

    for (row_index,row_data) in df.iterrows():
        stimulus.append(generate_stim_dist(row_data['set_size'],0.1))
    df['stim'] = stimulus
    for (row_index,row_data) in df.iterrows():
        mean.append(generate_mean_set(row_data['set_size']))
    df['mean'] = mean
    for (row_index,row_data) in df.iterrows():
        cue.append(generate_cue(row_data['stim']))
    df['cue'] = cue
    for (row_index,row_data) in df.iterrows():
        sample.append(generate_sample(row_data['mean'],std_sample))
    df['sample'] = sample
    
    return df
def delay_create_df(trials, std):
    set_sizes = np.random.randint(3,6,trials)
    df = pd.DataFrame(set_sizes, columns=['set_size'])
    
    delay_interval = []
    for i in range(0,trials):
        delay_interval.append(np.random.choice([0.5,1,1.5,2,3]))
    df['delay_s'] = delay_interval

    stimulus = []
    mean = []
    cue = []
    sample = []

    for (row_index,row_data) in df.iterrows():
        stimulus.append(generate_stim_dist(row_data['set_size'],0.1))
    df['stim'] = stimulus
    for (row_index,row_data) in df.iterrows():
        mean.append(generate_mean_delay(row_data['delay_s']))
    df['mean'] = mean
    for (row_index,row_data) in df.iterrows():
        cue.append(generate_cue(row_data['stim']))
    df['cue'] = cue
    for (row_index,row_data) in df.iterrows():
        sample.append(generate_sample(row_data['mean'],std))
    df['sample'] = sample
    
    return df

def run_model_random(df, threshold, std_decision):
    df = df.copy()
    decision = []
    guessing = []
    normalized_decision = []
    stim_rad = []
    choice = []
    choice_rad = []

    for (row_index,row_data) in df.iterrows():
        if row_data['sample'] < 0.1:
            guessing.append(1)
        else:
            guessing.append(0)
        decision.append(decision_random(row_data['stim'],row_data['sample'],row_data['cue'],threshold,std_decision))

    df['decision'] = decision
    df['guessing'] = guessing
    df['decision_rad'] = analysis_pipeline.rad_convert(df['decision'])

    for (row_index,row_data) in df.iterrows():
        buffer = []
        for value in row_data['stim']:
            buffer.append(analysis_pipeline.rad_convert(value))
        stim_rad.append(buffer)
    df['stim_rad'] =  stim_rad

    for (row_index,row_data) in df.iterrows():
        normalized_decision.append(normalize_decision(row_data['decision_rad']))
    df['normalized_decision'] = normalized_decision

    for (row_index,row_data) in df.iterrows():
        choice.append(find_stim_choice(row_data['stim_rad'],row_data['normalized_decision']))
    df['choice'] = choice

    return df

def run_model_confused(df, threshold,std_decision):
    df = df.copy()
    decision = []
    guessing = []
    normalized_decision = []
    stim_rad = []
    choice = []
    choice_rad = []

    for (row_index,row_data) in df.iterrows():
        if row_data['sample'] < 0.1:
            guessing.append(1)
        else:
            guessing.append(0)
        decision.append(decision_confused(row_data['stim'],row_data['sample'],row_data['cue'],threshold,std_decision))

    df['decision'] = decision
    df['guessing'] = guessing
    df['decision_rad'] = analysis_pipeline.rad_convert(df['decision'])

    for (row_index,row_data) in df.iterrows():
        buffer = []
        for value in row_data['stim']:
            buffer.append(analysis_pipeline.rad_convert(value))
        stim_rad.append(buffer)
    df['stim_rad'] =  stim_rad

    for (row_index,row_data) in df.iterrows():
        normalized_decision.append(normalize_decision(row_data['decision_rad']))
    df['normalized_decision'] = normalized_decision

    for (row_index,row_data) in df.iterrows():
        choice.append(find_stim_choice(row_data['stim_rad'],row_data['normalized_decision']))
    df['choice'] = choice

    return df

def model_analysis(df):
    df = df.copy()
    distances = []
    accuracy = []
    ang_dist_cue = []
    ang_dist = []

    for (row_index,row_data) in df.iterrows():
        buffer = []
        for value in row_data['stim_rad']:
            buffer.append(analysis_pipeline.find_angular_dist(value,row_data['normalized_decision']))
        distances.append(buffer)
    df['distances'] = distances
    
    for (row_index,row_data) in df.iterrows():
        accuracy.append(generate_accuracy(row_data['cue'],row_data['choice']))
    df['correct'] = accuracy

    for (row_index,row_data) in df.iterrows():
        ang_dist_cue.append(analysis_pipeline.find_angular_dist(row_data['stim_rad'][row_data['cue']],row_data['decision_rad']))
    df['precision_difference_0'] = ang_dist_cue
    df['precision_difference_0_abs'] = df['precision_difference_0'].abs()

    for (row_index,row_data) in df.iterrows():
        if math.isnan(row_data['choice']):
            ang_dist.append(None)
        else:
            ang_dist.append(analysis_pipeline.find_angular_dist(row_data['stim_rad'][int(row_data['choice'])],row_data['decision_rad']))
    df['precision_difference_choice'] = ang_dist
    df['precision_difference_choice_abs'] = df['precision_difference_choice'].abs()
    return df
    




