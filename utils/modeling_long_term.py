import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from utils import analysis_pipeline
from utils import modeling
import math

def generate_delay(df):
    delay_interval = np.random.choice([0.5,1,1.5,2,3])
    df['delay_s'] = delay_interval
    return df['delay_s']

def generate_encoding(df):
    encoding_s = np.random.choice([0.5,1,1.5,2,3])
    df['encoding_s'] = encoding_s
    return df['delay_s']

def generate_set(df):
    set_sizes = np.random.randint(3,6)
    df['set_sizes'] = set_sizes
    return df['set_sizes']

def generate_trial_length(df):
    return df['delay'] + df['encoding'] + df['rxn_time_s']

def generate_rxn_time():
    np.random.normal(0.8773594377510041, 0.7636419306909825)

def generate_trial_type(df):
    num_trials = 20
    curr_trial = df['trial_num']                       
    p_visible = np.clip(
            (num_trials - curr_trial)/num_trials,
            a_min=0,
            a_max=1)
    p_visible = p_visible ** 3
    trial_type =  np.random.choice([0, 1], p=[1 - p_visible, p_visible])
    if trial_type == 1:
        return 'train'
    else:
        return 'test'

# #def run_train_trial(df, std):
#     delay = generate_delay(df)
#     df['delay_s'] = delay

#     df['encoding_s'] = generate_encoding(df)

#     set_size = generate_set(df)
#     df['set_size'] = set_size

#     df['trial_length'] = generate_trial_length(df)

#     df['rxn_time'] = generate_rxn_time(df)

#     stim = modeling.generate_stim(set_size)
#     df['stim'] = stim

#     mean = modeling.generate_mean_delay(delay)
#     df['mean'] = mean

#     df['cue'] = modeling.generate_cue(stim)

#     df['sample'] = modeling.generate_sample(mean,std)
    
#     return df

def run_train_trial(df,std):
    delay = generate_delay(df)
    encoding = generate_encoding(df)
    set_size = generate_set(df)
    trial_length = generate_trial_length(df)
    rxn_time = generate_rxn_time(df)
    stim = modeling.generate_stim(set_size)
    mean = modeling.generate_mean_delay(delay)
    cue = modeling.generate_cue(stim)
    sample = modeling.generate_sample(mean,std)

    dict = {'delay_s': [delay],
            'encoding': [encoding],
            'set_size': [set_size],
            'trial_length': [trial_length],
            'rxn_time': [rxn_time],
            'stim': [stim],
            'mean': [mean],
            'cue': [cue],
            'sample': [sample],
            }
    df = df.append(dict, ignore_index = True)

    return df

def run_test_trial(df,std):
    last_train_trial = df['trial_type'].rfind('train')
    delay = 0
    encoding = 0
    set_size = df.iloc[last_train_trial]['set_size']
    trial_length = generate_trial_length(df)
    rxn_time = generate_rxn_time(df)
    stim = df.iloc[last_train_trial]['stim']
    mean = modeling.generate_mean_delay(delay)
    cue = modeling.generate_cue(stim)
    sample = modeling.generate_sample(mean,std)
    
    trial_num = df.iloc[-1]['trial_num'] + 1

    dict = {'trial_num':[trial_num],
            'delay_s': [delay],
            'encoding': [encoding],
            'set_size': [set_size],
            'trial_length': [trial_length],
            'rxn_time': [rxn_time],
            'stim': [stim],
            'mean': [mean],
            'cue': [cue],
            'sample': [sample],
            }
    df = df.append(dict, ignore_index = True)
    
    return df

def create_ltm_block(trials,df,std):
    trial_num = 1
    for i in range(0,trials):
        trial_num += 1
        trial_type = generate_trial_type()
        if trial_type == 'test':
            run_test_trial(df,std)
        if trial_type == 'train':
            run_train_trial(df,std)

def create_df(blocks,std,trials):
    df = pd.DataFrame(1, columns=['trial_num'])
    for i in range(0,blocks):
        create_ltm_block(trials,df,std)
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
        accuracy.append(modeling.generate_accuracy(row_data['cue'],row_data['choice']))
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