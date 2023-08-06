import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from utils import analysis_pipeline
from utils import modeling
import math

def generate_delay(df):
    return np.random.choice([0.5,1,1.5,2,3])
    
def generate_encoding(df):
    return np.random.choice([0.5,1,1.5,2,3])

def generate_set():
    return np.random.randint(3,6)
     
def generate_trial_length(delay,encoding,rxn):
    return delay + encoding + rxn

def generate_rxn_time():
    return np.clip(np.random.normal(0.8773594377510041, 0.7636419306909825),0,3)

def generate_trial_type(curr_trial, num_trials):
    num_trials = num_trials                      
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
    

def run_train_trial(df,std):
    last_train_trial = len(df[df['trial_type'] == 'train']) - 1
    delay = generate_delay(df)
    encoding = generate_encoding(df)
    rxn_time = generate_rxn_time()
    set_size = df.iloc[0]['set_size']
    trial_length = generate_trial_length(delay,encoding,rxn_time)
    stim = df.iloc[0]['stim']
    mean = modeling.generate_mean_delay(delay)
    cue = modeling.generate_cue(stim)
    sample = modeling.generate_sample(mean,std)
    total_delay = df.iloc[last_train_trial:]['trial_length'].sum()


    trial_num = df.iloc[-1]['trial_num'] + 1


    dict = {'trial_num':trial_num,
            'delay_s': delay,
            'encoding': encoding,
            'set_size': set_size,
            'rxn_time': rxn_time,
            'trial_length': trial_length,
            'stim': stim,
            'mean': mean,
            'cue': cue,
            'sample': sample,
            'trial_type': 'train',
            'cumulative_delay': total_delay,
            'cumulative_encoding':0,
            }
    
    df_new = pd.DataFrame([dict])

    return pd.concat([df,df_new],ignore_index=True)

def run_test_trial(df,std):
    last_train_trial = len(df[df['trial_type'] == 'train']) - 1
    delay = 0
    encoding = 0
    set_size = df.iloc[0]['set_size']
    rxn_time = generate_rxn_time()
    trial_length = generate_trial_length(delay,encoding,rxn_time)
    stim = df.iloc[0]['stim']
    
    total_delay = df.iloc[last_train_trial:]['trial_length'].sum()

    total_encoding = df['encoding'].sum()  
    
    mean = modeling.generate_mean_delay(total_delay)
    cue = modeling.generate_cue(stim)
    sample = modeling.generate_sample(mean,std)
    
    trial_num = df.iloc[-1]['trial_num'] + 1

    dict = {'trial_num':trial_num,
            'delay_s': delay,
            'encoding': encoding,
            'set_size': set_size,
            'trial_length': trial_length,
            'rxn_time': rxn_time,
            'stim': stim,
            'mean': mean,
            'cue': cue,
            'sample': sample,
            'trial_type': 'test',
            'cumulative_delay': total_delay,
            'cumulative_encoding': total_encoding,
            }
    df_new = pd.DataFrame([dict])

    return pd.concat([df,df_new],ignore_index=True)

def create_ltm_block(trials,std):
    set_size = generate_set()
    stim = modeling.generate_stim_dist(set_size,0.1)
    #stim = modeling.generate_stim(set_size)
    dict = {'trial_num':-1,
            'delay_s': 0,
            'encoding': 0,
            'set_size': set_size,
            'trial_length': 0,
            'rxn_time': 0,
            'stim': stim,
            'mean': 0,
            'cue': 0,
            'sample': 0,
            'trial_type' : None,
            'cumulative_delay': 0,
            'cumulative_encoding':0,
            }
    
    df = pd.DataFrame([dict])

    trial_num = -1
    for i in range(0,trials):
        trial_num += 1
        trial_type = generate_trial_type(curr_trial=trial_num,num_trials=trials)
        if trial_type == 'test':
            df = run_test_trial(df,std)
        if trial_type == 'train':
            df = run_train_trial(df,std)

    df = df.drop(labels=0,axis=0)
    df['cumulative_delay_bins'] = pd.qcut(df['cumulative_delay'],4,labels=np.arange(4),duplicates='drop')
    return df

def create_df(blocks,std,trials):
    df = pd.DataFrame()
    for i in range(0,blocks):
        df = pd.concat([df,create_ltm_block(trials,std)], ignore_index = True)    
    return df
