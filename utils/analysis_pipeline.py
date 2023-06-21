"""Function to streamline plotting and analysis of behavioral data"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def _conv_angle(theta):
    """change the angle to be measured in the counterclockwise direction

    Args:
        theta (float): angle to be converted

    Returns:
        _type_: if angle is negative it will be changed to its measurement in the counterclockwise direction; otherwise it will stay the same
    """
    if theta < 0:
        return 2 * np.pi + theta
    return theta

def find_angular_dist(theta1, theta2):
    """find the angular distance between two angles

    Args:
        theta1 (float): first angle
        theta2 (float): second angle

    Returns:
        float : direction and magnitude from the first angle 
    """
    conv_theta1 = _conv_angle(theta1)
    conv_theta2 = _conv_angle(theta2)
    TAU = 2 * np.pi
    a = (theta1 - theta2) % TAU
    b = (theta2 - theta1) % TAU
    return -a if a < b else b

def round(column,df):
    """rounding the variables you want to plot against to be able to group the data

    Args:
        column (str): name of the column you want to round
        df (DataFrame): the dataframe from which you will be pulling from 
    """
    df["rounded_" + column] = np.round(df[column], 1)

def plot_accuracy(column, df, xlabel, title):
    """plotting the accuracy against another variable in the dataset 

    Args:
        column (str): name of the column you want to plot against accuracy
        df (DataFrame): the dataframe from which you will be pulling from 
        xlabel (str): the x axis label for the graph
        ylabel (str): the y axis label for the graph
        title (str): the title of the graph
    """
    ylabel = "Accuracy"
    df.groupby(column).correct.mean().plot(xlabel= xlabel, ylabel= ylabel, title=title)

def plot_error(column, df, xlabel, title):
    """plotting the error against another variable in the dataset

    Args:
        column (str): name of the column you want to plot against error
        df (DataFrame): the dataframe from which you will be pulling from 
        xlabel (str): the x axis label for the graph
        title (str): the title of the graph
    """
    precision_difference_arr = []

    for (row_index,row_data) in df.iterrows():
        precision_difference_arr.append(find_angular_dist(row_data['object_0_theta'],row_data['response_theta']))

    df['precision_difference_0'] = precision_difference_arr
    df['precision_difference_0_abs'] = df['precision_difference_0'].abs()

    ylabel = 'Angular Distance from Target (rad)'

    df.groupby(column).precision_difference_0_abs.mean().plot(xlabel= xlabel, ylabel= ylabel, title=title)

def plot_rt(column, df, xlabel, title):
    """plotting the reaction time against another variable in the dataset

    Args:
        column (str): name of the column you want to plot against reaction time
        df (DataFrame): the dataframe from which you will be pulling from 
        xlabel (str): the x axis label for the graph
        ylabel (str): the y axis label for the graph
        title (str): the title of the graph
    """
    ylabel = "Reaction Time (s)"
    df['reaction_time_s'] = df['reaction_time_steps'] / 60
    df.groupby(column).reaction_time_s.mean().plot(xlabel= xlabel, ylabel= ylabel, title=title)

def plot_hist(column, df, xlabel, title):
    """plotting a distribution of the parameter given

    Args:
        column (str): name of the column you want to plot 
        df (DataFrame): the dataframe from which you will be pulling from 
        xlabel (str): the x axis label for the graph
        title (str): the title of the graph
    """
    df[column].plot(kind="hist",xlabel=xlabel,title=title)

def plot_precision(column,df,xlabel,title):
    """plotting precision of the choice in regards to the neighborhod that was chosen

    Args:
        df (DataFrame): the dataframe from which you will be pulling from 
        xlabel (str): the x axis label for the graph
        title (str): the title of the graph
    """

    response_precision_arr = []
    # s = 'object_' + str(i) + '_theta'
    # df.response_object_ind.astype(int)

    for (row_index,row_data) in df.iterrows():
        s = 'object_' + str(int(row_data['response_object_ind'])) + '_theta'
        response_precision_arr.append(find_angular_dist(row_data[s],row_data['response_theta']))

    df['precision_difference'] = response_precision_arr

    ylabel = 'Angular Distance from Object (rad)'
    df.groupby(column).precision_difference.mean().plot(xlabel= xlabel, ylabel= ylabel, title=title)

def plot_SD(column, df, xlabel, title):
    """Plot the standard deviation as a function of a column

    Args:
        column (str): the column you want to find the standard deviation of
        df (DataFrame): the datafram from which you will be pulling from 
        xlabel (str): the x axis label for the graph
        title (str): the title of the graph
    """

    df[column].std().plot(xlabel=xlabel,ylabel='Standard Deviation',title=title)

