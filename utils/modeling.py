import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def sample(mean, std, size):
    return np.random.normal(mean, std, size)

def decision(samples, threshold):
    decision1 = 1
    decision2 = 0
    decisions = []
    for sample in samples:
        if sample >= threshold:
            decisions.append(decision1)
        if sample < threshold:
            decisions.append(decision2)
    return decisions

def accuracy(ground_truth, responses):
    i = 0
    accuracy = []
    for decision in responses:
        if ground_truth[i] == decision:
            accuracy.append(1)
            print(accuracy)
        if ground_truth[i] != decision:
            accuracy.append(0)
            print(accuracy)
        i += 1

    mean_accuracy = np.mean(accuracy)
    return accuracy, mean_accuracy
