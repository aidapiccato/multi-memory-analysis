"""Function to get trial paths from a dataset path."""

import json
import numpy as np
import os
from pathlib import Path


def get_trial_paths(logs_path, datasets):
    """Returns list of paths to trial files for a set of datasets

    Args:
        logs_path (_type_): _description_
        datasets (_type_): _description_

    Returns:
        list: List of trial paths
    """
    logs_path = Path(logs_path)
    dataset_paths = [logs_path / dataset for dataset in datasets]    

    # Chronological trial paths
    trial_paths = []
    for dataset_path in dataset_paths:
        trial_paths += [
            os.path.join(dataset_path, x)
            for x in sorted(os.listdir(dataset_path)) if x.isnumeric()
        ] 

    # Print number of trials
    num_trials = len(trial_paths)
    print(f'Number of trials:  {num_trials}')

    return trial_paths