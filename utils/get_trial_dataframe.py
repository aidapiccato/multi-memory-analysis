import json
import numpy as np
import numpy as np
import copy
from utils import constants
import pandas as pd

_X_INDEX = constants.ATTRIBUTES_PARTIAL_INDICES['x']
_Y_INDEX = constants.ATTRIBUTES_PARTIAL_INDICES['y']
_METADATA_INDEX_FULL = np.argwhere(
    [x == 'metadata' for x in constants.ATTRIBUTES_FULL])[0][0]

def _get_angle(x, y):
    x = x - 0.5
    y = y - 0.5
    return np.arctan2(x, y)    

def _get_trial_data(trial, trial_num):

    init_state, init_meta_state = trial[0]
    stim = init_meta_state['stim']

    d = {'trial_num': trial_num, 'time': trial[1][0][1]}
    for i in range(6):
        d[f'object_{i}_x'] = None
        d[f'object_{i}_y'] = None
        d[f'object_{i}_id'] = None
        d[f'object_{i}_theta'] = None
    for k, v in init_state:
        if k == 'object':
            for i, p_v in enumerate(v):
                s = 'object_' + str(i)
                d[s + '_x'] = p_v[_X_INDEX]
                d[s + '_y'] = p_v[_Y_INDEX]
                d[s + '_id'] = p_v[_METADATA_INDEX_FULL]['id']
                d[s + '_theta'] = _get_angle(p_v[_X_INDEX], p_v[_Y_INDEX])
                if p_v[_METADATA_INDEX_FULL]['target']:
                    d['target_id'] = d[s + '_id']
                    d['target_x'] = d[s + '_x']
                    d['target_y'] = d[s + '_y']      
                    d['target_theta'] = _get_angle(d['target_x'], d['target_y'])                          
            d['num_object'] = i + 1

    d['response_object_ind'] = None
    d['response_x'] = None
    d['response_id'] = None
    d['response_theta'] = None
    d['correct'] = None

    done_response = False
    response = False


    d['phase_fixation_time'] = None
    d['phase_visible_time'] = None
    d['phase_delay_time'] = None
    d['phase_cue_time'] = None
    d['phase_response_time'] = None
    d['final_phase'] = None
    for step_i, t in enumerate(trial[1:]):
        meta_state = t[constants.META_STATE_INDEX][1]
        phase = meta_state['phase']
        curr_time = t[constants.TIME_INDEX][1]

        if d['phase_fixation_time'] is None and phase == 'fixation':
            d['phase_fixation_time'] = curr_time
            d['final_phase'] = 'fixation'
        if d['phase_visible_time'] is None and phase == 'visible':
            d['phase_visible_time'] = curr_time
            d['final_phase'] = 'visible'
        if d['phase_delay_time'] is None and phase == 'delay':
            d['phase_delay_time'] = curr_time
            d['final_phase'] = 'delay'
        if d['phase_cue_time'] is None and phase == 'cue':
            d['phase_cue_time'] = curr_time
            d['final_phase'] = 'cue'
        if d['phase_response_time'] is None and phase == 'response':
            d['phase_response_time'] = curr_time
            d['final_phase'] = 'response'

        if phase == 'response' and not done_response:
            response_step = copy.copy(step_i)
            done_response = True
        
        # Get response
        if phase == 'reveal' and not response:
            if meta_state['response'] is not None:

                d['reaction_time_steps'] = step_i - response_step

                d['response_object_ind'] = meta_state['response_object']
                d['correct'] = meta_state['success']

                if d['response_object_ind'] is None:
                    d['response_id'] = None
                    d['response_x'] = None
                    d['response_y'] = None
                    d['response_theta'] = None
                else:

                    d['response_id'] = d[
                        'object_' + str(d['response_object_ind']) + '_id']
                    d['response_x'] = meta_state['response'][0]                                                            
                    d['response_y'] = meta_state['response'][1]
                    d['response_theta'] = _get_angle(d['response_x'], d['response_y'])
                response = True
    if not response:
        return None
    
    d['visible_s'] = d['phase_delay_time'] - d['phase_visible_time']
    if d['final_phase'] != 'visible':
        d['delay_s'] = d['phase_cue_time'] - d['phase_delay_time']
    elif d['final_phase'] != 'delay':
        d['cue_s'] = d['phase_response_time'] - d['phase_cue_time']
    elif d['final_phase'] != 'cue':
        d['reaction_time_s'] = d['reaction_time_steps'] / 60

    
    # Removing keys we don't need
    for k in ['phase_visible_time', 'phase_delay_time', 'phase_fixation_time', 
              'phase_cue_time', 'phase_response_time', 'reaction_time_steps',
              'final_phase']:
        d.pop(k)

    return d

def get_trial_dataframe(trial_paths):
    trial_dicts = []
    for trial_num, trial_path in enumerate(trial_paths):
        trial = json.load(open(trial_path, 'r'))
        trial_data = _get_trial_data(
            trial, trial_num=trial_num)
        if trial_data is not None:
            trial_dicts.append(trial_data)
    full_data = {
        k: [d[k] for d in trial_dicts]
        for k in trial_dicts[0].keys()
    }
    df = pd.DataFrame(full_data)
    df = df.dropna(subset='response_object_ind')
    
    return df
