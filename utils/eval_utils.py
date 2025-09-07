import numpy as np
import pickle
from pathlib import Path

def array_format_check_for_inverse_transform(array):
    # Check if the array is 1D
    assert len(array.shape) == 1, f"Invalid array shape: {array.shape}"
    # array is numpy array
    assert isinstance(array, np.ndarray), f"Invalid array type: {type(array)}"

target_dict = {
            'elec': ('elec', 'none'),
            'elec_q': ('elec', 'quantile'),
            'elec_lognorm': ('elec', 'log-standardize'),
            'avg_load': ('avg_load', 'none'),
            'avg_load_q': ('avg_load', 'quantile'),
            'avg_load_lognorm': ('avg_load', 'log-standardize')
        }

def load_inverse_transform(target_opt, transform_path):
    target_var, target_transform = target_dict[target_opt]
    if target_transform == 'none':
        def transform_y(value):
            array_format_check_for_inverse_transform(value)
            return value
    
    elif target_transform == 'quantile':
        with open(Path(transform_path, f'{target_var}_transform.pkl'), 'rb') as f:
            transform_func = pickle.load(f)

        def transform_y(value):
            array_format_check_for_inverse_transform(value)
            value = transform_func.inverse_transform(value.reshape(-1,1))
            value = value.reshape(-1)
            return value

    elif target_transform == 'log-standardize':
        mean, std = {
            'elec': (1.98753, 0.10281),
            'avg_load': (2.34426, 0.19886),
            'sum_load': (3.8141, 0.3772)
        }[target_var]

        def transform_y(value):
            array_format_check_for_inverse_transform(value)
            value = value * std + mean
            value = np.power(10, value)
            return value
        
    else:
        raise ValueError(f"Invalid target_transform: {target_transform}")

    return transform_y