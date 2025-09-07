import os
from pathlib import Path
import copy
import glob
import re
import numpy as np
import pandas as pd
import utils
import time
from collections import defaultdict

GENERAL_VARIABLES = ['district', 'units_qt', 'lon_mm', 'lat_mm', 'altitude_mm']
TIMECYC_VARIABLES = ['dow', 'dom', 'doy', 'woy', 'month', 'h24', 'routine']
TIMEBIN_VARIABLES = ['weekend', 'holiday']
CLIMATE_VARIABLES = ['temp_st', 'humid_st', 'tchi_st', 'dci_st', 'hi_st', 'wchi_st', 'atemp_st', 'rain_qt', 'wind_mm']      

INPUT_COLS = {
    'district_c0': 0, 'district_c1': 1, 'district_c2': 2, 'district_c3': 3,
    'units_qt': 4, 'lon_mm': 5, 'lat_mm': 6, 'holiday': 7, 'altitude_mm': 8,
    'temp_st': 9, 'humid_st': 10, 'tchi_st': 11, 'dci_st': 12, 'hi_st': 13,
    'wchi_st': 14, 'atemp_st': 15, 'rain_qt': 16, 'wind_mm': 17
}

def split_routine(dt_hour):
    conditions = [
        (dt_hour >= 0) & (dt_hour  <= 5),
        (dt_hour >= 6) & (dt_hour <= 9),
        (dt_hour >= 10) & (dt_hour <= 13),
        (dt_hour >= 14) & (dt_hour <= 17),
        (dt_hour >= 18)
    ]
    choices = [0, 1, 2, 3, 4]
    return np.select(conditions, choices, default=-1)

TIMECYC_LAMBDA_T = {
    'dow': (lambda dt: dt.dayofweek, 7),
    'dom': (lambda dt: dt.day, 31),
    'doy': (lambda dt: dt.dayofyear, 365),
    'woy': (lambda dt: dt.isocalendar().week.clip(upper=51), 51),
    'month': (lambda dt: dt.month, 12),
    'h24': (lambda dt: dt.hour+1, 24),
    'routine': (lambda dt: split_routine(dt.hour), 5)
}

class NPZLoader:

    def __init__(self, data_config, data_path):
        self.data_config = data_config
        # self.tfopt: default false. if true, climlag off, and do not trim dummy data
        self.tfopt = self.data_config['option_tf']
        if self.tfopt:
            print('TF option is on. Climate lag is off, and dummy data is not trimmed.')
        self.data_path = data_path
        self.float16 = self.data_config['float16']
        self.fltype = np.float16 if self.float16 else np.float32

        self.npz_dir_path  = {
            'train': Path(self.data_path, 'NPZ/train'),
            'test' : Path(self.data_path, 'NPZ/test')
        }
        # set input variables for selecting input cols, and timecyc, timebin, climate variables to be processed
        self._initialize_variable()
        # set climate lag interval
        self._set_climate_lag()
        # set total variable ordering
        self._set_total_variable_order()

        # yearnum = year * 100000 + num
        # npz_dict: {yearnum: npz_path}, npz_order: [yearnum1, yearnum2, ...]
        self.train_npz_dict, self.train_npz_order = self._get_npz_path_by_yearnum('train')
        self.test_npz_dict, self.test_npz_order   = self._get_npz_path_by_yearnum('test')

    def process(self):

        self.train_input_dict = {}
        self.train_target_dict = {}
        self.train_is_dummy_dict = {}
        for yearnum, npz_path in self.train_npz_dict.items():
            input_stack, target, is_dummy = self.process_npz_path(npz_path, 'train')
            self.train_input_dict[yearnum] = input_stack
            self.train_target_dict[yearnum] = target
            self.train_is_dummy_dict[yearnum] = is_dummy

        self.test_input_dict = {}
        self.test_is_dummy_dict = {}
        for yearnum, npz_path in self.test_npz_dict.items():
            input_stack, is_dummy = self.process_npz_path(npz_path, 'test')
            self.test_input_dict[yearnum] = input_stack
            self.test_is_dummy_dict[yearnum] = is_dummy

        print('Data processing done')

    def get_train_data(self):
        input_list = []
        target_list = []
        is_dummy_list = []
        category_list = []
        for yearnum in self.train_npz_order:
            if yearnum not in self.train_input_dict: continue
            input_list.append(self.train_input_dict[yearnum])
            target_list.append(self.train_target_dict[yearnum])
            is_dummy_list.append(self.train_is_dummy_dict[yearnum])
            category_list.append(np.array([yearnum] * len(self.train_target_dict[yearnum]), dtype=np.int32))
        input_concat = np.concatenate(input_list, axis=0)
        target_concat = np.concatenate(target_list, axis=0)
        is_dummy_concat = np.concatenate(is_dummy_list, axis=0)
        category_concat = np.concatenate(category_list, axis=0)

        return input_concat, target_concat, category_concat, is_dummy_concat
    
    def get_test_data(self):
        input_list = []
        is_dummy_list = []
        category_list = []
        for yearnum in self.test_npz_order:
            if yearnum not in self.test_input_dict: continue
            input_list.append(self.test_input_dict[yearnum])
            is_dummy_list.append(self.test_is_dummy_dict[yearnum])
            category_list.append(np.array([yearnum] * len(self.test_input_dict[yearnum]), dtype=np.int32))
        input_concat = np.concatenate(input_list, axis=0)
        is_dummy_concat = np.concatenate(is_dummy_list, axis=0)
        category_concat = np.concatenate(category_list, axis=0)

        return input_concat, category_concat, is_dummy_concat

    ###      ###      ###      ###      ###      ###      ###      ###      ###

    def process_npz_path(self, npz_path, mode):
        npz_data = np.load(npz_path)
        input_data = npz_data['input'][..., self.input_col_idx]
        climlag_base_data = npz_data['input'][..., self.input_climlag_idx]
        var_names  = copy.deepcopy(self.input_vars)
        datetime = pd.Series(pd.to_datetime(npz_data['datetime']))
        is_dummy  = npz_data['is_dummy']
        if mode == 'train':
            if self.tfopt:
                target = npz_data['elec'].astype(self.fltype)
            else:
                target = npz_data['elec'][is_dummy == 0].astype(self.fltype)
        del npz_data

        input_data = input_data.astype(self.fltype)

        timecyc_stack = []
        for var in self.timecyc_vars:
            lambda_func, T = TIMECYC_LAMBDA_T[var]
            cyclic_var = lambda_func(datetime.dt)
            cyclic_var_cos = np.cos(2 * np.pi * cyclic_var / T)
            cyclic_var_sin = np.sin(2 * np.pi * cyclic_var / T)

            timecyc_stack.append(cyclic_var_cos)
            timecyc_stack.append(cyclic_var_sin)
            var_names.append(var+'_cos')
            var_names.append(var+'_sin')
        timecyc_stack = np.stack(timecyc_stack, axis=-1).astype(self.fltype)

        timebin_stack = []
        if 'weekend' in self.timebin_vars:
            weekend = datetime.dt.dayofweek // 5
            timebin_stack.append(weekend.astype(self.fltype))
            var_names.append('weekend')
        timebin_stack = np.stack(timebin_stack, axis=-1).astype(self.fltype)

        input_stack = np.concatenate([input_data, timecyc_stack, timebin_stack], axis=-1)
        if self.tfopt:
            is_dummy_return = is_dummy
        else:
            climlag_stack = []
            for climlag_interval, climlag_tag in zip(self.clim_lag_interval, self.clim_lag_tag):
                window_size = climlag_interval[1] - climlag_interval[0] + 1
                lag_min = climlag_interval[0]
                # 각 열에 대해 np.convolve를 적용
                kernel = np.ones(window_size) / window_size
                convolved_data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='full'), axis=0, arr=climlag_base_data)
                climlag_data = convolved_data[np.where(is_dummy == 0)[0] - lag_min, :]
                climlag_stack.append(climlag_data)
                for var in self.climlag_vars:
                    var_names.append(var+'_'+climlag_tag)
            climlag_stack = np.concatenate(climlag_stack, axis=-1).astype(self.fltype)

            input_stack = np.concatenate([input_stack[is_dummy == 0], climlag_stack], axis=-1)
            is_dummy_return = is_dummy[is_dummy == 0] # well, its all 0


        assert set(var_names) == set(self.ordered_var_list), 'Variable names mismatch'
        ordered_idx = [var_names.index(var) for var in self.ordered_var_list]
        input_stack = input_stack[..., ordered_idx]

        if mode == 'train':
            return input_stack, target, is_dummy_return
        else:
            return input_stack, is_dummy_return

    def _get_npz_path_by_yearnum(self, mode):
        assert mode in ['train', 'test'], 'Invalid mode'
        npz_dir_path = self.npz_dir_path[mode]

        npz_dict = {}
        pattern = re.compile(r'(\d+)_(\d+)\.npz')
        for npz_path in glob.glob(str(Path(npz_dir_path, '*.npz'))):
            file_name = os.path.basename(npz_path)
            match = pattern.match(file_name)
            if match:
                num = int(match.group(1))
                year = int(match.group(2))
                yearnum = year * 100000 + num
                npz_dict[yearnum] = npz_path

        npz_order = np.load(Path(npz_dir_path, 'num_year_order.npy'))
        npz_order = [year * 100000 + num for num, year in npz_order]

        return npz_dict, npz_order
    
    ## Below methods are for setting variables

    def _initialize_variable(self):

        exclude_general_vars = set(self.data_config['exclude_general_vars'])
        exclude_timecyc_vars = set(self.data_config['exclude_timecyc_vars'])
        exclude_timebin_vars = set(self.data_config['exclude_timebin_vars'])
        exclude_climate_vars = set(self.data_config['exclude_climate_vars'])

        self.general_vars = list(set(GENERAL_VARIABLES) - exclude_general_vars)
        self.timecyc_vars = list(set(TIMECYC_VARIABLES) - exclude_timecyc_vars)
        self.timebin_vars = list(set(TIMEBIN_VARIABLES) - exclude_timebin_vars)
        self.climate_vars = list(set(CLIMATE_VARIABLES) - exclude_climate_vars)

        if 'district' in exclude_general_vars:
            exclude_general_vars.remove('district')
            exclude_general_vars.extend(['district_c0', 'district_c1', 'district_c2', 'district_c3'])

        exclude_vars = set(exclude_general_vars | exclude_timecyc_vars | exclude_timebin_vars | exclude_climate_vars)
        self.input_vars = list(set(INPUT_COLS.keys()) - exclude_vars)
        self.input_col_idx = [INPUT_COLS[var] for var in self.input_vars]

        exclude_climlag_vars = set(self.data_config['exclude_climlag_vars'])
        self.climlag_vars = list(set(CLIMATE_VARIABLES) - exclude_climlag_vars)
        self.input_climlag_idx = [INPUT_COLS[var] for var in self.climlag_vars]

    def _set_climate_lag(self):
        self.clim_lag_min    = self.data_config['clim_lag_min']
        self.clim_lag_max    = self.data_config['clim_lag_max']
        self.clim_lag_step   = self.data_config['clim_lag_step']
        self.clim_lag_window = self.data_config['clim_lag_window']

        self.clim_lag_interval = np.arange(self.clim_lag_min, self.clim_lag_max - self.clim_lag_window + 2, self.clim_lag_step)
        self.clim_lag_interval = [(x, x+self.clim_lag_window-1) for x in self.clim_lag_interval] # interval is inclusive
        self.clim_lag_tag = ['{}t{}'.format(x, y) for x, y in self.clim_lag_interval]

    def _set_total_variable_order(self):
        self.ordered_var_list = []
        for var in GENERAL_VARIABLES:
            if var not in self.general_vars: continue
            if var == 'district':
                for i in range(4):
                     self.ordered_var_list.append('district_c{}'.format(i))
            else: self.ordered_var_list.append(var)

        for var in TIMEBIN_VARIABLES:
            if var not in self.timebin_vars: continue
            self.ordered_var_list.append(var)

        for var in TIMECYC_VARIABLES:
            if var not in self.timecyc_vars: continue
            for tail in ['_cos', '_sin']:
                self.ordered_var_list.append(var+tail)

        for var in CLIMATE_VARIABLES:
            if var not in self.climate_vars: continue
            self.ordered_var_list.append(var)

            if var not in self.climlag_vars or self.tfopt: continue
            for lag_tag in self.clim_lag_tag:
                self.ordered_var_list.append(var+'_'+lag_tag)


if __name__ == '__main__':
    start_time = time.time()
    config = utils.get_configs('base')

    loader = NPZLoader(config['data'], config['paths']['data_path'])
    loader.process()
    ordered_var_list = loader.ordered_var_list
    train_input, train_target, train_category, _ = loader.get_train_data()
    test_input, test_category, _ = loader.get_test_data()
    
    print()
    print('Ordered variable list:', ordered_var_list)
    print()
    print('Train input data shape:', train_input.shape)
    print('Train target data shape:', train_target.shape)
    print('Train category data shape:', train_category.shape)
    # print train_category example
    print('Train category example:', train_category[:10])

    print()
    print('Test input data shape:', test_input.shape)
    print('Test category data shape:', test_category.shape)

    print()
    print(f'Elapsed time: {time.time() - start_time:0.1f} seconds')