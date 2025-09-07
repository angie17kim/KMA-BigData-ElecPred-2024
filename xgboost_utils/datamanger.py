import numpy as np
import cupy as cp
from copy import deepcopy

from utils.data_utils import NPZLoader
from utils.eval_utils import load_inverse_transform
    
class XGBoostDataManager:
    def __init__(self, data_config, data_path, cupy=False):

        self.npzloader = NPZLoader(data_config, data_path)
        self.data_config = data_config
        self.tfopt = data_config['option_tf']
        self.data_path = data_path
        self.cupy = cupy

        self.float16 = data_config['float16']
        self.arrtype = {
            (False, False): np.float32,
            (False, True): cp.float32,
            (True, False): np.float16,
            (True, True): cp.float16
        }[(self.float16, self.cupy)]
        
        self._set_target_info()
        self._init_process()

    def _set_target_info(self):
        self.target_opt = self.data_config['target_opt']
        # TODO: Add support for other target options
        assert self.target_opt == 'elec_lognorm', "XGBoost: Only 'elec_lognorm' target is supported"
        self.y_inverse_transform = load_inverse_transform(self.target_opt, self.data_path)

    def _init_process(self):
        self.npzloader.process()
        self.train_category_order = deepcopy(self.npzloader.train_npz_order)
        self.test_category_order  = deepcopy(self.npzloader.test_npz_order)
        self.ordered_var_list     = deepcopy(self.npzloader.ordered_var_list)
        self.train_input, self.train_target, self.train_category, self.train_dummy_flag = self.npzloader.get_train_data()
        self.train_years = (self.train_category // 100000).astype(int)
        self.test_input, self.test_category, self.test_dummy_flag = self.npzloader.get_test_data()

        assert np.all(self.train_dummy_flag == 0), "XGBoost: Dummy flag must be zero for all training data"
        assert np.all(self.test_dummy_flag == 0), "XGBoost: Dummy flag must be zero for all test data"

        self.train_input = self._retype(self.train_input)
        self.train_target = self._retype(self.train_target)
        self.test_input = self._retype(self.test_input)

    def get_yearwise_mask(self):
        assert self.tfopt == False, "XGBoost: Dummy Flag should be considered for TFOpt"
        yearwise_mask = {}
        for val_year in [2020, 2021, 2022]:
            train_years = set([2020, 2021, 2022]) - {val_year}

            validation_mask = self.train_years == val_year
            train_mask = np.isin(self.train_years, list(train_years))
            yearwise_mask[val_year] = (train_mask, validation_mask)
        return yearwise_mask

    def get_test_input(self):
        assert self.tfopt == False, "XGBoost: Dummy Flag should be considered for TFOpt"
        return self.test_input

    def _retype(self, arr):
        if self.cupy:
            return cp.asarray(arr, dtype=self.arrtype)
        elif isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr, dtype=self.arrtype)
        else:
            return arr.astype(self.arrtype)
        
    def as_numpy(self, arr):
        if self.cupy:
            return cp.asnumpy(arr)
        else:
            return arr
    
    def get_elec(self, y, categories):
        answer = self.y_inverse_transform(y)
        elec = self._elec_normalize(answer, categories)
        return elec
    
    def get_test_elec(self, y_test):
        return self.get_elec(y_test, self.test_category)
    
    def _elec_normalize(self, y, categories):
        y_elec = np.zeros_like(y)
        for category in np.unique(categories):
            category_mask = categories == category
            y_elec[category_mask] = y[category_mask] / np.mean(y[category_mask])
        y_elec = y_elec * 100.0
        return y_elec