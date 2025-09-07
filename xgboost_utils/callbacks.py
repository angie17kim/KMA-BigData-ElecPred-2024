import numpy as np
import xgboost as xgb
from copy import deepcopy
import gc

class CustomKFoldCallback(xgb.callback.TrainingCallback):
    def __init__(self, xgboost_config):
        self.n_trials = xgboost_config['n_trials']
        self.period = xgboost_config['callback_period']

        self.mention = f'TRIAL: {0:03d}'
        self.exp_desc = 'NONE'

    def set_callback(self, mention, exp_desc):
        self.mention = mention
        self.exp_desc = exp_desc

    def after_iteration(self, model, epoch, evals_log):
        _epoch = epoch + 1
        if _epoch % self.period == 0:
            results = np.asarray(evals_log['test']['rmse'])
            last_result = results[-1]
            best_epoch, best_score = np.argmin(results), np.min(results)
            best_epoch += 1
            print(f'[{self.mention}, EXP: {self.exp_desc}, Epoch: {_epoch:04d}] '
                  f'RMSE: {last_result:.4e} (Best: {best_score:.4e} at [Epoch: {best_epoch:04d}])')
        return False

class CustomEvalCallback(xgb.callback.TrainingCallback):
    def __init__(self, period=20):
        self.period = period
        self.set_callback()

    def reset_callback(self):
        self.best_epoch = 0
        self.best_score = np.inf
        if self.best_model is not None:
            del self.best_model
            gc.collect()

    def get_best_model(self):
        return self.best_model

    def after_iteration(self, model, epoch, evals_log):
        results = np.asarray(evals_log['test']['rmse'])
        new_best_epoch = np.argmin(results) + 1

        if new_best_epoch < self.best_epoch:
            self.best_epoch = new_best_epoch
            self.best_score = np.min(results)
            self.best_model = deepcopy(model) 

        return False