from pathlib import Path
import os, sys
import gc
import numpy as np
import cupy as cp
import pandas as pd

import utils.utils as utils
global config
config = utils.get_configs('base')
xgboost_config = config['xgboost']

sys.path.append(config['paths']['repo_path'])
seed = config['seed']
utils.set_seed(seed)
device = config['device']
if 'cuda:' in device:
    os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[-1]
else:
    raise ValueError('Only CUDA devices are supported')
output_path = utils.set_output_path(config)
logger = utils.get_logger('xgboost_cuda', output_path, add_stream_handler=True)

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
# Custom utility imports
from xgboost_utils.datamanger import XGBoostDataManager
from xgboost_utils.callbacks import CustomKFoldCallback
from xgboost_utils.metrics import FocusMSE, FocusEvalMetric, pearson_eval

XGBOOST_PARAMS = ['booster', 'eta', 'gamma', 'alpha', 'lambda', 'max_depth', 'min_child_weight', 'colsample_bytree']
HYPER_PARAMS   = ['n_estimators', 'early_stopping_rounds', 'highslope']

class OptunaManager:
    def __init__(self, train_func):
        self.n_trials = xgboost_config['n_trials']
        self.train_func = train_func

        self._initialize_parameters()
        self._param_assertions()
        self.trial_additional_infos = {
            'metric_valyear_2020': [],
            'metric_valyear_2021': [],
            'metric_valyear_2022': [],
            'best_round_valyear_2020': [],
            'best_round_valyear_2021': [],
            'best_round_valyear_2022': []
        }
        logger.info('OptunaManager initialized')

    def _param_assertions(self):
        pass

    def _initialize_parameters(self):
        """Initializes default and optimized hyperparameters."""
        self.param_default  = xgboost_config['default']
        self.param_optimize = xgboost_config['optimize']
        self.param_suggest  = xgboost_config['suggest']
        # 'objective': 'reg:squarederror',
        # 'eval_metric': 'rmse',
        self.xgboost_params_default = {
            'tree_method': 'hist',
            'device': 'cuda'
        }
        # Set default hyperparameters for xgboost_params
        self.xgboost_params_default.update(zip(XGBOOST_PARAMS, [self.param_default[key] for key in XGBOOST_PARAMS]))
        self.hyper_params_default = dict(zip(HYPER_PARAMS, [self.param_default[key] for key in HYPER_PARAMS])) 

    def _get_default_params(self):
        return self.hyper_params_default.copy(), self.xgboost_params_default.copy()

    def _trial_params(self, trial):
        hyper_params, xgboost_params = self._get_default_params()
        for key in HYPER_PARAMS:
            if self.param_optimize[key]:
                hyper_params[key] = self._suggest_params(trial, key)
        for key in XGBOOST_PARAMS:
            if self.param_optimize[key]:
                xgboost_params[key] = self._suggest_params(trial, key)
        return hyper_params, xgboost_params

    def _suggest_params(self, trial, key):
        param = self.param_suggest[key]
        if param['type'] == 'categorical':
            return trial.suggest_categorical(key, param['choices'])
        elif param['type'] == 'int':
            return trial.suggest_int(key, *param['range'], step=param.get('step', 1))
        elif param['type'] == 'float':
            return trial.suggest_float(key, *param['range'], log=param.get('log', False))
    
    ###
    def run_optuna_search(self):
        self.study = optuna.create_study(direction='maximize', sampler=TPESampler(), study_name='xgboost_optuna')
        self.study.optimize(self.action, n_trials=self.n_trials)
        
        logger.info('Optuna search finished')
        logger.info(f'Saving trials to {output_path}')
        self._save_trials()

        best_trial = self.study.best_trial
        final_hyper_params, final_xgboost_params = self._trial_params(best_trial)
        logger.info(f'Best trial: {best_trial.value}')
        logger.info(f'Best params: {best_trial.params}')

        return final_hyper_params, final_xgboost_params

    def action(self, trial):
        """Objective function for Optuna optimization."""
        hyper_params, xgboost_params = self._trial_params(trial)
        logger.info(f'Trial-{trial.number+1:04d} for params: {trial.params} started')

        trial_num = trial.number + 1
        mention = f'TRIAL: {trial_num:03d}/{self.n_trials:03d}' 
        metric_valyear, best_round_valyear = self.train_func(hyper_params, xgboost_params, mention)
        for valyear in [2020, 2021, 2022]:
            self.trial_additional_infos[f'metric_valyear_{valyear}'].append(metric_valyear[valyear])
            self.trial_additional_infos[f'best_round_valyear_{valyear}'].append(best_round_valyear[valyear])

        logger.info(f'Trial-{trial_num:04d} for params: {trial.params} finished.')
        logger.info(f'Pearson R with Val Years: 2020-{metric_valyear[2020]:.4f}, 2021-{metric_valyear[2021]:.4f}, 2022-{metric_valyear[2022]:.4f}')
        obj_metric = np.mean(list(metric_valyear.values()))
        logger.info(f'Objective as Avg: {obj_metric:.4f}')
        return obj_metric
    
    def _save_trials(self):
        trials_df = self.study.trials_dataframe()
        for key, values in self.trial_additional_infos.items():
            trials_df[key] = values
        trials_df.to_csv(Path(output_path, 'optuna_trials.csv'), index=False)

class XGBoostTrainer:
    def __init__(self):
        self._initialize_environment()
        self._initialize_datastream()
        logger.info('XGBoostTrainer initialized')

    def _initialize_environment(self):        
        self.xgboost_data_manager = XGBoostDataManager(config['data'], config['paths']['data_path'], cupy=True)
        self.feature_names = self.xgboost_data_manager.ordered_var_list

        self.n_trials = xgboost_config['n_trials']
        self.KFold_callback = CustomKFoldCallback(xgboost_config)

    def _initialize_datastream(self):
        """Initializes the data manager and prepares the data for K-fold cross-validation."""
        self.yearwise_mask = self.xgboost_data_manager.get_yearwise_mask()
        self.QuantileDMatrix_list = {}
        for val_year in [2020, 2021, 2022]:
            train_mask, val_mask = self.yearwise_mask[val_year]

            X_train = self.xgboost_data_manager.train_input[train_mask]
            Y_train = self.xgboost_data_manager.train_target[train_mask]
            X_val   = self.xgboost_data_manager.train_input[val_mask]
            Y_val   = self.xgboost_data_manager.train_target[val_mask]
            C_val   = self.xgboost_data_manager.train_category[val_mask]

            dtrain = xgb.QuantileDMatrix(X_train, label=Y_train)
            dval = xgb.QuantileDMatrix(X_val, label=Y_val)
            self.QuantileDMatrix_list[val_year] = (dtrain, dval, cp.asnumpy(Y_val), C_val)

        self.dtest = xgb.QuantileDMatrix(self.xgboost_data_manager.get_test_input(), label=None)
        
    def _calculate_pearson(self, Y_val, Y_pred, C_val, return_groupwise=False):
        elec_val = self.xgboost_data_manager.get_elec(Y_val, C_val)
        elec_pred = self.xgboost_data_manager.get_elec(Y_pred, C_val)
        return pearson_eval(elec_val, elec_pred, C_val, return_groupwise=return_groupwise)

    def focus_mse(self, y_pred, dtrain):
        return FocusMSE(y_pred, dtrain, self.current_high_slope)

    def focus_eval_metric(self, y_pred, dtrain):
        return FocusEvalMetric(y_pred, dtrain, self.current_high_slope)
    
    def _get_xgbtrain_args(self, hyper_params, xgboost_params, dtrain, dval):
        self.current_high_slope = hyper_params['highslope']
        xgbtrain_args = {
            'params': xgboost_params,
            'dtrain': dtrain,
            'num_boost_round': hyper_params['n_estimators'],
            'evals': [(dval, 'test')],
            'early_stopping_rounds': hyper_params['early_stopping_rounds'],
            'verbose_eval': False,
            'callbacks': [self.KFold_callback],
            'obj': self.focus_mse,
            'custom_metric': self.focus_eval_metric
        }
        return xgbtrain_args

    def train(self, hyper_params, xgboost_params, mention):
        metric_valyear = {}
        best_round_valyear = {}
        for val_year in [2020, 2021, 2022]:
            self.KFold_callback.set_callback(
                mention = mention,
                exp_desc = f'Val Year {val_year}'
            )
            dtrain, dval, Y_val, C_val = self.QuantileDMatrix_list[val_year]
            model = xgb.train(
                **self._get_xgbtrain_args(hyper_params, xgboost_params, dtrain, dval)
            )
            Y_pred  = cp.asnumpy(model.predict(dval))
            r_score = self._calculate_pearson(Y_val, Y_pred, C_val)

            metric_valyear[val_year] = r_score
            best_round_valyear[val_year] = model.best_iteration + 1

        return metric_valyear, best_round_valyear

    def evaluation(self, hyper_params, xgboost_params, mention='Final Eval'):        
        FE_answers = {}
        FE_info = {
            'best_round': {},
            'rmse': {},
            'pearson': {}
        }
        FE_feature_importances = {}
        FE_category_pearson = {}

        for val_year in [2020, 2021, 2022]:
            logger.info(f'Final Evaluation for EXP: Val Year {val_year} started')
            self.KFold_callback.set_callback(mention = mention, 
                                             exp_desc = f'Val Year {val_year}')
            dtrain, dval, Y_val, C_val = self.QuantileDMatrix_list[val_year]
            model = xgb.train(
                **self._get_xgbtrain_args(hyper_params, xgboost_params, dtrain, dval)
            )

            Y_pred  = cp.asnumpy(model.predict(dval))
            r_score, r_score_by_category = self._calculate_pearson(Y_val, Y_pred, C_val, return_groupwise=True)
            FE_answers[f'val_year_{val_year}_validation_answer'] = self.xgboost_data_manager.get_elec(Y_pred, C_val)
            answer_test_pred = self.xgboost_data_manager.get_test_elec(cp.asnumpy(model.predict(self.dtest)))
            FE_answers[f'val_year_{val_year}_test_answer'] = answer_test_pred

            FE_feature_importances[f'val_year_{val_year}_gain'] = model.get_score(importance_type='gain')
            FE_feature_importances[f'val_year_{val_year}_weight'] = model.get_score(importance_type='weight')

            FE_info['best_round'][val_year] = model.best_iteration + 1
            FE_info['rmse'][val_year] = model.best_score
            FE_info['pearson'][val_year] = r_score

            FE_category_pearson.update(r_score_by_category)

        del self.QuantileDMatrix_list

        X_total_train = self.xgboost_data_manager.train_input
        Y_total_Train = self.xgboost_data_manager.train_target
        dtotaltrain = xgb.QuantileDMatrix(X_total_train, label=Y_total_Train)

        self.current_high_slope = hyper_params['highslope']
        max_best_round = np.max(list(FE_info['best_round'].values()))
        logger.info('Final Evaluation for total train data started')
        logger.info(f'Set boost round for final eval with toal data as the maximum of best rounds: {max_best_round}')

        model = xgb.train(
            params=xgboost_params,
            dtrain=dtotaltrain,
            num_boost_round=max_best_round,
            verbose_eval=False,
            obj=self.focus_mse
        )
        answer_test_total_pred = self.xgboost_data_manager.get_test_elec(cp.asnumpy(model.predict(self.dtest)))
        FE_answers['total_test_answer'] = answer_test_total_pred
        del dtotaltrain, self.dtest

        # feature importance: index: feature name, columns: year
        rename_dict = dict(zip([f'f{idx}' for idx in range(len(self.feature_names))], self.feature_names))
        feature_importance_df = pd.DataFrame(FE_feature_importances).rename(index=rename_dict).fillna(0.0)

        info_df = pd.DataFrame.from_dict(FE_info, orient='index')

        # Convert the category-wise pearson scores to a DataFrame
        category_pearson_df = pd.DataFrame.from_dict(FE_category_pearson, orient='index', columns=['pearson'])
        category_pearson_df.reset_index(inplace=True)
        category_pearson_df.rename(columns={'index': 'yearnum'}, inplace=True)
        category_pearson_df['val_year'] = category_pearson_df['yearnum'] // 100000
        category_pearson_df['num'] = category_pearson_df['yearnum'] % 100000
        category_pearson_df.drop(columns=['yearnum'], inplace=True)
        category_pearson_df = category_pearson_df[['val_year', 'num', 'pearson']]

        return FE_answers, feature_importance_df, info_df, category_pearson_df

if __name__ == '__main__':

    trainer  = XGBoostTrainer()
    optuna_manager = OptunaManager(train_func=trainer.train)

    if config['xgboost']['optuna_search']:
        logger.info('Optuna search started')
        final_hyper_params, final_xgboost_params = optuna_manager.run_optuna_search()
    else:
        logger.info('Optuna search skipped, using default parameters')
        final_hyper_params, final_xgboost_params = optuna_manager._get_default_params()

    logger.info('Generating test prediction with final parameters') 
    answers_dict, feature_importance_df, info_df, category_pearson_df = trainer.evaluation(final_hyper_params, final_xgboost_params)

    logger.info(f'Saving predictions to {output_path}')
    np.savez(Path(output_path, 'answers.npz'), **answers_dict)
    feature_importance_df.to_csv(Path(output_path, 'feature_importances.csv'))
    info_df.to_csv(Path(output_path, 'eval_info.csv'))
    category_pearson_df.to_csv(Path(output_path, 'category_pearson.csv'), index=False)

    logger.info('XGBoost training finished')
    gc.collect()