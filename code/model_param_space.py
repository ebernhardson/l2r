# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: model parameter space

"""

import numpy as np
from hyperopt import hp

import config


## xgboost
xgb_random_seed = config.RANDOM_SEED
xgb_nthread = config.NUM_CORES
xgb_n_estimators_min = 100
xgb_n_estimators_max = 1000
xgb_n_estimators_step = 10

## rankpy
rankpy_random_state = config.RANDOM_SEED
rankpy_n_jobs = config.NUM_CORES
rankpy_n_estimators_min = 100
rankpy_n_estimators_max = 1000
rankpy_n_estimators_step = 10

# ---------------------------- XGBoost ---------------------------------------

## pairwse rank with tree booster
param_space_reg_xgb_rank = {
    "booster": "gbtree",
    "objective": "rank:pairwise",
    #"base_score": config.BASE_SCORE,
    "n_estimators" : hp.quniform("n_estimators", xgb_n_estimators_min, xgb_n_estimators_max, xgb_n_estimators_step),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "gamma": hp.loguniform("gamma", np.log(1e-10), np.log(1e1)),
    "reg_alpha" : hp.loguniform("reg_alpha", np.log(1e-10), np.log(1e1)),
    "reg_lambda" : hp.loguniform("reg_lambda", np.log(1e-10), np.log(1e1)),
    "min_child_weight": hp.loguniform("min_child_weight", np.log(1e-10), np.log(1e2)),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "subsample": hp.quniform("subsample", 0.1, 1, 0.05),
    "colsample_bytree": 1,
    "colsample_bylevel": hp.quniform("colsample_bylevel", 0.1, 1, 0.05),
    "nthread": xgb_nthread,
    "seed": xgb_random_seed,
}

# Generated w/ 2k source queries
# Mean weighted Kendall Tau: 0.826609
param_space_reg_xgb_rank_single_best = {
    'booster': 'gbtree',
    'colsample_bylevel': 0.7,
    'colsample_bytee': 1,
    'gamma': 1.69558399913e-08,
    'learning_rate': 0.08,
    'max_depth': 5,
    'min_child_weight': 24.8626131926,
    'n_estimators': 5790,
    'objective': 'rank:pairwise',
    'reg_alpha': 0.0334853384897,
    'reg_lambda': 5.41555940285e-10,
    'subsample': 0.1,
    'nthread': xgb_nthread,
    'seed': xgb_random_seed,
}

param_space_reg_rankpy_lambdamart = {
    'metric': 'ERR@10',
    'n_estimators': hp.quniform("n_estimators", rankpy_n_estimators_min, rankpy_n_estimators_max, rankpy_n_estimators_step),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "max_leaf_nodes": None,
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "shrinkage": hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "subsample": hp.quniform("subsample", 0.1, 1, 0.05),
    "n_jobs": rankpy_n_jobs,
    "random_state": rankpy_random_state,
    "min_samples_leaf": 40,
}

# -------------------------------------- All ---------------------------------------------
param_space_dict = {
    # xgboost
    "reg_xgb_rank": param_space_reg_xgb_rank,
    # rankpy
    "reg_rankpy_lambdamart": param_space_reg_rankpy_lambdamart,
}

int_params = [
    "num_round", "n_estimators", "min_samples_split", "min_samples_leaf",
    "n_neighbors", "leaf_size", "seed", "random_state", "max_depth", "degree",
    "hidden_units", "hidden_layers", "batch_size", "nb_epoch", "dim", "iter", 
    "factor", "iteration", "n_jobs", "max_leaf_forest", "num_iteration_opt", 
    "num_tree_search", "min_pop", "opt_interval",
]
int_params = set(int_params)


class ModelParamSpace:
    def __init__(self, learner_name):
        s = "Wrong learner_name, " + \
            "see model_param_space.py for all available learners."
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

    def _convert_int_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k,v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_int_param(v[i])
                elif isinstance(v, dict):
                    self._convert_int_param(v)
        return param_dict
