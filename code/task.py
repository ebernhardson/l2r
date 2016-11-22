# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: definitions for
        - learner & ensemble learner
        - feature & stacking feature
        - task & stacking task
        - task optimizer

"""

import os
import sys
import time
from optparse import OptionParser
from itertools import chain

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Lasso, Ridge, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from rankpy.metrics._metrics import KendallTau

import config
from utils import dist_utils, logging_utils, table_utils, time_utils
from utils.xgb_utils import XGBRank
from utils.rankpy_utils import LambdaMART

try:
    from utils.keras_utils import KerasDNNRegressor
except:
    pass
from model_param_space import ModelParamSpace


class Learner:
    def __init__(self, learner_name, param_dict):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = self._get_learner()

    def __str__(self):
        return self.learner_name

    def _get_learner(self):
        # xgboost
        if self.learner_name == "reg_xgb_rank":
            return XGBRank(**self.param_dict)
        # ranky
        if self.learner_name == "reg_rankpy_lambdamart":
            return LambdaMART(**self.param_dict)
        # TODO: https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/
        # TODO: http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
        return None

    def fit(self, X, y, ids, weight=None, feature_names=None):
        if feature_names is not None:
            self.learner.fit(X, y, ids, weight, feature_names)
        else:
            self.learner.fit(X, y, ids, weight)
        return self

    def predict(self, X, ids, weight=None, feature_names=None):
        if feature_names is not None:
            y_pred = self.learner.predict(X, ids, weight, feature_names)
        else:
            y_pred = self.learner.predict(X, ids, weight)
        return y_pred

    def plot_importance(self):
        ax = self.learner.plot_importance()
        return ax


class Feature:
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.data_dict = self._load_data_dict()

    def __str__(self):
        return self.feature_name

    def _load_data_dict(self):
        fname = os.path.join(config.FEAT_DIR+"/Combine", self.feature_name+config.FEAT_FILE_SUFFIX)
        data_dict = table_utils._read(fname)
        return data_dict

    ## for refit
    def _get_train_test_data(self):
        # feature
        X = self.data_dict["X"]
        y = self.data_dict["y"]
        w = self.data_dict["weights"]
        ids = self.data_dict["query_ids"]
        # split into train/test. Leaves out 30% of queries.
        # TODO: Use all 3 splits for CV.
        splitter = GroupKFold(n_splits=3)
        train, test = next(splitter.split(X, None, ids))

        # TODO: split in feature_combiner
        X_train, X_test, y_train, y_test, ids_train, ids_test, w_train, w_test = chain.from_iterable(
            ((a[train], a[test]) for a in [X, y, w, ids]))

        return X[train], y[train], w[train], ids[train], X[test], y[test], w[test], ids[test]

    ## for feature importance
    def _get_feature_names(self):
        return self.data_dict["feature_names"]


class Task:
    def __init__(self, learner, feature, suffix, logger, verbose=True, plot_importance=False):
        self.learner = learner
        self.feature = feature
        self.suffix = suffix
        self.logger = logger
        self.verbose = verbose
        self.plot_importance = plot_importance
        self.r2 = 0

    def __str__(self):
        return "[Feat@%s]_[Learner@%s]%s"%(str(self.feature), str(self.learner), str(self.suffix))

    def _print_param_dict(self, d, prefix="      ", incr_prefix="      "):
        for k,v in sorted(d.items()):
            if isinstance(v, dict):
                self.logger.info("%s%s:" % (prefix,k))
                self._print_param_dict(v, prefix+incr_prefix, incr_prefix)
            else:
                self.logger.info("%s%s: %s" % (prefix,k,v))

    def fit(self):
        X_train, y_train, w_train, ids_train, X_test, y_test, w_test, ids_test = self.feature._get_train_test_data()
        if self.plot_importance:
            feature_names = self.feature._get_feature_names()
            self.learner.fit(X_train, y_train, ids_train, w_train, feature_names)
            y_pred = self.learner.predict(X_test, ids_test, w_test, feature_names)
        else:
            self.learner.fit(X_train, y_train, ids_train, w_train)
            y_pred = self.learner.predict(X_test, ids_test, w_test)

        # Compare y_pred vs y_test
        taus = []
        # Weighted kendall tau
        metric = KendallTau(lambda i: -1 / np.log2(i + 2))
        qweight_sum = 0
        for i in np.unique(ids_test):
            # probably a better way to do this...
            condition = ids_test == i
            # I think argsort is all we need...
            yi_test = np.argsort(y_test[condition])
            yi_pred = np.argsort(y_pred[condition])
            tau = metric.distance(yi_test, yi_pred)
            taus.append(tau)
        self.mean_tau = np.mean(taus)
        self.std_tau = np.std(taus)
        self.logger.info("[%s] Mean Kendalls Tau: %f" % (self.__str__(), self.mean_tau))
        self.logger.info("[%s] Std Kendalls Tau: %f" % (self.__str__(), self.std_tau))

        # plot importance
        if self.plot_importance:
            ax = self.learner.plot_importance()
            ax.figure.savefig("%s/%s.pdf"%(config.FIG_DIR, self.__str__()))
        return self

    def go(self):
        self.fit()
        return self


class TaskOptimizer:
    def __init__(self, learner_name, feature_name, logger,
                    max_evals=100, verbose=True, plot_importance=False):
        self.learner_name = learner_name
        self.feature_name = feature_name
        self.feature = self._get_feature()
        self.logger = logger
        self.max_evals = max_evals
        self.verbose = verbose
        self.plot_importance = plot_importance
        self.trial_counter = 0
        self.model_param_space = ModelParamSpace(self.learner_name)

    def _get_feature(self):
        return Feature(self.feature_name)

    def _obj(self, param_dict):
        self.trial_counter += 1
        param_dict = self.model_param_space._convert_int_param(param_dict)
        learner = Learner(self.learner_name, param_dict)
        suffix = "_[Id@%s]"%str(self.trial_counter)
        self.task = Task(learner, self.feature, suffix, self.logger, self.verbose, self.plot_importance)
        self.task.go()
        ret = {
            "loss": 1. - self.task.mean_tau,
            "attachments": {
                "std_tau": self.task.std_tau,
            },
            "status": STATUS_OK,
        }
        return ret

    def run(self):
        start = time.time()
        trials = Trials()
        best = fmin(self._obj, self.model_param_space._build_space(), tpe.suggest, self.max_evals, trials)
        best_params = space_eval(self.model_param_space._build_space(), best)
        best_params = self.model_param_space._convert_int_param(best_params)
        # To turn this into a loss function these are actually 1 - tau,
        # converting back is same
        trial_mean_taus = 1 - np.asarray(trials.losses(), dtype=float)
        best_ind = np.argmin(trial_mean_taus)
        best_mean_tau = trial_mean_taus[best_ind]
        self.logger.info("-"*50)
        self.logger.info("Best Mean Kendalls Tau: %.6f" % (best_mean_tau))
        self.logger.info("Best param")
        self.task._print_param_dict(best_params)
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        self.logger.info("Time")
        if _min > 0:
            self.logger.info("      %d mins"%_min)
        else:
            self.logger.info("      %d secs"%_sec)
        self.logger.info("-"*50)


#------------------------ Main -------------------------
def main(options):
    logname = "[Feat@%s]_[Learner@%s]_hyperopt_%s.log"%(
        options.feature_name, options.learner_name, time_utils._timestamp())
    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    optimizer = TaskOptimizer(options.learner_name,
        options.feature_name, logger, options.max_evals, verbose=True,
        plot_importance=options.plot_importance)
    optimizer.run()

def parse_args(parser):
    parser.add_option("-f", "--feat", type="string", dest="feature_name",
        help="feature name", default="basic")
    parser.add_option("-l", "--learner", type="string", dest="learner_name",
        help="learner name", default="reg_skl_ridge")
    parser.add_option("-e", "--eval", type="int", dest="max_evals",
        help="maximun number of evals for hyperopt", default=100)
    parser.add_option("-p", default=False, action="store_true", dest="plot_importance",
        help="plot feautre importance (currently only for xgboost)")

    (options, args) = parser.parse_args()
    return options, args


if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)
