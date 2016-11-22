# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for XGBoost models

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import xgboost as xgb


class XGBRank(object):
    def __init__(self, booster='gbtree', base_score=0., colsample_bylevel=1.,
                colsample_bytree=1., gamma=0., learning_rate=0.1, max_delta_step=0.,
                max_depth=6, min_child_weight=1., missing=None, n_estimators=100,
                nthread=1, objective='rank:pairwise', reg_alpha=1., reg_lambda=0.,
                reg_lambda_bias=0., seed=0, silent=True, subsample=1.):
        self.param = {
            "objective": objective,
            "booster": booster,
            "eta": learning_rate,
            "max_depth": max_depth,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "alpha": reg_alpha,
            "lambda": reg_lambda,
            "lambda_bias": reg_lambda_bias,
            "seed": seed,
            "silent": 1 if silent else 0,
            "nthread": nthread,
            "max_delta_step": max_delta_step,
        }
        self.missing = missing if missing is not None else np.nan
        self.n_estimators = n_estimators
        self.base_score = base_score

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ("%s(booster=\'%s\', base_score=%f, colsample_bylevel=%f, \n"
                    "colsample_bytree=%f, gamma=%f, learning_rate=%f, max_delta_step=%f, \n"
                    "max_depth=%d, min_child_weight=%f, missing=\'%s\', n_estimators=%d, \n"
                    "nthread=%d, objective=\'%s\', reg_alpha=%f, reg_lambda=%f, \n"
                    "reg_lambda_bias=%f, seed=%d, silent=%d, subsample=%f)" % (
                    self.__class__.__name__,
                    self.param["booster"],
                    self.base_score,
                    self.param["colsample_bylevel"],
                    self.param["colsample_bytree"],
                    self.param["gamma"],
                    self.param["eta"],
                    self.param["max_delta_step"],
                    self.param["max_depth"],
                    self.param["min_child_weight"],
                    str(self.missing),
                    self.n_estimators,
                    self.param["nthread"],
                    self.param["objective"],
                    self.param["alpha"],
                    self.param["lambda"],
                    self.param["lambda_bias"],
                    self.param["seed"],
                    self.param["silent"],
                    self.param["subsample"],
                ))

    def _calc_groups(self, ids):
        # transform id list into groups format needed by xgboost
        groups = []
        group = None
        seen = set()
        for id in ids:
            if group is None:
                group = id
                count = 1
            elif group == id:
                count += 1
            else:
                assert id not in seen
                seen.add(id)
                groups.append(count)
                group = id
                count = 1
        if count > 0:
            groups.append(count)
        return groups

    def fit(self, X, y, ids, weight=None, feature_names=None):
        data = xgb.DMatrix(X, label=y, weight=weight, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score*np.ones(X.shape[0]))
        data.set_group(self._calc_groups(ids))
        self.model = xgb.train(self.param, data, self.n_estimators)
        return self

    def predict(self, X, ids, weight=None, feature_names=None):
        data = xgb.DMatrix(X, weight=weight, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score*np.ones(X.shape[0]))
        data.set_group(self._calc_groups(ids))
        y_pred = self.model.predict(data)
        return y_pred

    def plot_importance(self):
        ax = xgb.plot_importance(self.model)
        self.save_topn_features()
        return ax

    def save_topn_features(self, fname="XGBRegressor_topn_features.txt", topn=-1):
        ax = xgb.plot_importance(self.model)
        yticklabels = ax.get_yticklabels()[::-1]
        if topn == -1:
            topn = len(yticklabels)
        else:
            topn = min(topn, len(yticklabels))
        with open(fname, "w") as f:
            for i in range(topn):
                f.write("%s\n"%yticklabels[i].get_text())

