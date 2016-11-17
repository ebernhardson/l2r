import sys

import numpy as np
import math
from scipy.stats import pearsonr
from collections import Counter

sys.path.append("..")
import config

def _mean(x):
    idx = np.isfinite(x)
    if sum(idx) == 0:
        return float(config.MISSING_VALUE_NUMERIC)
    else:
        return np.mean(x[idx])

def _corr(x, y_train):
    if _dim(x) == 1:
        corr = pearsonr(x.flatten(), y_train)[0]
        if str(corr) == "nan":
            corr = 0.
    else:
        # Why?
        corr = 1.
    return corr

def _dim(x):
    # Why isn't this len(x.shape)?
    return 1 if len(x.shape) == 1 else x.shape[1]

def _entropy(proba):
    return -np.sum(proba * np.log(proba))

def _try_divide(x, y, val=0.0):
    return val if y == 0.0 else float(x) / y

def _split(x, size):
    return np.array_split(x, math.ceil(len(x)/float(size)))
