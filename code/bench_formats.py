import pandas as pd
import feather
import os
import timeit

import config
from utils import table_utils

df = table_utils._read(config.ALL_DATA)

FILE_HDF = os.path.join(config.TMP_DIR, 'test.h5')
FILE_PICKLE = os.path.join(config.TMP_DIR, 'test.pkl')
FILE_FEATHER = os.path.join(config.TMP_DIR, 'test.feather')

def test_hdf_write():
    df.to_hdf(FILE_HDF, 'test', mode='w')

def test_hdf_read():
    pd.read_hdf(FILE_HDF, 'test')

def test_pickle_write():
    df.to_pickle(FILE_PICKLE)

def test_pickle_read():
    pd.read_pickle(FILE_PICKLE)

def test_feather_write():
    feather.write_dataframe(df.copy(), FILE_FEATHER)

def test_feather_read():
    feather.read_dataframe(FILE_FEATHER)


def test(func):
    took = timeit.timeit("%s()" % (func.__name__), setup="from __main__ import %s" % (func.__name__), number=3)
    print "%s: %.3f" % (func.__name__, took)

if __name__ == "__main__":
    res = []
    res.append(test(test_hdf_write))
    res.append(test(test_hdf_read))
    res.append(test(test_pickle_write))
    res.append(test(test_pickle_read))
    res.append(test(test_feather_write))
    res.append(test(test_feather_read))
    print "\n\n\n"
    print "\n".join(res)
