import os
import sys
import imp
from optparse import OptionParser

import pandas as pd
import numpy as np

import config
from utils import logging_utils, table_utils, np_utils, time_utils

class Combiner(object):
    def __init__(self, feature_dict, feature_name, corr_threshold=0):
        self.feature_dict = feature_dict
        self.feature_name = feature_name
        self.corr_threshold = corr_threshold

        self.feature_names_basic = []
        self.feature_names = []
        logname = "feature_combiner_%s.log" % (feature_name)
        self.logger = logging_utils._get_logger(config.LOG_DIR, logname)

    def load_feature(self, feature_dir, feature_name):
        fname = os.path.join(feature_dir, feature_name + config.FEAT_FILE_SUFFIX)
        return table_utils._read(fname)

    def combine(self):
        dfAll = table_utils._read(config.INFO_DATA)
        dfAll_raw = dfAll.copy()
        y = dfAll['relevance'].values

        feat_cnt = 0
        self.logger.info('Run for basic...')
        for file_name in sorted(os.listdir(config.FEAT_DIR)):
            if not config.FEAT_FILE_SUFFIX in file_name:
                continue
            fname = os.path.splitext(file_name)[0]
            if fname not in self.feature_dict:
                continue
            x = self.load_feature(config.FEAT_DIR, fname)
            x = np.nan_to_num(x)
            # Still necessary?
            if np.isnan(x).any():
                self.logger.info("%s nan" % (fname))
                continue
            # Apply feature transformers (?)
            mandatory = self.feature_dict[fname][0]
            transformer = self.feature_dict[fname][1]
            x = transformer.fit_transform(x)
            dim = np_utils._dim(x)
            if dim == 1:
                corr = np_utils._corr(x, y)
                if not mandatory and abs(corr) < self.corr_threshold:
                    self.logger.info("Drop: {} ({}D) (abs_corr = {}, < threshold {})".format(
                        fname, dim, abs(corr), self.corr_threshold))
                    continue
                dfAll[fname] = x
                self.feature_names.append(fname)
            else:
                columns = ["%s_%d" % (fname, x) for x in range(dim)]
                df = pd.DataFrame(x, columns=columns)
                dfAll = pd.concat([dfAll, df], axis=1)
                self.feature_names.extend(columns)
            feat_cnt += 1
            self.feature_names_basic.append(fname)
            if dim == 1:
                self.logger.info("Combine {:>3}/{:>3} feat: {} ({}D) (corr = {})".format(
                    feat_cnt, len(self.feature_dict.keys()), fname, dim, corr))
            else:
                self.logger.info("Combine {:>3}/{:>3} feat: {} ({}D)".format(
                    feat_cnt, len(self.feature_dict.keys()), fname, dim))

        dfAll.fillna(config.MISSING_VALUE_NUMERIC, inplace=True)
        self.y = dfAll["relevance"].values.astype(float)
        self.weights = dfAll['weight'].values
        dfAll.drop(["relevance", "weight"], axis=1, inplace=True)
        self.X = dfAll.values.astype(float)

        self.logger.info("Overall Shape: %d x %d" % (len(self.y), self.X.shape[1]))
        self.logger.info("Done combining")

    def save(self):
        data_dict = {
            "X": self.X,
            "y": self.y,
            "weights": self.weights,
            "feature_names": self.feature_names,
        }
        fname = os.path.join(config.FEAT_DIR, 'Combine', self.feature_name + config.FEAT_FILE_SUFFIX)
        table_utils._write(fname, data_dict)
        self.logger.info("Save to %s" % (fname))

def main(options):
    feature_conf = imp.load_source("", os.path.join(config.FEAT_CONF_DIR, options.feature_conf + '.py'))
    combiner = Combiner(feature_dict=feature_conf.feature_dict,
                        feature_name=options.feature_name,
                        corr_threshold=options.corr_threshold)
    combiner.combine()
    combiner.save()

def parse_args(parser):
    parser.add_option('-c', '--config', default='feature_conf', type='string',
        dest='feature_conf', help='feature config name')
    parser.add_option('-n', '--name', default='basic%s' % (time_utils._timestamp()),
        type='string', dest='feature_name', help='feature name')
    parser.add_option('-t', '--threshold', default=0.0, type='float',
        dest='corr_threshold', help='correlation threshold for dropping features')
    return parser.parse_args()

if __name__ == "__main__":
    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)





