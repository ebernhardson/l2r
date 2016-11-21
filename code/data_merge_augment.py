import pandas as pd

import subprocess

import config
from utils import table_utils

def dfRel():
    return table_utils._read(config.RELEVANCE_DATA)

def shelve_keys(fname):
    # This is pathological for spinning disks if the data isn't already in the
    # kernel disk cache, so try and pull it in
    subprocess.call(['dd', 'if=%s' % (fname), 'of=/dev/null'])
    data = table_utils._open_shelve_read(fname)
    keys = data.keys()
    data.close()
    return keys

def main():
    # uses inner to ensure we have complete rows. Otherwise
    # we would have NaN mixed in with the relevance scores,
    # and we can't train on those
    dfClicks = table_utils._read(config.CLICK_DATA)
    dfAll = dfClicks \
            .join(dfRel().set_index(['norm_query', 'hit_title']),
                  on=['norm_query', 'hit_title'], how='inner')

    dfClicks_len = len(dfClicks)
    rows_w_rel = len(dfAll)
    del dfClicks


    # Filter out pages that couldn't be loaded as docs/termvecs
    es_docs_keys = set(map(int, shelve_keys(config.ES_PAGE_DOCS_SHELVE)))
    dfAll = dfAll[dfAll['hit_page_id'].isin(es_docs_keys)]

    es_termvec_keys = set(map(int, shelve_keys(config.ES_PAGE_TERM_VEC_SHELVE)))
    dfAll = dfAll[dfAll['hit_page_id'].isin(es_termvec_keys)]

    # drop some unnecessary columns
    dfAll.drop(['session_id', 'clicked'], axis=1, inplace=True)

    # average out hit_position and hit_score to improve de-duplication.
    # on 10k queries this is a 90% reduction in data
    dfAll = dfAll.join(dfAll.groupby(['query', 'hit_page_id'])['hit_position'].mean(), on=['query', 'hit_page_id'], rsuffix='_mean')
    dfAll = dfAll.join(dfAll.groupby(['query', 'hit_page_id'])['hit_score'].mean(), on=['query', 'hit_page_id'], rsuffix='_mean')
    dfAll.drop(['hit_position', 'hit_score'], axis=1, inplace=True)

    # turn duplicates into a 'weight' column
    dfAll = dfAll.groupby(map(str, dfAll.columns)).size().reset_index()
    dfAll.rename(columns={0: 'weight'}, inplace=True)

    table_utils._write(config.ALL_DATA, dfAll)

    dfInfo = dfAll[["relevance"]].copy()
    table_utils._write(config.INFO_DATA, dfInfo)

    print 'Source clicks len: %d' % (dfClicks_len)
    print 'Rows with relevance: %d' % (rows_w_rel)
    print 'Final data len: %d' % (len(dfAll))
    print 'Ratio: %.3f' % (float(len(dfAll))/dfClicks_len)


if __name__ == "__main__":
    main()
