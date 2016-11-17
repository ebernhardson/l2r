import pandas as pd

import config
from utils import table_utils

# Helper methods make it a bit cleaner to just pull in the df
# once for a join, and let gc throw it back out, hopefully
def dfRel():
    return table_utils._read(config.RELEVANCE_DATA)

def dfEsPageDocs():
    return table_utils._read(config.ES_PAGE_DOCS)

def dfEsPageTermVec():
    return table_utils._read(config.ES_PAGE_TERM_VEC)

def dfEsQueryTermVec():
    return table_utils._read(config.ES_QUERY_TERM_VEC)

def main():
    # Consistently uses inner, now sure if it's necessary but ensures
    # most rows are complete
    dfClicks = table_utils._read(config.CLICK_DATA)
    # Keep our data volumes down by removing things that are only duplicated
    # by identity. Not perfect, but is it good enough?
    # Alternatively could groupby(['query', 'hit_page_id'] and try to normalize
    # out the other data.
    del dfClicks['identity']
    dfClicks = dfClicks.drop_duplicates()

    dfAll = dfClicks \
            .join(dfRel().set_index(['query', 'hit_title']),
                          on=['query', 'hit_title'], how='left') \
            .join(dfEsPageDocs(), on='hit_page_id', how='left') \
            .join(dfEsPageTermVec(), on='hit_page_id', how='left') \
            .join(dfEsQueryTermVec(), on='query', how='left')

    table_utils._write(config.ALL_DATA, dfAll)

    print 'Source clicks len: %d' % (len(dfClicks))
    print 'Final data len: %d' % (len(dfAll))
    print 'Ratio: %.3f' % (float(len(dfAll))/len(dfClicks))


if __name__ == "__main__":
    main()
