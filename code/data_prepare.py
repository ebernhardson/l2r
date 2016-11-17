import numpy as np
import pandas as pd

import os
import tempfile
import subprocess
import codecs
import json

import config
from utils import table_utils

def main():
    # Fetch input data from hive
    # Attempting to select all clicks for a limited number of queries makes
    # this rather complicated...but simpler than doing the top few sub-queries
    # in pandas
    
    sql = """
SELECT
    query,
    identity,
    search_id,
    -- group contains multiple searches (forward, read, back navigation, or pagination)
    -- so this collects potentially multiple click's and checks if any were this hit
    ARRAY_CONTAINS(COLLECT_LIST(click_page_id), hit.pageid) AS clicked,
    -- Some things can have the same score, so this isn't the same order we displayed, should
    -- add to CirrusSearchRequestSet to know the real hit position
    ROW_NUMBER() OVER(PARTITION BY query, identity ORDER BY -MAX(hit.score)) AS hit_position,
    hit.title AS hit_title,
    hit.pageid AS hit_page_id,
    -- If the user performed the query multiple times (forward, read, back navigation) the
    -- scores could be slightly different, just take a reasonable one they are probably
    -- close anyways. Could AVG() i suppose?
    MAX(hit.score) AS hit_score
FROM (
    SELECT
        query,
        identity,
        search_id,
        click.page_id AS click_page_id,
        click.hits AS hits
    FROM (
        SELECT
            query,
            meta.identity AS identity,
            -- This assigns a single search_id to a (query, identity) pair, may not be
            -- necessary but seemed plausibly useful
            ROW_NUMBER() OVER () as search_id,
            COLLECT_LIST(NAMED_STRUCT('page_id', meta.page_id, 'hits', meta.hits)) AS clicks
        FROM (
            SELECT
                query,
                -- Collects all searches by all users to one query
                COLLECT_LIST(NAMED_STRUCT('identity', identity, 'page_id', page_id, 'hits', hits)) AS collected
            FROM
                ebernhardson.top_query_clicks
            WHERE
                num_searches > %d
                AND project = '%s'
            GROUP BY
                query
            ORDER BY
                RAND()
            LIMIT %d
            ) x
        LATERAL VIEW
            EXPLODE(collected) m AS meta
        GROUP BY
            query,
            meta.identity
        ) y
    LATERAL VIEW
        EXPLODE(clicks) c AS click
    ) z
LATERAL VIEW
    EXPLODE(hits) h AS hit
GROUP BY
    search_id,
    query,
    identity,
    hit.pageid,
    hit.title
;
""" % (config.MIN_NUM_SEARCHES, config.WIKI_PROJECT, config.MAX_QUERIES)

    if not os.path.isfile(config.CLICK_DATA_TSV):
        with tempfile.TemporaryFile() as tmp:
            p = subprocess.Popen(['ssh', '-o', 'Compression=yes', 'stat1002.eqiad.wmnet',
                                  'hive', '-S', '-e', '"' + sql + '"'],
                                 stdin=subprocess.PIPE, stdout=tmp, stderr=subprocess.PIPE)
            _, stderr = p.communicate(input=sql)
            if not os.fstat(tmp.fileno()).st_size > 0:
                print stderr
                raise Exception
            tmp.seek(0)
            with codecs.open(config.CLICK_DATA_TSV, 'w', 'utf-8') as f:
                for line in tmp:
                    try:
                        f.write(line.decode('utf-8'))
                    except UnicodeDecodeError:
                        pass

    # Read the tsv into pandas
    df = pd.read_csv(config.CLICK_DATA_TSV, sep="\t", index_col=False, encoding="utf-8")
    
    # and write it back out as hdf5
    table_utils._write(config.CLICK_DATA, df)


if __name__ == '__main__':
    main()
