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
    z.query AS query,
    z.norm_query AS norm_query,
    z.session_id AS session_id,
    hit.pageid AS hit_page_id,
    hit.title AS hit_title,
    -- Some things can have the same score, so this isn't the same order we displayed, should
    -- add to CirrusSearchRequestSet to know the real hit position, or find the right hive solution
    ROW_NUMBER() OVER(PARTITION BY z.query, z.session_id ORDER BY -AVG(hit.score)) AS hit_position,
    -- If the user performed the query multiple times (forward, read, back navigation) the
    -- scores could be slightly different, just take a reasonable one they are probably
    -- close anyways.
    AVG(hit.score) AS hit_score,
    -- group contains multiple searches (forward, read, back navigation, or pagination)
    -- so this collects potentially multiple click's and checks if any were this hit
    ARRAY_CONTAINS(COLLECT_LIST(z.click_page_id), hit.pageid) AS clicked
FROM (
    SELECT
        top_query_clicks.query,
        top_query_clicks.norm_query,
        top_query_clicks.click_page_id,
        top_query_clicks.search_timestamp,
        top_query_clicks.click_timestamp,
        top_query_clicks.hits,
        top_query_clicks.session_id
    FROM
        ebernhardson.top_query_clicks
    JOIN (
            -- Randomly select N unique normalized queries
        SELECT
            x.project, x.norm_query
        FROM (
            SELECT
                project,
                norm_query,
                count(distinct year, month, day, session_id) as num_searchs
            FROM
                ebernhardson.top_query_clicks
            WHERE
                year = 2016
                AND project = '%s'
            GROUP BY
                project,
                norm_query
            ) x
        WHERE
            x.num_searchs >= %d
        DISTRIBUTE BY
            rand()
        SORT BY
            rand()
        LIMIT
            %d
        ) y
    ON
        y.norm_query = top_query_clicks.norm_query
        AND y.project = top_query_clicks.project
    WHERE
        year = 2016
    ) z
LATERAL VIEW
    -- ideally we want to know the order within hits, as this is the display
    -- order, but how?
    EXPLODE(z.hits) h AS hit
GROUP BY
    z.query,
    z.norm_query,
    z.session_id,
    hit.pageid,
    hit.title
;
""" % (config.WIKI_PROJECT, config.MIN_NUM_SEARCHES, config.MAX_QUERIES)

    if not os.path.isfile(config.CLICK_DATA_TSV):
        with tempfile.TemporaryFile() as tmp:
            command = []
            if False:
                command += ['ssh', '-o', 'Compression=yes', 'stat1002.eqiad.wmnet']
            command += ['hive', '-S', '-e', '"' + sql + '"']
            p = subprocess.Popen(command, stdin=None, stdout=tmp, stderr=subprocess.PIPE)
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
