import pandas as pd
import numpy as np

import os
import codecs
import json
import progressbar

from clickmodels.inference import DbnModel
from clickmodels.input_reader import InputReader, SessionItem

import config
from utils import table_utils

def main():
    df = table_utils._read(config.CLICK_DATA)
    # TODO: using only session_id might be more efficient, but (query, session_id)
    # is more obvious to debug
    grouped = df.groupby(['norm_query', 'session_id'])
    clickmodel_data_file = os.path.join(config.TMP_DIR, 'clickmodel.txt')
    pos = 0
    with codecs.open(clickmodel_data_file, 'w', 'utf8') as f:
        with progressbar.ProgressBar() as bar:
            for (norm_query, session_id), group in bar(grouped):
                assert "\t" not in norm_query
                assert type(norm_query) == unicode
                results = []
                clicks = []
                # TODO: groupby still necessary? check all group lengths, they might be 1
                # after adjustments to the source hive query
                for title, rows in group \
                        .sort_values(['hit_score'], ascending=False) \
                        .groupby(['hit_title'], sort=False):
                    # TODO: using page id instead of title might be more efficient,
                    # but title is easier for debugging
                    results.append(title)
                    clicks.append(True in list(rows['clicked']))
                    if len(results) >= config.DBN_CONFIG['MAX_DOCS_PER_QUERY']:
                        break
                # exclude groups with no clicks
                if not any(clicks):
                    continue
                # exclude too small result sets as well
                if len(results) < config.DBN_CONFIG['MIN_DOCS_PER_QUERY']:
                    continue
                f.write("\t".join([
                    str(pos), # hash digest
                    norm_query, # query
                    '0', # region
                    '0', # intent weight
                    json.dumps(results), # urls
                    json.dumps([False] * len(results)), # layout
                    json.dumps(clicks) # clicks
                ]) + "\n")
                pos += 1

    del df
    with codecs.open(clickmodel_data_file, 'r', 'utf8') as f:
        reader = InputReader(config.DBN_CONFIG['MIN_DOCS_PER_QUERY'],
                             config.DBN_CONFIG['MAX_DOCS_PER_QUERY'],
                             False,
                             config.DBN_CONFIG['SERP_SIZE'],
                             False,
                             discard_no_clicks=True)
        sessions = reader(f)
        dbn_config = config.DBN_CONFIG.copy()
        dbn_config['MAX_QUERY_ID'] = reader.current_query_id + 1
        model = DbnModel((0.9, 0.9, 0.9, 0.9), config=dbn_config)
        model.train(sessions)

        f.seek(0)
        results = []
        # This is a bit ugly and hackish ... but trying to not explode memory
        # by flipping the giant url_to_id and query_to_id dicts.
        seen = set()
        # hax
        with progressbar.ProgressBar(max_value=pos) as bar:
            pos = 0
            for line in f:
                bar.update(pos)
                pos += 1

                _, norm_query, _, _, titles, _, clicks = line.rstrip().split('\t')
                titles = json.loads(titles)
                if len(titles) < dbn_config['MIN_DOCS_PER_QUERY']:
                    continue
                query_id = reader.query_to_id[(norm_query, "0")]
                title_ids = [reader.url_to_id[t] for t in titles]
                session = SessionItem(0, query_id, title_ids, 0, [], {})
                relevances = model.get_model_relevances(session)
                for title, relevance in zip(titles, relevances):
                    if (norm_query, title) in seen:
                        continue
                    results.append([norm_query, title, relevance])
                    # alternatly could use drop_duplicates, not sure which
                    # is cheaper on memory usage
                    seen.add((norm_query, title))
        df = pd.DataFrame(results, columns=['norm_query', 'hit_title', 'relevance'])
        print 'Hits with relevance: %d' % (len(results))
        table_utils._write(config.RELEVANCE_DATA, df)


if __name__ == "__main__":
    main()
