import pandas as pd

import requests
import grequests
import json
import progressbar

import config
from utils import np_utils, table_utils, es_utils

def exception_handler(req, e):
    raise e

def main():
    # returns an ndarray
    page_ids = table_utils._read(config.CLICK_DATA)['hit_page_id'].unique()

    url = config.ES_URL + '/page/_mtermvectors'
    # shelve is a disk-backed dict, it's a good bit slower than pandas in-memory
    # implementation, but one large sets there is just too much data to hold in memory
    docs = table_utils._open_shelve_write(config.ES_PAGE_TERM_VEC_SHELVE)
    i = 0
    try:
        with progressbar.ProgressBar(max_value=len(page_ids)) as bar:
            # doing it all in one grequests.imap seems to hold onto a ton
            # of extra memory. On a sample of ~200k page id's i was seeing
            # 14G+, but breaking into smaller pieces brought it down to 7G
            for top_batch in np_utils._split(page_ids, 10000):
                session = requests.Session()
                reqs = []
                for batch in np_utils._split(top_batch, 100):
                    data = {
                        'ids': list(batch),
                        'parameters': {
                            'fields': config.ES_TERM_FIELDS,
                            'positions': False,
                            'offsets': False,
                            'term_statistics': True,
                            'filter': {
                                'max_num_terms': config.TERM_VEC_MAX_NUM_TERMS,
                            }
                        }
                    }
                    reqs.append(grequests.post(url, data=json.dumps(data), session=session))
    
                for r in grequests.imap(reqs, size=20, exception_handler=exception_handler):
                    found = json.loads(r.text)['docs']
                    for d in json.loads(r.text)['docs']:
                        # TODO: Store empty doc, or handle when reading?
                        if d['found']:
                            docs[str(d['_id'])] = es_utils.parse_termvec(d)
                    i += len(found)
                    bar.update(i)
                # sync every 10k docs frees up memory
                docs.sync()
    finally:
        docs.close()

if __name__ == "__main__":
    main()
