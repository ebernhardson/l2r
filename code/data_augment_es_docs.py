import pandas as pd

import requests
import grequests
import json
import progressbar

import config
from utils import np_utils, table_utils

def exception_handler(req, e):
    raise e

def main():
    # returns an ndarray
    page_ids = table_utils._read(config.CLICK_DATA)['hit_page_id'].unique()

    url = config.ES_URL + '/page/_mget'
    params = {'fields': ','.join(config.ES_DOC_FIELDS)}

    defaults = config.ES_DOC_FIELDS_DEFAULTS
    multi_value_fields = [k for k, v in defaults.iteritems() if isinstance(v, tuple)]

    docs = table_utils._open_shelve_write(config.ES_PAGE_DOCS_SHELVE)
    i = 0
    try:
        with progressbar.ProgressBar(max_value=len(page_ids)) as bar:
            for top_batch in np_utils._split(page_ids, 10000):
                session = requests.Session()
                reqs = []
                for batch in np_utils._split(top_batch, 100):
                    data = json.dumps({'ids': list(batch)})
                    reqs.append(grequests.post(url, data=data, params=params, session=session))
                for r in grequests.imap(reqs, size=20, exception_handler=exception_handler):
                    found = json.loads(r.text)['docs']
                    for d in found:
                        if not d['found']:
                            continue
                        res = defaults.copy()
                        for field, v in d['fields'].iteritems():
                            # ES alwards returns a list, even if there is only one item.
                            # Flatten down single valued fields
                            res[field] = tuple(v) if field in multi_value_fields else v[0]
                        docs[str(d['_id'])] = res
                    i += len(found)
                    bar.update(i)
                docs.sync()
    finally:
        docs.close()


if __name__ == "__main__":
    main()
