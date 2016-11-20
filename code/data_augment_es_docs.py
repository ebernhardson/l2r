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
    session = requests.Session()
    reqs = []
    for batch in np_utils._split(page_ids, 1000):
        data = json.dumps({'ids': list(batch)})
        reqs.append(grequests.post(url, data=data, params=params, session=session))

    defaults = config.ES_DOC_FIELDS_DEFAULTS
    multi_value_fields = [k for k, v in defaults.iteritems() if isinstance(v, tuple)]

    docs = []
    with progressbar.ProgressBar(max_value=len(reqs)) as bar:
        for r in bar(grequests.imap(reqs, size=20, exception_handler=exception_handler)):
            found = json.loads(r.text)['docs']
            for d in found:
                if not d['found']:
                    continue
                res = {'hit_%s' %(k): v for k, v in defaults.iteritems()}
                for k, v in d['fields'].iteritems():
                    # ES alwards returns a list, even if there is only one item.
                    # Flatten down single valued fields
                    res['hit_%s' % (k)] = tuple(v) if k in multi_value_fields else v[0]
                res['hit_page_id'] = int(d['_id'])
                docs.append(res)

    table_utils._write(config.ES_PAGE_DOCS, pd.DataFrame(docs).set_index(['hit_page_id']))

if __name__ == "__main__":
    main()
