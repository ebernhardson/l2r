import numpy as np
import pandas as pd

import config
from utils import table_utils

def main():
	df = table_utils._read(config.CLICK_DATA) \
		.join(table_utils._read(config.RELEVANCE_DATA).set_index(['norm_query', 'hit_title']), on=['norm_query', 'hit_title'], how='inner')

	dfQ = df['norm_query'].drop_duplicates()
	queries = dfQ.reindex(np.random.permutation(dfQ.index))[:20]
	del dfQ

	condition = np.zeros((len(df)), dtype=bool)
	for q in queries:
		condition = condition | (df['norm_query'] == q)
	dfShort = df[condition]

	pd.set_option('display.width', 1000)
	for norm_query, query_group in dfShort.groupby(['norm_query']):
		hits = []
		for title, hit_group in query_group.groupby(['hit_title']):
			num_clicks = np.sum(hit_group['clicked'])
			avg_position = np.mean(hit_group['hit_position'])
			relevance = np.mean(hit_group['relevance'])
			hits.append((title, num_clicks, avg_position, relevance))
                queries = list(query_group['query'].drop_duplicates())
		print("Normalized Query: %s" % (norm_query.encode('utf8')))
                print("Queries:\n\t%s" % ("\n\t".join(map(lambda x: x.encode('utf8'), queries))))
		hitDf = pd.DataFrame(hits, columns=['title', 'num_clicks', 'avg_pos', 'dbn_rel'])
		print(hitDf.sort_values(['dbn_rel'], ascending=False))
		print("\n\n")

if __name__ == "__main__":
	main()
