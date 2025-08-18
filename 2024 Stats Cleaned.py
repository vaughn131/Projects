import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import lxml
from sklearn.cluster import KMeans




url = "https://www.pro-football-reference.com/years/2024//passing.htm"
dfs = pd.read_html(url)
qb_stats = dfs[0]


qb_stats = qb_stats[qb_stats['Player'] != 'Player']


if isinstance(qb_stats.columns, pd.MultiIndex):
    qb_stats.columns = qb_stats.columns.droplevel(0)


qb_stats.reset_index(drop=True, inplace=True)


qb_stats = qb_stats[qb_stats['Player'].notnull()]


numeric_cols = qb_stats.columns.drop(['Player', 'Team'])
qb_stats[numeric_cols] = qb_stats[numeric_cols].apply(pd.to_numeric, errors='coerce')


qb_stats = qb_stats[qb_stats['Att'] > 100]


qb_stats['TD-INT Ratio'] = qb_stats['TD'] / qb_stats['Int'].replace(0, np.nan)
qb_stats['Yards/Game'] = qb_stats['Yds'] / qb_stats['G']
qb_stats['Comp%'] = qb_stats['Cmp'] / qb_stats['Att']


qb_stats.replace([np.inf, -np.inf], np.nan, inplace=True)
qb_stats.fillna(0, inplace=True)

features = qb_stats[['Yds', 'TD', 'Int', 'Rate']].copy()
kmeans = KMeans(n_clusters=3, random_state=42)
qb_stats['Tier'] = kmeans.fit_predict(features)





qb_stats.to_csv("qb_clustered_stats_2024.csv", index=False)

print("✅ QB stats saved to qb_clustered_stats_2024.csv")
