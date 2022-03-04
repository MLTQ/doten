import pandas as pd
import json
from tqdm import tqdm
import plotly_express as px
import numpy as np
file = '6227612490.json'
data = []
for line in open(file, 'r'):
    data.append(json.loads(line))
data_df = pd.DataFrame(data)


def get_hero_list(data_df):
    units=set(data_df['unit'].to_list())
    return [hero for hero in units if type(hero) is str]
heros = get_hero_list(data_df)
paths =[]
for hero in heros:
    hero_df = data_df[data_df['unit']==hero]
    path_df = pd.DataFrame()
    path_df['time'] = hero_df['time']
    path_df['x'] = hero_df['x']
    path_df['y'] = hero_df['y']
    path_df['unit'] = hero_df['unit']
    paths.append[path_df]
paths_df = pd.concat(paths)
t = paths_df['time'].to_list()
paths_df['opacity'] = (t - np.min(t))/np.ptp(t)
fig = px.scatter_3d(paths_df, x='x', y='y', z='time',
              color='unit', opacity='opacity')
fig.show()