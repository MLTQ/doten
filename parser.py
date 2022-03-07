import pandas as pd
import json
from tqdm import tqdm
import plotly_express as px
import numpy as np
from PIL import Image
from scipy import misc
import plotly.graph_objects as go

def loader(file):

    data = []
    for line in open(file, 'r'):
        data.append(json.loads(line))
    data_df = pd.DataFrame(data)
    return data, data_df

def get_slot_dict(data_df):
    '''
    At the beginning of a tick there is an event type 'interval', which reiterates the mapping of units to slot ids
    :param data_df: df form pls
    :return:
    '''
    tick = data_df[data_df['time'] == 2]
    interval = tick[tick['type'] == 'interval']
    slot_dict = {}
    for unit, slot in zip(interval['unit'].to_list(), interval['slot'].to_list()):
        slot_dict[slot] = unit
    return slot_dict

def get_wards(data_df,slot_dict):
    '''

    :param data_df: the df form pls
    :param slot_dict: mapping of slot to heroes, output of get_slot_dict
    :return:
    '''
    npc_form_heros = [hero.replace('CDOTA_Unit', 'npc_dota').lower() for hero in heros]
    hero_map = dict(zip(npc_form_heros, heros))
    obs = data_df[data_df['type'] == 'obs'][['time','x','y','slot']]
    sen = data_df[data_df['type']=='sen'][['time','x','y','slot']]
    obs['unit'] = obs.apply(lambda row: slot_dict[row['slot']]+'_obs', axis=1)
    sen['unit'] = sen.apply(lambda row: slot_dict[row['slot']]+'_sen', axis=1)
    obs = obs.drop(columns=['slot'])
    sen = sen.drop(columns=['slot'])
    deobs_inst = data_df[(data_df['targetname']=='npc_dota_observer_wards') & (data_df['type']=='DOTA_COMBATLOG_DEATH') & (data_df['value']==50)]
    desen_inst = data_df[(data_df['targetname']=='npc_dota_sentry_wards') & (data_df['type']=='DOTA_COMBATLOG_DEATH') & (data_df['value']==50)]
    deobs_list=[]
    desen_list=[]
    for row in deobs_inst.iterrows():
        deobs_list.append(data_df[(data_df['time']==row[1]['time'])&(data_df['type']=='interval')&(data_df['unit']==hero_map[row[1]['attackername']])])
    for row in desen_inst.iterrows():
        desen_list.append(data_df[(data_df['time'] == row[1]['time']) & (data_df['type'] == 'interval') & (data_df['unit'] == hero_map[row[1]['attackername']])])
    deobs = pd.concat(deobs_list)[['time','x','y','slot']]
    deobs['unit'] = deobs.apply(lambda row: slot_dict[row['slot']]+'_dob', axis=1)
    desen = pd.concat(desen_list)[['time','x','y','slot']]
    desen['unit'] = desen.apply(lambda row: slot_dict[row['slot']]+'_dse', axis=1)
    return obs, sen, deobs, desen

def get_hero_list(data_df):
    '''

    :param data_df: df form pls
    :return:
    '''
    units=set(data_df['unit'].to_list())
    return [hero for hero in units if type(hero) is str]

def get_map(version):
    #from https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot
    version = version.replace('.', '_')
    im = np.asarray(Image.open(f"minimap_{version}.jpg").resize((128,128)).transpose(Image.FLIP_LEFT_RIGHT))
    im_x, im_y, im_layers = im.shape
    eight_bit_img = Image.fromarray(im).convert('P', palette='WEB', dither=None)
    dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
    idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))
    colorscale = [[i / 255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    x = np.linspace(0, im_x, im_x)
    y = np.linspace(0, im_y, im_y)
    z = np.zeros(im.shape[:2])-1.5
    return eight_bit_img, colorscale, x, y, z


def generate_kd_pairs(kills, deaths):
    pairs = []
    #Because not all deaths are from other heroes lol
    deaths_lim = deaths[deaths['time'].isin(kills['time'].to_list())]
    #pairs = list(zip([tuple(r) for r in kills[['time', 'x', 'y']].to_numpy()],
    #                 [tuple(r) for r in deaths_lim[['time', 'x', 'y']].to_numpy()]))
    for k in kills[['time', 'x', 'y']].to_numpy():
        d = deaths[deaths['time']==k[0]][['time', 'x', 'y']].to_numpy()
        pairs.append((tuple(k), tuple(tuple(d)[0])))
    return pairs

def plot_paths(paths_df, kills, deaths, version):
    eight_bit_img, colorscale, x, y, z = get_map(version)
    traces = []
    x_adj = 64
    y_adj = 64
    for unit in set(paths_df['unit'].to_list()):
        if 'obs' in unit:
            unit_df = paths_df[paths_df['unit'] == unit]
            traces.append(go.Scatter3d(x=unit_df['x']-x_adj,
                                       y=unit_df['y']-y_adj,
                                       z=unit_df['time']/60,
                                       name=unit,
                                       mode='markers',
                                       marker=dict(
                                           color='yellow',
                                           opacity=1
                                       ),
                                       line=dict(width=0),
                                       ))
        elif 'sen' in unit:
                unit_df = paths_df[paths_df['unit'] == unit]
                traces.append(go.Scatter3d(x=unit_df['x'] - x_adj,
                                           y=unit_df['y'] - y_adj,
                                           z=unit_df['time'] / 60,
                                           name=unit,
                                           mode='markers',
                                           marker=dict(
                                               color='blue',
                                               opacity=1
                                           ),
                                           line=dict(width=0),
                                           ))
        elif 'dse' in unit:
                unit_df = paths_df[paths_df['unit'] == unit]
                traces.append(go.Scatter3d(x=unit_df['x'] - x_adj,
                                           y=unit_df['y'] - y_adj,
                                           z=unit_df['time'] / 60,
                                           name=unit,
                                           mode='markers',
                                           marker=dict(
                                               color='navy',
                                               opacity=1
                                           ),
                                           line=dict(width=0),
                                           ))
        elif 'dob' in unit:
                unit_df = paths_df[paths_df['unit'] == unit]
                traces.append(go.Scatter3d(x=unit_df['x'] - x_adj,
                                           y=unit_df['y'] - y_adj,
                                           z=unit_df['time'] / 60,
                                           name=unit,
                                           mode='markers',
                                           marker=dict(
                                               color='brown',
                                               opacity=1
                                           ),
                                           line=dict(width=0),
                                           ))
        elif 'kill' in unit:
            unit_df = paths_df[paths_df['unit'] == unit]
            traces.append(go.Scatter3d(x=unit_df['x'] - x_adj,
                                       y=unit_df['y'] - y_adj,
                                       z=unit_df['time'] / 60,
                                       name=unit,
                                       mode='markers',
                                       marker=dict(
                                           color='red',
                                           opacity=1
                                       ),
                                       line=dict(width=0),
                                       ))
        elif 'death' in unit:
            unit_df = paths_df[paths_df['unit'] == unit]
            traces.append(go.Scatter3d(x=unit_df['x'] - x_adj,
                                       y=unit_df['y'] - y_adj,
                                       z=unit_df['time'] / 60,
                                       name=unit,
                                       mode='markers',
                                       marker=dict(
                                           color='Black',
                                           opacity=1
                                       ),
                                       line=dict(width=0),
                                       ))

        else:
            unit_df = paths_df[paths_df['unit']==unit]
            traces.append(go.Scatter3d(x=unit_df['x']-x_adj,
                                       y=unit_df['y']-y_adj,
                                       z=unit_df['time']/60,
                                       name=unit,
                                       mode='lines',
                                       marker=dict(
                                       opacity=0.6
                                        ),
                                ))

    pairs = generate_kd_pairs(kills, deaths)
    #
    #For each pair, add them into a single list and then slap a 'None' between them
    x_lines = []
    y_lines = []
    time_lines = []
    for p in pairs:
        time_lines.append(p[0][0]/60)
        time_lines.append(p[1][0]/60)
        time_lines.append(None)
        x_lines.append(p[0][1] - x_adj)
        x_lines.append(p[1][1] - x_adj)
        x_lines.append(None)
        y_lines.append(p[0][2] - y_adj)
        y_lines.append(p[1][2] - y_adj)
        y_lines.append(None)

    traces.append(go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=time_lines,
        mode='lines',
        name='kd_lines',
        marker=dict(
            color='black',
            line=dict(
                width=2
            )
        ),
        line=dict(width=6)
    ))


    fig = go.Figure(data=traces)

    fig.add_trace(go.Surface(x=x, y=y, z=z,
                             surfacecolor=eight_bit_img,
                             cmin=0,
                             cmax=255,
                             colorscale=colorscale,
                             showscale=False,
                             lighting_diffuse=1,
                             lighting_ambient=1,
                             lighting_fresnel=1,
                             lighting_roughness=1,
                             lighting_specular=0.5,

                             ))
    fig.show()


def get_kills(data_df, paths_df, heros):
    npc_form_heros = [hero.replace('CDOTA_Unit', 'npc_dota').lower() for hero in heros]
    kills_df = data_df[data_df['type'] == 'DOTA_COMBATLOG_DEATH']
    kills_df = kills_df[kills_df['targetname'].isin(npc_form_heros)]
    hero_map = dict(zip(npc_form_heros, heros))
    kills = []
    deaths = []
    for row in kills_df.iterrows():
        if row[1]['attackername'] in npc_form_heros:
            kills.append(paths_df[(paths_df['time']==row[1]['time'])&(paths_df['unit']==hero_map[row[1]['attackername']])])
        deaths.append(paths_df[(paths_df['time']==row[1]['time'])&(paths_df['unit']==hero_map[row[1]['targetname']])])
    kills_df = pd.concat(kills)
    kills_df['unit'] = kills_df.apply(lambda row: row['unit'] + '_kill', axis=1)
    deaths_df = pd.concat(deaths)
    deaths_df['unit'] = deaths_df.apply(lambda row: row['unit'] + '_death', axis=1)
    return kills_df, deaths_df

if __name__=='__main__':

    file = '6227612490.json'
    data, data_df = loader(file)
    heros = get_hero_list(data_df)
    paths =[]


    version = '7.29'

    for hero in heros:
        hero_df = data_df[data_df['unit']==hero]
        path_df = pd.DataFrame()
        path_df['time'] = hero_df['time']
        path_df['x'] = hero_df['x']
        path_df['y'] = hero_df['y']
        path_df['unit'] = hero_df['unit']
        paths.append(path_df)

    slot_dict = get_slot_dict(data_df)
    obs, sen, deobs, desen = get_wards(data_df, slot_dict)

    paths.append(sen)
    paths.append(obs)
    paths.append(deobs)
    paths.append(desen)
    paths_df = pd.concat(paths)
    kills, deaths = get_kills(data_df, paths_df, heros)
    paths.append(kills)
    paths.append(deaths)
    paths_df = pd.concat(paths)
    t = paths_df['time'].to_list()
    paths_df['opacity'] = (t - np.min(t))/np.ptp(t)
    plot_paths(paths_df,kills, deaths, version)
