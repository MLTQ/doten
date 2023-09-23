import pandas as pd
import json
from tqdm import tqdm
import plotly_express as px
import numpy as np
from PIL import Image
from scipy import misc
import plotly.graph_objects as go
from glob import glob
import numpy as np
from scipy import stats
from hero import *
from plotly.subplots import make_subplots

from multiprocessing import cpu_count, Process, Queue
import json
from glob import glob


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


def get_map(version):
    #from https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot
    version = version.replace('.', '_')
    im = np.asarray(Image.open(f"minimap_{version}.jpg").resize((128,128)).rotate(180).transpose(Image.FLIP_LEFT_RIGHT))
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

def get_kills(data_df, paths_df):
    heros, hero_map = get_hero_map(data_df)
    npc_form_heros = list(hero_map.keys())
    kills_df = data_df[data_df['type'] == 'DOTA_COMBATLOG_DEATH']
    kills_df = kills_df[kills_df['targetname'].isin(npc_form_heros)]

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

def get_posmap(games, path_type, team_id, vic_or_def):
    relevant_teams = []
    team_id = int(team_id)
    if vic_or_def == 'vic':
        state = 'Victory'
        for match in games:
            if match[0].winner == team_id:
                relevant_teams.append(match[team_id])
    elif vic_or_def == 'def':
        state = 'Defeat'
        for match in games:
            if match[0].winner != team_id:
                relevant_teams.append(match[team_id])
    elif vic_or_def == None:
        state = 'All matches'
        for match in games:
            relevant_teams.append(match[team_id])
    poses = []

    for team in relevant_teams:
        for member in team:
            if path_type == 'kills':
                if member.kills is not None:
                    poses.append(member.kills)
            elif path_type == 'deaths':
                if member.kills is not None:
                    poses.append(member.deaths)
            elif path_type == 'farm':
                if member.kills is not None:
                    poses.append(member.farm)
            elif path_type == 'wards':
                for kind in ['obs', 'sen']:
                    if len(member.wards_placed[kind]) > 0:
                        poses.append(member.wards_placed[kind])
            else:
                print(f'Kind {path_type} not recognized')

    path_df = pd.concat(poses)

    return path_df, state


def plot_path_heatmap(path_df, path_type, team_id, state,  version='7.29'):
    if path_type == 'kills':
        colors = 'PuRd'
    elif path_type == 'deaths':
        colors = 'greys'
    elif path_type == 'wards':
        colors = 'viridis'
    elif path_type == 'farm':
        colors = 'Mint'

    x = path_df['x'].to_numpy()
    y = path_df['y'].to_numpy()
    z = (path_df['time']/60).to_numpy()

    xyz = np.vstack([x, y, z])
    kde = stats.gaussian_kde(xyz)

    # Evaluate kde on a grid
    xmin, ymin, zmin = x.min(), y.min(), z.min()
    xmax, ymax, zmax = x.max(), y.max(), z.max()
    xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    density = kde(coords).reshape(xi.shape)

    names = ['x', 'y', 'z']
    index = pd.MultiIndex.from_product([range(s) for s in density.shape], names=names)
    df = pd.DataFrame(pd.DataFrame({'Density': density.flatten()}, index=index)['Density']).reset_index()

    df['C_scaled'] = df['Density'].max() / df['Density']
    df['C_scaled'] = 50 * (df['Density'] - df['Density'].min()) / (df['Density'].max() - df['Density'].min())
    traces = go.Volume(
        x=xi.flatten() - x_adj,
        y=yi.flatten() - y_adj,
        z=zi.flatten(),
        value=df['C_scaled'],
        opacity=0.1,
        name=f'{path_type.capitalize()} for {team_id}, {state}',
        colorscale=colors,# needs to be small to see through all surfaces
        surface_count=40,  # needs to be a large number for good volume rendering
    )
    fig = go.Figure(data=traces)
    return traces

def plot_slice_heatmap(path_df, path_type, team_id, state,  version='7.29'):
    if path_type == 'kills':
        colors = 'PuRd'
    elif path_type == 'deaths':
        colors = 'greys'
    elif path_type == 'wards':
        colors = 'viridis'
    elif path_type == 'farm':
        colors = 'Mint'

    x = path_df['x'].to_numpy()
    y = path_df['y'].to_numpy()
    z = (path_df['time']/60).to_numpy()

    xyz = np.vstack([x, y, z])
    kde = stats.gaussian_kde(xyz)

    # Evaluate kde on a grid
    xmin, ymin, zmin = x.min(), y.min(), z.min()
    xmax, ymax, zmax = x.max(), y.max(), z.max()
    xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    density = kde(coords).reshape(xi.shape)

    names = ['x', 'y', 'z']
    index = pd.MultiIndex.from_product([range(s) for s in density.shape], names=names)
    df = pd.DataFrame(pd.DataFrame({'Density': density.flatten()}, index=index)['Density']).reset_index()

    df['C_scaled'] = df['Density'].max() / df['Density']
    df['C_scaled'] = 50 * (df['Density'] - df['Density'].min()) / (df['Density'].max() - df['Density'].min())
    traces = []
    for i in range(0,50,10):
        traces.append(go.Volume(
            x=xi.flatten() - x_adj,
            y=yi.flatten() - y_adj,
            z=zi.flatten(),
            value=df['C_scaled'],
            opacity=0.1,
            visible=False,
            slices_z=dict(show=True, locations=[i]),
            name=f'{path_type.capitalize()} for {team_id}, {state}',
            colorscale=colors,# needs to be small to see through all surfaces
            surface_count=40,  # needs to be a large number for good volume rendering
        ))
    fig = go.Figure(data=traces)
    fig.data[0].visible = True

    return traces

def load_games(games):
    game_dfs = []
    for game in tqdm(games, desc='Loading games...'):
        _, data_df = loader(game)
        game_dfs.append(data_df)
    return game_dfs

def multigame_paths(games):
    paths = []
    for data_df in tqdm(games, desc='Getting Paths'):
        heros = get_hero_list(data_df)


        for hero in heros:
            hero_df = data_df[data_df['unit'] == hero]
            path_df = pd.DataFrame()
            path_df['time'] = hero_df['time']
            path_df['x'] = hero_df['x']
            path_df['y'] = hero_df['y']
            path_df['unit'] = hero_df['unit']
            paths.append(path_df)

        slot_dict = get_slot_dict(data_df)
        obs, sen, deobs, desen, obs_pos, sen_pos = get_wards_pos(data_df)

        paths.append(sen)
        paths.append(obs)
        paths.append(deobs)
        paths.append(desen)
        paths_df = pd.concat(paths)
        kills, deaths = get_kills(data_df, paths_df)
        paths.append(kills)
        paths.append(deaths)
    paths_df = pd.concat(paths)
    return paths_df

def load_teams(game_df):

    game_obj = game(game_df)
    teamRad = []
    teamDir = []
    for hero in game_obj.heros:
        temp_hero = player(game=game_obj, name=hero, npc_name=game_obj.pam_oreh[hero])
        temp_hero.get_path()
        temp_hero.get_kills_and_deaths()
        temp_hero.get_team()
        temp_hero.get_wards(kind='obs')
        temp_hero.get_wards(kind='sen')
        temp_hero.get_farm()
        if temp_hero.team == 1:
            teamRad.append(temp_hero)
        else:
            teamDir.append(temp_hero)

    return (game_obj, teamRad, teamDir)

def plot_path_matrix(traces, title):
    fig = make_subplots(specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}]],
                        rows=2, cols=2,
                        subplot_titles=("Radiant Victory", "Dire Victory", "Radiant Defeat", "Dire Defeat"),
                        horizontal_spacing=0.01, vertical_spacing=0.02)
    eight_bit_img, colorscale, x, y, z = get_map(version)
    map_img = go.Surface(x=x, y=y, z=z - 1,
                         surfacecolor=eight_bit_img,
                         cmin=0,
                         cmax=255,
                         colorscale=colorscale,
                         showscale=False,
                         lighting_diffuse=1,
                         lighting_ambient=1,
                         lighting_fresnel=1,
                         lighting_roughness=1,
                         lighting_specular=0.5, )
    fig.add_trace(traces[0], row=1, col=1)
    fig.add_trace(map_img, row=1, col=1)
    fig.add_trace(traces[1], row=1, col=2)
    fig.add_trace(map_img, row=1, col=2)
    fig.add_trace(traces[2], row=2, col=1)
    fig.add_trace(map_img, row=2, col=1)
    fig.add_trace(traces[3], row=2, col=2)
    fig.add_trace(map_img, row=2, col=2)
    fig.update_layout(go.Layout(
        template='plotly_dark',
        showlegend=True,
        title = title
    ))
    fig.write_html(f"{title}.html")

def plot_single_heatmap(trace, title):
    layout = go.Layout(
        template='plotly_dark',
        showlegend=True,
        title=title,
    )
    fig = go.Figure(data=trace, layout=layout)
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to step: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    eight_bit_img, colorscale, x, y, z = get_map(version)
    map_img = go.Surface(x=x, y=y, z=z - 1,
                         surfacecolor=eight_bit_img,
                         cmin=0,
                         cmax=255,
                         colorscale=colorscale,
                         showscale=False,
                         lighting_diffuse=1,
                         lighting_ambient=1,
                         lighting_fresnel=1,
                         lighting_roughness=1,
                         lighting_specular=0.5, )
    fig.add_trace(map_img)
    fig.show()


if __name__=='__main__':

    file = 'replays/6227612490.json'
    version = '7.31'
    x_adj = 64
    y_adj = 64

    for MMR in ['3',]:#'7']:
        games = glob(f'replays/{MMR}kmmr/*')
        games = load_games(games)

        matches = []
        for game_inst in tqdm(games):
            matches.append(load_teams(game_inst))

        #path_df, state = get_posmap(matches, 'farm', 1, 'vic')
        #title = 'Radiant farm given victory'
        #plot_single_heatmap(plot_path_heatmap(path_df, 'farm', 1, state, version='7.29'), title)

        for path_type in ['wards','kills','deaths','farm']:
            traces = []
            team = 1
            v = 'vic'
            path_df, state = get_posmap(matches,path_type, team, v)
            traces.append(plot_path_heatmap(path_df, path_type, team, state, version='7.29'))
            v = 'def'
            path_df, state = get_posmap(matches, path_type, team, v)
            traces.append(plot_path_heatmap(path_df, path_type, team, state, version='7.29'))

            v = 'vic'
            team=2
            path_df, state = get_posmap(matches,path_type, team, v)
            traces.append(plot_path_heatmap(path_df, path_type, team, state, version='7.29'))
            v='def'
            path_df, state = get_posmap(matches,path_type, team, v)
            traces.append(plot_path_heatmap(path_df, path_type, team, state, version='7.29'))

            plot_path_matrix(traces, f'{MMR}kMMR average {path_type} heatmap')



    paths_df = multigame_paths(games)
