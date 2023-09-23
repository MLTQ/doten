import pandas as pd
import numpy as np
import json

def get_hero_list(data_df):
    '''

    :param data_df: df form pls
    :return:
    '''
    units=set(data_df['unit'].to_list())
    heros = []
    for hero in units:
        if type(hero) == str:
            if 'CDOTA_Unit_Hero' in hero:
                heros.append(hero)
    return heros


def get_hero_map(data_df):
    heros = get_hero_list(data_df)
    npc_form = [hero for hero in list(set(data_df['attackername'])) if (type(hero) == str) and 'hero' in hero]
    clean = [hero.replace('npc_dota_hero_','').replace('_','') for hero in npc_form]
    clean_map = dict(zip(clean, npc_form))
    hero_map = {}
    for c in clean:
        for h in heros:
            h_clean=h.replace('CDOTA_Unit_Hero_','').replace('_','').lower()
            if c in h_clean:
                hero_map[clean_map[c]]=h
    return heros, hero_map

def get_wards(data_df,slot_dict):
    '''

    :param data_df: the df form pls
    :param slot_dict: mapping of slot to heroes, output of get_slot_dict
    :return:
    '''
    heros, hero_map = get_hero_map(data_df)
    obs = data_df[data_df['type'] == 'obs'][['time','x','y','ehandle']]
    sen = data_df[data_df['type']=='sen'][['time','x','y','ehandle']]
    obs = get_wards_hero(data_df, obs, hero_map, 'obs')
    sen = get_wards_hero(data_df, sen, hero_map, 'sen')
    #obs['unit'] = obs.apply(lambda row: slot_dict[row['slot']]+'_obs', axis=1)
    #sen['unit'] = sen.apply(lambda row: slot_dict[row['slot']]+'_sen', axis=1)
    #obs = obs.drop(columns=['slot'])
    #sen = sen.drop(columns=['slot'])
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

def get_wards_hero(data_df, wards, hero_map, kind):
    attribution = []
    kinds = [f'item_ward_{kind}','item_ward_dispenser']
    for row in wards.iterrows():
        user_row = data_df[(data_df['time']==row[1]['time']) & data_df['inflictor'].isin(kinds)]
        if user_row.empty:
            user_row = data_df[(data_df['time'] == row[1]['time']+1) & data_df['inflictor'].isin(kinds)]
            #TODO: multiple wards placed simultaneously by different heros
        aname= user_row['attackername'].values[0]
        if aname in hero_map.keys():
            attribution.append(hero_map[aname]+'_'+kind)
        else:
            attribution.append(f'{aname}_{kind}')
    wards['unit'] = attribution
    return wards

def extend_ward_lives(ward_df):
    ward_pos = []
    for ward in ward_df.iterrows():
        if ward[1]['ehandle'] in ward_df['ehandle'].to_list():
            start = ward[1]['time']
            end = ward_df[ward_df['ehandle']==ward[1]['ehandle']]['time'].values[0]
            ticks = range(start, end+1)
            for tick in ticks:
                ward_pos.append((tick, ward[1]['x'], ward[1]['y'], ward[1]['ehandle'],ward[1]['unit']))
        else:
            start = ward[1]['time']
            end = start+480
            ticks = range(start, end + 1)
            for tick in ticks:
                ward_pos.append((tick, ward[1]['x'], ward[1]['y'], ward[1]['ehandle'],ward[1]['unit']))
    return pd.DataFrame(ward_pos, columns=['time','x','y','ehandle','unit'])

def get_wards_pos(data_df):
    heros, hero_map = get_hero_map(data_df)
    obs = data_df[data_df['type'] == 'obs'][['time','x','y','ehandle']]
    sen = data_df[data_df['type']=='sen'][['time','x','y','ehandle']]
    obs = get_wards_hero(data_df, obs, hero_map, 'observer')
    sen = get_wards_hero(data_df, sen, hero_map, 'sentry')
    deob = []
    dese = []
    for ob in obs.iterrows():
        deob.append(data_df[(data_df['ehandle']==ob[1]['ehandle']) & (data_df['type']=='obs_left')])
    deob = pd.concat(deob)[['time','x','y','attackername','ehandle']]
    for ob in sen.iterrows():
        dese.append(data_df[(data_df['ehandle']==ob[1]['ehandle']) & (data_df['type']=='sen_left')])
    dese = pd.concat(dese)[['time','x','y','attackername','ehandle']]
    unit = []
    for ob in deob.iterrows():
        if ob[1]['attackername'] in hero_map.keys():
            unit.append(hero_map[ob[1]['attackername']])
        else:
            unit.append(ob[1]['attackername'])
    deob['unit'] = unit
    unit = []
    for ob in dese.iterrows():
        if ob[1]['attackername'] in hero_map.keys():
            unit.append(hero_map[ob[1]['attackername']])
        else:
            unit.append(ob[1]['attackername'])
    dese['unit'] = unit
    sen_pos = extend_ward_lives(sen)
    obs_pos = extend_ward_lives(obs)
    return obs, sen, deob, dese, obs_pos, sen_pos


class game:

    def get_winner(self):
        loser = self.data[(self.data['type']=='DOTA_COMBATLOG_TEAM_BUILDING_KILL') & (self.data['targetname'].isin(['npc_dota_goodguys_fort','npc_dota_badguys_fort']))]['targetname']
        if 'goodguys' in loser._values[0]:
            winner = 2
        if 'badguys' in loser._values[0]:
            winner = 1
        return winner

    def __init__(self, data_df):
        self.data = data_df
        self.heros, self.hero_map = get_hero_map(data_df)
        self.pam_oreh = {v: k for k, v in self.hero_map.items()}
        self.observers, self.sentries, self.deobs, self.desen, self.obs_pos, self.sen_pos = get_wards_pos(data_df)
        self.ward_deaths = {'obs':[],'sen':[]}
        self.winner = self.get_winner()#json.loads(data_df[data_df['type']=='epilogue']['key']._values[0])['gameInfo_']['dota_']['gameWinner_']




class player:

    def __init__(self, game, name, npc_name,):
        self.name = name
        self.game = game
        #team 1 is radiant, team 2 is dire
        self.team = None
        self.npc_name = npc_name
        self.natural_name = name.replace('CDOTA_Unit_Hero_','')
        self.kills = None
        self.deaths = None
        self.path = None
        self.heros_in_game = None
        self.heros_on_team = None
        self.wards_placed = {'obs':[],'sen':[]}
        self.wards_killed = {'obs':[],'sen':[]}
        self.items = None
        self.abilities_used = None
        self.lane = None
        self.farm = None

    def get_path(self):
        hero_df = self.game.data[self.game.data['unit'] == self.name]
        path_df = pd.DataFrame()
        path_df['time'] = hero_df['time']
        path_df['x'] = hero_df['x']
        path_df['y'] = hero_df['y']
        path_df['unit'] = hero_df['unit']
        self.path = path_df

    def get_kills_and_deaths(self):

        kills_df = self.game.data[
            (self.game.data['type'] == 'DOTA_COMBATLOG_DEATH') & (self.game.data['attackername'] == self.npc_name)]
        kills_locs = []
        for row in kills_df.iterrows():
            if 'hero' in row[1]['targetname']:
                kill = self.game.data[(self.game.data['type'] == 'interval') & (self.game.data['unit'] == self.name) & (
                        self.game.data['time'] == row[1]['time'])][['time', 'x', 'y']]
                kill['target'] = row[1]['targetname']
                kills_locs.append(kill)
        if kills_locs:
            self.kills = pd.concat(kills_locs)

        deaths_df = self.game.data[
            (self.game.data['type'] == 'DOTA_COMBATLOG_DEATH') & (self.game.data['targetname'] == self.npc_name)]
        deaths_locs = []
        for row in deaths_df.iterrows():
            death = self.game.data[(self.game.data['type'] == 'interval') & (self.game.data['unit'] == self.name) & (
                    self.game.data['time'] == row[1]['time'])][['time', 'x', 'y']]
            death['source'] = row[1]['attackername']
            deaths_locs.append(death)
        if deaths_locs:
            self.deaths = pd.concat(deaths_locs)

    def get_farm(self):
        kills_df = self.game.data[
            (self.game.data['type'] == 'DOTA_COMBATLOG_DEATH') & (self.game.data['attackername'] == self.npc_name)]
        kills_locs = []
        for row in kills_df.iterrows():
            if 'hero' not in row[1]['targetname']:
                kill = self.game.data[(self.game.data['type'] == 'interval') & (self.game.data['unit'] == self.name) & (
                        self.game.data['time'] == row[1]['time'])][['time', 'x', 'y']]
                kill['target'] = row[1]['targetname']
                kills_locs.append(kill)
        if kills_locs:
            self.farm = pd.concat(kills_locs)


    def get_team(self):
        if self.path[self.path['time']==self.path['time'].min()][['x','y']].T.sum()._values[0]>200:
            self.team = 2
        else:
            self.team = 1
        #Dire = 2, radiant = 1. I wanted 0 indexing but the dota client actually uses this system

    def get_wards(self, kind='observer'):
        if kind == 'observer':
            kind = 'obs'
        if kind == 'sentry':
            kind = 'sen'
        kind_map = {'obs':'observer','sen':'sentry'}
        wards = self.game.data[self.game.data['type'] == kind][['time','x','y','ehandle']]
        attribution = []
        kinds = [f'item_ward_{kind_map[kind]}', 'item_ward_dispenser'] # Dispenser????
        ward_list = []
        for row in wards.iterrows():
            user_row = self.game.data[(self.game.data['time'] == row[1]['time']) & self.game.data['inflictor'].isin(kinds)]
            if user_row.empty:
                user_row = self.game.data[(self.game.data['time'] == row[1]['time'] + 1) & self.game.data['inflictor'].isin(kinds)]
                # TODO: multiple wards placed simultaneously by different heros
            aname = user_row['attackername'].values[0]
            if aname ==self.npc_name:
                attribution.append(self.natural_name + '_' + kind)
                ward_list.append(row[1])
        if attribution:
            wards = pd.DataFrame(ward_list)
            wards['unit'] = attribution
            self.wards_placed[kind] = wards
        deward = []
        for ward in wards.iterrows():
            deward.append(self.game.data[(self.game.data['ehandle'] == ward[1]['ehandle']) &
                                         (self.game.data['type'] == f'{kind}_left') &
                                         (self.game.data['attackername'] ==self.npc_name)])
        deward = pd.concat(deward)[['time', 'x', 'y', 'attackername', 'ehandle']]

        self.wards_killed[kind]=deward

