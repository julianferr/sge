import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist

from constant_variables import HOME_TEAM, PERIOD2_FRAME, AWAY_TEAM


camera_path = 'tracking.csv'
players_path = 'tracking_data.csv'

player_df = pd.read_csv(players_path)
camera_df = pd.read_csv(camera_path)

del camera_df['MATCH_ID']
del player_df['MATCH_ID']
del player_df['TRACKABLE_OBJECT']

camera_df['TIMESTAMP'] = pd.to_datetime(camera_df['TIMESTAMP'], format='%H:%M:%S.%f', errors='coerce')
camera_df['TIMESTAMP'] = camera_df['TIMESTAMP'].dt.time

camera_df = camera_df.dropna(subset=['TIMESTAMP'])

# Total game length to determine avg fps
total_time = max(camera_df['TIMESTAMP'])
total_seconds = (total_time.hour * 3600 + total_time.minute * 60 + total_time.second + total_time.microsecond / 1e6)

total_frames = max(camera_df['FRAME'])

avg_fps = total_frames / total_seconds
avg_frame_length = total_seconds / total_frames  # spf

usain_speed = 11  # max human speed (10.44mps)
usain_distance = usain_speed * avg_frame_length  # max distance (m per frame)

player_df = player_df.sort_values(by=['TRACK_ID', 'FRAME'])


def smooth(adj_pos=False):
    for player_id in player_df['TRACK_ID'].unique():
        player_data = player_df[player_df['TRACK_ID'] == player_id][['FRAME', 'X', 'Y', 'IS_VISIBLE']]

        player_data['Prev_X'] = player_data['X'].shift(1)
        player_data['Prev_Y'] = player_data['Y'].shift(1)
        player_data['Post_X'] = player_data['X'].shift(-1)
        player_data['Post_Y'] = player_data['Y'].shift(-1)

        player_data['Prev_X'] = player_data['Prev_X'].fillna(player_data['X'])
        player_data['Prev_Y'] = player_data['Prev_Y'].fillna(player_data['Y'])
        player_data['Post_X'] = player_data['Post_X'].fillna(player_data['X'])
        player_data['Post_Y'] = player_data['Post_Y'].fillna(player_data['Y'])

        player_data['Distance'] = np.sqrt(
            (player_data['X'] - player_data['Prev_X']) ** 2 +
            (player_data['Y'] - player_data['Prev_Y']) ** 2)

        player_data['MA_X'] = (player_data['Post_X'] + player_data['Prev_X']) / 2
        player_data['MA_Y'] = (player_data['Post_Y'] + player_data['Prev_Y']) / 2

        if adj_pos:
            player_data['X'] = player_data.apply(lambda x: x['MA_X'] if x['Distance'] > usain_distance else x['X'])
            player_data['Y'] = player_data.apply(lambda x: x['MA_Y'] if x['Distance'] > usain_distance else x['Y'])

        merged_df = pd.merge(player_df[player_df['TRACK_ID'] == player_id], player_data,
                             on='FRAME', how='left', suffixes=('_original', '_new'))

        merged_df['X_new'] = merged_df['X_new'].fillna(merged_df['X_original'])
        merged_df['Y_new'] = merged_df['Y_new'].fillna(merged_df['Y_original'])

        merged_df['playdir'] = merged_df['Post_X'] < merged_df['Prev_X']  # 1 direction of goal
        merged_df.loc[merged_df['FRAME'] >= PERIOD2_FRAME, 'playdir'] =\
            ~merged_df.loc[merged_df['FRAME'] >= PERIOD2_FRAME, 'playdir']

        player_df.loc[player_df['TRACK_ID'] == player_id, ['X', 'Y']] = merged_df[['X_new', 'Y_new']].values
        player_df.loc[player_df['TRACK_ID'] == player_id, 'playdir'] = merged_df['playdir'].values
        player_df.loc[player_df['TRACK_ID'] == player_id, 'distance'] = merged_df['Distance'].values

        player_df['velo'] = player_df['distance'] / avg_frame_length


player_df = pd.merge(player_df, camera_df[['FRAME', 'TIMESTAMP', 'POSSESSION_GROUP']], on='FRAME')
player_df['TIMESTAMP'] = player_df['TIMESTAMP'].apply(lambda x:
                                                      x.hour * 3600 +
                                                      x.minute * 60 +
                                                      x.second +
                                                      x.microsecond / 1e6)
player_df['playdir'] = False
player_df['distance'] = 0.0

smooth()

player_df['velo'] = player_df['velo'].round(3)
player_df['distance'] = player_df['distance'].round(3)

# TODO: smooth velo MA
player_df = player_df[player_df['velo'] <= usain_speed]

home_team_df = player_df[player_df['TRACK_ID'].isin(HOME_TEAM)]


if __name__ == '__main__':
    output_path = 'tracking_prepped.csv'

    home_team_df.to_csv(output_path, index=False)


player_df = player_df[player_df['TRACK_ID'] != 55]

player_df = player_df.sort_values(by=['FRAME'])

tmp_path = 'player_tmp.csv'

if os.path.exists(tmp_path):
    player_df = pd.read_csv(tmp_path)
else:
    frames = player_df['FRAME'].unique()

    for t in frames:
        df_t = player_df.loc[player_df['FRAME'] == t, ['X', 'Y']]

        coords = df_t[['X', 'Y']]
        coords_list = list(coords.itertuples(index=False, name=None))

        D = cdist(coords_list, coords_list)
        D[range(len(D)), range(len(D))] = float('inf')  # set diagonal to inf

        min_dist = np.min(D, axis=1)

        df_t.loc[:, 'MinDist'] = min_dist

        player_df.loc[player_df['FRAME'] == t, 'MinDist'] = df_t['MinDist'].values

    teams = player_df['POSSESSION_GROUP'].dropna().unique().tolist()

    player_df['play_index'] = 0

    prev_possession = 'away team'
    i = 0

    for t in frames:
        possession_t = camera_df.loc[camera_df['FRAME'] == t, 'POSSESSION_GROUP'].values[0]

        if possession_t not in teams:
            player_df.loc[player_df['FRAME'] == t, 'POSSESSION_GROUP'] = prev_possession
            continue

        if possession_t != prev_possession:
            prev_possession = possession_t
            i += 1

        player_df.loc[player_df['FRAME'] == t, 'play_index'] = i

    player_df.to_csv(tmp_path, index=False)
    # TODO: Only opponent


home_poss_df = player_df.copy(deep=True)

pressure_vals = {
    'avg_team_velo': home_poss_df[home_poss_df['TRACK_ID'].isin(HOME_TEAM)].groupby('FRAME')['velo'].mean(),
    'avg_opp_velo': home_poss_df[home_poss_df['TRACK_ID'].isin(AWAY_TEAM)].groupby('FRAME')['velo'].mean(),
    'sum_opp_dist': home_poss_df[home_poss_df['TRACK_ID'].isin(AWAY_TEAM)].groupby('FRAME')['distance'].sum(),
    'avg_min_dist': home_poss_df[home_poss_df['TRACK_ID'].isin(AWAY_TEAM)].groupby('FRAME')['MinDist'].mean()
}

unique_elems_home = set(pressure_vals['avg_team_velo'].index.to_list()) -\
                    set(pressure_vals['avg_opp_velo'].index.to_list())
pressure_vals['avg_team_velo'] = pressure_vals['avg_team_velo'].drop(index=list(unique_elems_home))

pressure_df = pd.DataFrame(index=pressure_vals['avg_team_velo'].index)

unique_elems_opp = set(pressure_vals['avg_opp_velo'].index.to_list()) -\
                   set(pressure_vals['avg_team_velo'].index.to_list())

velo_max = max(pressure_vals['avg_team_velo'].max(), pressure_vals['avg_opp_velo'].max())

for key, val in pressure_vals.items():
    if key != 'avg_team_velo':
        val = val.drop(index=list(unique_elems_opp))

    if key in ['avg_team_velo', 'avg_opp_velo']:
        x_max = velo_max
    else:
        x_max = val.max()

    pressure_df[key] = (val / x_max).values

pressure_df = pressure_df.dropna()

pressure_df['pressure'] = 0.5 * pressure_df['sum_opp_dist'] + \
                          0.25 * pressure_df['avg_min_dist'] + \
                          0.25 * pressure_df['avg_opp_velo']

p_max = pressure_df['pressure'].max()
pressure_df['pressure'] = pressure_df['pressure'] / p_max

pressure_df['FRAME'] = pressure_df.index
pressure_df.reset_index(drop=True, inplace=True)

del pressure_df['sum_opp_dist']
del pressure_df['avg_min_dist']

pressure_df = pd.merge(pressure_df, player_df[['FRAME', 'POSSESSION_GROUP', 'play_index']], on='FRAME')\
    .drop_duplicates(subset='FRAME')

pressure_per_play_df = pressure_df[['avg_team_velo', 'avg_opp_velo', 'pressure', 'play_index']].groupby('play_index')\
    .mean()
play_possession_map = pressure_df[['play_index', 'POSSESSION_GROUP']].drop_duplicates()

pressure_df = pd.merge(pressure_per_play_df, play_possession_map, on='play_index').drop_duplicates(subset='play_index')

if __name__ == '__main__':
    output_path = 'pressure.csv'

    pressure_df.to_csv(output_path, index=False)
