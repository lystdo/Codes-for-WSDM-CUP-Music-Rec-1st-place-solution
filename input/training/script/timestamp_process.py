import numpy as np
import pandas as pd
from collections import defaultdict

## load the data
tr = pd.read_csv('../temporal_data/train_id_cnt_svd.csv')
te = pd.read_csv('../temporal_data/test_id_cnt_svd.csv')
mem = pd.read_csv('../temporal_data/members_id_cnt_svd.csv')
song = pd.read_csv('../temporal_data/songs_id_cnt_isrc_svd.csv')

## continous index
concat = tr[['msno', 'song_id']].append(te[['msno', 'song_id']])
concat['timestamp'] = range(len(concat))

## windows_based count
window_sizes = [10, 25, 500, 5000, 10000, 50000]

msno_list = concat['msno'].values
song_list = concat['song_id'].values

def get_window_cnt(values, idx, window_size):
    lower = max(0, idx-window_size)
    upper = min(len(values), idx+window_size)
    return (values[lower:idx] == values[idx]).sum(), (values[idx:upper] == values[idx]).sum()

for window_size in window_sizes:
    msno_before_cnt = np.zeros(len(concat))
    song_before_cnt = np.zeros(len(concat))
    msno_after_cnt = np.zeros(len(concat))
    song_after_cnt = np.zeros(len(concat))
    for i in range(len(concat)):
        msno_before_cnt[i], msno_after_cnt[i] = get_window_cnt(msno_list, i, window_size)
        song_before_cnt[i], song_after_cnt[i] = get_window_cnt(song_list, i, window_size)
    concat['msno_%d_before_cnt'%window_size] = msno_before_cnt
    concat['song_%d_before_cnt'%window_size] = song_before_cnt
    concat['msno_%d_after_cnt'%window_size] = msno_after_cnt
    concat['song_%d_after_cnt'%window_size] = song_after_cnt
    
    print('Window size for %d done.'%window_size)

## till_now count
msno_dict = defaultdict(lambda: 0)
song_dict = defaultdict(lambda: 0)

msno_till_now_cnt = np.zeros(len(concat))
song_till_now_cnt = np.zeros(len(concat))
for i in range(len(concat)):
    msno_till_now_cnt[i] = msno_dict[msno_list[i]]
    msno_dict[msno_list[i]] += 1
    
    song_till_now_cnt[i] = song_dict[song_list[i]]
    song_dict[song_list[i]] += 1

concat['msno_till_now_cnt'] = msno_till_now_cnt
concat['song_till_now_cnt'] = song_till_now_cnt

print('Till-now count done.')

## varience
def timestamp_map(x):
    if x < 7377418:
        x = (x - 0.0) / (7377417.0 - 0.0) * (1484236800.0 - 1471190400.0) + 1471190400.0
    else:
        x = (x - 7377417.0) / (9934207.0 - 7377417.0) * (1488211200.0 - 1484236800.0) + 1484236800.0

    return x
    
concat['timestamp'] = concat['timestamp'].apply(timestamp_map)

msno_mean = concat.groupby(by='msno').mean()['timestamp'].to_dict()
mem['msno_timestamp_mean'] = mem['msno'].apply(lambda x: msno_mean[x])

msno_std = concat.groupby(by='msno').std()['timestamp'].to_dict()
mem['msno_timestamp_std'] = mem['msno'].apply(lambda x: msno_std[x])

song_mean = concat.groupby(by='song_id').mean()['timestamp'].to_dict()
song['song_timestamp_mean'] = song['song_id'].apply(lambda x: song_mean[x])

song_std = concat.groupby(by='song_id').std()['timestamp'].to_dict()
song['song_timestamp_std'] = song['song_id'].apply(lambda x: song_std[x])

print('Varience done.')

## save to files
features = ['msno_till_now_cnt', 'song_till_now_cnt']
for window_size in window_sizes:
    features += ['msno_%d_before_cnt'%window_size, 'song_%d_before_cnt'%window_size, \
            'msno_%d_after_cnt'%window_size, 'song_%d_after_cnt'%window_size]
for feat in features:
    concat[feat] = np.log1p(concat[feat])

features = ['timestamp'] + features

data = concat[features].values
for i in range(len(features)):
    tr[features[i]] = data[:len(tr), i]
    te[features[i]] = data[len(tr):, i]

tr.to_csv('../temporal_data/train_id_cnt_svd_stamp.csv', index=False)
te.to_csv('../temporal_data/test_id_cnt_svd_stamp.csv', index=False)
mem.to_csv('../temporal_data/members_id_cnt_svd_stamp.csv', index=False)
song.to_csv('../temporal_data/songs_id_cnt_isrc_svd_stamp.csv', index=False)
