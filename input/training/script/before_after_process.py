import numpy as np
import pandas as pd
from collections import defaultdict

## load data
tr = pd.read_csv('../temporal_data/train_id_cnt_svd_stamp.csv')
te = pd.read_csv('../temporal_data/test_id_cnt_svd_stamp.csv')

print('data loaded.')
print(len(tr))
print(len(te))

## continous index
concat = tr[['msno', 'song_id', 'source_type', 'source_screen_name', 'timestamp']].append(te[['msno', \
        'song_id', 'source_type', 'source_screen_name', 'timestamp']])

## before data
song_dict = defaultdict(lambda: None)
type_dict = defaultdict(lambda: None)
name_dict = defaultdict(lambda: None)
time_dict = defaultdict(lambda: None)

before_data = np.zeros((len(concat), 4))
for i in range(len(concat)):
    msno = concat['msno'].values[i]
    
    if(song_dict[msno] == None):
        before_data[i] = concat[['song_id', 'source_type', 'source_screen_name', 'timestamp']].values[i]
        before_data[i, 3] = np.nan
    else:
        before_data[i, 0] = song_dict[msno]
        before_data[i, 1] = type_dict[msno]
        before_data[i, 2] = name_dict[msno]
        before_data[i, 3] = time_dict[msno]

    song_dict[msno] = concat['song_id'].values[i]
    type_dict[msno] = concat['source_type'].values[i]
    name_dict[msno] = concat['source_screen_name'].values[i]
    time_dict[msno] = concat['timestamp'].values[i]

print('data before done.')

## after data
song_dict = defaultdict(lambda: None)
type_dict = defaultdict(lambda: None)
name_dict = defaultdict(lambda: None)
time_dict = defaultdict(lambda: None)

after_data = np.zeros((len(concat), 4))
for i in range(len(concat))[::-1]:
    msno = concat['msno'].values[i]
    
    if(song_dict[msno] == None):
        after_data[i] = concat[['song_id', 'source_type', 'source_screen_name', 'timestamp']].values[i]
        after_data[i, 3] = np.nan
    else:
        after_data[i, 0] = song_dict[msno]
        after_data[i, 1] = type_dict[msno]
        after_data[i, 2] = name_dict[msno]
        after_data[i, 3] = time_dict[msno]

    song_dict[msno] = concat['song_id'].values[i]
    type_dict[msno] = concat['source_type'].values[i]
    name_dict[msno] = concat['source_screen_name'].values[i]
    time_dict[msno] = concat['timestamp'].values[i]

print('data after done.')

## to_csv
idx = 0
for i in ['song_id', 'source_type', 'source_screen_name', 'timestamp']:
    tr['before_'+i] = before_data[:len(tr), idx]
    tr['after_'+i] = after_data[:len(tr), idx]
    
    te['before_'+i] = before_data[len(tr):, idx]
    te['after_'+i] = after_data[len(tr):, idx]
    
    idx += 1

for i in ['song_id', 'source_type', 'source_screen_name']:
    tr['before_'+i] = tr['before_'+i].astype(int)
    te['before_'+i] = te['before_'+i].astype(int)
    tr['after_'+i] = tr['after_'+i].astype(int)
    te['after_'+i] = te['after_'+i].astype(int)

tr['before_timestamp'] = np.log1p(tr['timestamp'] - tr['before_timestamp'])
te['before_timestamp'] = np.log1p(te['timestamp'] - te['before_timestamp'])

tr['after_timestamp'] = np.log1p(tr['after_timestamp'] - tr['timestamp'])
te['after_timestamp'] = np.log1p(te['after_timestamp'] - te['timestamp'])

tr['before_timestamp'].fillna(np.nanmean(tr['before_timestamp']), inplace=True)
te['before_timestamp'].fillna(np.nanmean(te['before_timestamp']), inplace=True)
tr['after_timestamp'].fillna(np.nanmean(tr['after_timestamp']), inplace=True)
te['after_timestamp'].fillna(np.nanmean(te['after_timestamp']), inplace=True)

tr.to_csv('../temporal_data/train_id_cnt_svd_stamp_before_after.csv', index=False)
te.to_csv('../temporal_data/test_id_cnt_svd_stamp_before_after.csv', index=False)
