import os
import numpy as np
import pandas as pd

## load the data
train = pd.read_csv('../temporal_data/train_id_cnt_svd_stamp_before_after.csv')
test = pd.read_csv('../temporal_data/test_id_cnt_svd_stamp_before_after.csv')
member = pd.read_csv('../temporal_data/members_id_cnt_svd_stamp.csv')
song = pd.read_csv('../temporal_data/songs_id_cnt_isrc_svd_stamp.csv')

## prepare data for train / test
train.to_csv('../train.csv', index=False, float_format='%.6f')
test.to_csv('../test.csv', index=False, float_format='%.6f')

'''
train['iid'] = train['song_id'] * 100000 + train['msno']
test['iid'] = test['song_id'] * 100000 + test['msno']

iid_set = set(test['iid'].values)
train['appeared'] = train['iid'].apply(lambda x: x in iid_set)
train = train[train['appeared'] == False]

train.drop(['iid', 'appeared'], axis=1, inplace=True)
train.to_csv('../train_part.csv', index=False, float_format='%.6f')
'''

## prepare data for member / song for GBDT
member.to_csv('../members_gbdt.csv', index=False)

columns = ['composer', 'lyricist', 'language', 'first_genre_id', 'second_genre_id', 'third_genre_id']
for col in columns:
    song[col].fillna(0, inplace=True)
    song[col] = song[col].astype(int)
song['artist_name'].fillna(np.max(song['artist_name'])+1, inplace=True)
song['artist_name'] = song['artist_name'].astype(int)
song['isrc_missing'] = song['isrc_missing'].astype(int)
song.to_csv('../songs_gbdt.csv', index=False)

## prepare data for member / song for NN
member['bd_missing'] = np.isnan(member['bd'].values) * 1

columns = ['bd']
for col in columns:
    member[col].fillna(np.nanmean(member[col]), inplace=True)

member['msno_timestamp_std'].fillna(np.nanmin(member['msno_timestamp_std']), inplace=True)
member.to_csv('../members_nn.csv', index=False)

song['song_id_missing'] = np.isnan(song['song_length'].values) * 1

columns = ['song_length', 'genre_id_cnt', 'artist_song_cnt', 'composer_song_cnt', \
       'lyricist_song_cnt', 'genre_song_cnt', 'song_rec_cnt', \
       'artist_rec_cnt', 'composer_rec_cnt', 'lyricist_rec_cnt', \
       'genre_rec_cnt', 'yy', 'cc_song_cnt', \
       'xxx_song_cnt', 'yy_song_cnt', 'cc_rec_cnt', 'xxx_rec_cnt', \
       'yy_rec_cnt', 'song_timestamp_std', 'artist_cnt', 'lyricist_cnt', \
       'composer_cnt', 'is_featured'] + ['artist_component_%d'%i for i in range(16)]
for col in columns:
    song[col].fillna(np.nanmean(song[col]), inplace=True)

song.to_csv('../songs_nn.csv', index=False)

