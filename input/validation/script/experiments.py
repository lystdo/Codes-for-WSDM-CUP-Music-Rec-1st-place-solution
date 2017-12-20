import pandas as pd
import numpy as np

tr = pd.read_csv('../train.csv')
te = pd.read_csv('../test.csv')
song = pd.read_csv('../songs_nn.csv')

concat = tr[['msno', 'song_id', 'source_system_tab', 'source_screen_name', \
        'source_type']].append(te[['msno', 'song_id', 'source_system_tab', \
        'source_screen_name', 'source_type']])
concat = concat.merge(song[['song_id', 'song_length', 'artist_name', 'first_genre_id', \
        'artist_rec_cnt', 'song_rec_cnt', 'artist_song_cnt', 'xxx', 'yy', \
        'language']], on='song_id', how='left')

concat['source'] = concat['source_system_tab'] * 10000 + concat['source_screen_name'] * 100 + \
        concat['source_type']
from sklearn.preprocessing import LabelEncoder
concat['source'] = LabelEncoder().fit_transform(concat['source'].values)

## member features

mem_add = pd.DataFrame({'msno': range(concat['msno'].max()+1)})
data_avg = concat[['msno', 'song_length', 'artist_song_cnt', \
        'artist_rec_cnt', 'song_rec_cnt', 'yy']].groupby('msno').mean()
data_avg.columns = ['msno_'+i+'_mean' for i in data_avg.columns]
data_avg['msno'] = data_avg.index.values
mem_add = mem_add.merge(data_avg, on='msno', how='left')

data_std = concat[['msno', 'song_length', 'artist_song_cnt', \
        'artist_rec_cnt', 'song_rec_cnt', 'yy']].groupby('msno').std()
data_std.columns = ['msno_'+i+'_std' for i in data_std.columns]
data_std['msno'] = data_std.index.values
mem_add = mem_add.merge(data_std, on='msno', how='left')

artist_msno = concat[['msno', 'artist_name']].groupby('msno').apply(lambda x: len(set(x['artist_name'].values)))
mem_add['artist_msno_cnt'] = artist_msno
mem_add['artist_msno_cnt'] = np.log1p(mem_add['artist_msno_cnt'])

language_dummy = pd.get_dummies(concat['language'])
language_dummy['msno'] = concat['msno'].values
language_prob = language_dummy.groupby('msno').mean()
language_prob.columns = ['msno_language_%d'%i for i in language_prob.columns]
language_prob['msno'] = language_prob.index
mem_add = mem_add.merge(language_prob, on='msno', how='left')

mem_add.to_csv('../members_add.csv', index=False)

## train/test features

col = ['artist_name', 'first_genre_id', 'xxx', 'language', 'yy', 'source']
for feat in col:
    concat['id'] = concat['msno'] * 100000 + concat[feat]
    id_cnt = concat[['msno', 'id']].groupby('id').count().to_dict()['msno']
    concat['msno_'+feat+'_cnt'] = concat['id'].apply(lambda x: id_cnt[x])

msno_cnt = concat[['msno', 'song_id']].groupby('msno').count().to_dict()['song_id']
concat['msno_cnt'] = concat['msno'].apply(lambda x: msno_cnt[x])
for feat in col:
    concat['msno_'+feat+'_prob'] = concat['msno_'+feat+'_cnt'] / concat['msno_cnt']

cols = ['source_system_tab', 'source_screen_name', 'source_type']
for col in cols:
    concat['id'] = concat['song_id'] * 10000 + concat[col]
    id_cnt = concat[['msno', 'id']].groupby('id').count().to_dict()['msno']
    concat['song_'+col+'_cnt'] = concat['id'].apply(lambda x: id_cnt[x])

song_cnt = concat[['msno', 'song_id']].groupby('song_id').count().to_dict()['msno']
concat['song_cnt'] = concat['song_id'].apply(lambda x: song_cnt[x])

for col in cols:
    concat['song_'+col+'_prob'] = concat['song_'+col+'_cnt'] / concat['song_cnt']

result = concat[['msno_artist_name_prob', 'msno_first_genre_id_prob', 'msno_xxx_prob', \
        'msno_language_prob', 'msno_yy_prob', 'song_source_system_tab_prob', \
        'song_source_screen_name_prob', 'song_source_type_prob', 'source', 'msno_source_prob']]

result[:len(tr)].to_csv('../train_add.csv', index=False)
result[len(tr):].to_csv('../test_add.csv', index=False)

