import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

## load the data
train = pd.read_csv('../temporal_data/train_id.csv')
test = pd.read_csv('../temporal_data/test_id.csv')
song = pd.read_csv('../temporal_data/songs_id_cnt.csv')

data = train[['msno', 'song_id']].append(test[['msno', 'song_id']])

print('Data loaded.')

## isrc process
isrc = song['isrc']
song['cc'] = isrc.str.slice(0, 2)
song['xxx'] = isrc.str.slice(2, 5)
song['yy'] = isrc.str.slice(5, 7).astype(float)
song['yy'] = song['yy'].apply(lambda x: 2000+x if x < 18 else 1900+x)

song['cc'] = LabelEncoder().fit_transform(song['cc'])
song['xxx'] = LabelEncoder().fit_transform(song['xxx'])
song['isrc_missing'] = (song['cc'] == 0) * 1.0

## song_cnt
song_cc_cnt = song.groupby(by='cc').count()['song_id'].to_dict()
song_cc_cnt[0] = None
song['cc_song_cnt'] = song['cc'].apply(lambda x: song_cc_cnt[x] if not np.isnan(x) else None)

song_xxx_cnt = song.groupby(by='xxx').count()['song_id'].to_dict()
song_xxx_cnt[0] = None
song['xxx_song_cnt'] = song['xxx'].apply(lambda x: song_xxx_cnt[x] if not np.isnan(x) else None)

song_yy_cnt = song.groupby(by='yy').count()['song_id'].to_dict()
song_yy_cnt[0] = None
song['yy_song_cnt'] = song['yy'].apply(lambda x: song_yy_cnt[x] if not np.isnan(x) else None)

data = data.merge(song, on='song_id', how='left')

song_cc_cnt = data.groupby(by='cc').count()['msno'].to_dict()
song_cc_cnt[0] = None
song['cc_rec_cnt'] = song['cc'].apply(lambda x: song_cc_cnt[x] if not np.isnan(x) else None)

song_xxx_cnt = data.groupby(by='xxx').count()['msno'].to_dict()
song_xxx_cnt[0] = None
song['xxx_rec_cnt'] = song['xxx'].apply(lambda x: song_xxx_cnt[x] if not np.isnan(x) else None)

song_yy_cnt = data.groupby(by='yy').count()['msno'].to_dict()
song_yy_cnt[0] = None
song['yy_rec_cnt'] = song['yy'].apply(lambda x: song_yy_cnt[x] if not np.isnan(x) else None)

## to_csv
features = ['cc_song_cnt', 'xxx_song_cnt', 'yy_song_cnt', 'cc_rec_cnt', \
        'xxx_rec_cnt', 'yy_rec_cnt']
for feat in features:
    song[feat] = np.log1p(song[feat])

song.drop(['name', 'isrc'], axis=1, inplace=True)
song.to_csv('../temporal_data/songs_id_cnt_isrc.csv', index=False)
