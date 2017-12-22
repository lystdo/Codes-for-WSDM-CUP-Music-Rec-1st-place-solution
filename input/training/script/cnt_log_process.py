import numpy as np
import pandas as pd

train = pd.read_csv('../temporal_data/train_id.csv')
test = pd.read_csv('../temporal_data/test_id.csv')
member = pd.read_csv('../temporal_data/members_id.csv')
song_origin = pd.read_csv('../temporal_data/songs_id.csv')
song_extra = pd.read_csv('../temporal_data/songs_extra_id.csv')

song = pd.DataFrame({'song_id': range(max(train.song_id.max(), test.song_id.max())+1)})
song = song.merge(song_origin, on='song_id', how='left')
song = song.merge(song_extra, on='song_id', how='left')

data = train[['msno', 'song_id']].append(test[['msno', 'song_id']])

## member_cnt
mem_rec_cnt = data.groupby(by='msno').count()['song_id'].to_dict()
member['msno_rec_cnt'] = member['msno'].apply(lambda x: mem_rec_cnt[x])

member['bd'] = member['bd'].apply(lambda x: np.nan if x <= 0 or x >= 75 else x)

## song_cnt
artist_song_cnt = song.groupby(by='artist_name').count()['song_id'].to_dict()
song['artist_song_cnt'] = song['artist_name'].apply(lambda x: artist_song_cnt[x] if not np.isnan(x) else np.nan)

composer_song_cnt = song.groupby(by='composer').count()['song_id'].to_dict()
composer_song_cnt[0] = np.nan
song['composer_song_cnt'] = song['composer'].apply(lambda x: composer_song_cnt[x] if not np.isnan(x) else np.nan)

lyricist_song_cnt = song.groupby(by='lyricist').count()['song_id'].to_dict()
lyricist_song_cnt[0] = np.nan
song['lyricist_song_cnt'] = song['lyricist'].apply(lambda x: lyricist_song_cnt[x] if not np.isnan(x) else np.nan)

genre_song_cnt = song.groupby(by='first_genre_id').count()['song_id'].to_dict()
genre_song_cnt[0] = np.nan
song['genre_song_cnt'] = song['first_genre_id'].apply(lambda x: genre_song_cnt[x] if not np.isnan(x) else np.nan)

data = data.merge(song, on='song_id', how='left')

song_rec_cnt = data.groupby(by='song_id').count()['msno'].to_dict()
song['song_rec_cnt'] = song['song_id'].apply(lambda x: song_rec_cnt[x] if not np.isnan(x) else np.nan)

artist_rec_cnt = data.groupby(by='artist_name').count()['msno'].to_dict()
song['artist_rec_cnt'] = song['artist_name'].apply(lambda x: artist_rec_cnt[x] if not np.isnan(x) else np.nan)

composer_rec_cnt = data.groupby(by='composer').count()['msno'].to_dict()
composer_rec_cnt[0] = np.nan
song['composer_rec_cnt'] = song['composer'].apply(lambda x: composer_rec_cnt[x] if not np.isnan(x) else np.nan)

lyricist_rec_cnt = data.groupby(by='lyricist').count()['msno'].to_dict()
lyricist_rec_cnt[0] = np.nan
song['lyricist_rec_cnt'] = song['lyricist'].apply(lambda x: lyricist_rec_cnt[x] if not np.isnan(x) else np.nan)

genre_rec_cnt = data.groupby(by='first_genre_id').count()['msno'].to_dict()
genre_rec_cnt[0] = np.nan
song['genre_rec_cnt'] = song['first_genre_id'].apply(lambda x: genre_rec_cnt[x] if not np.isnan(x) else np.nan)

## msno context features
dummy_feat = ['source_system_tab', 'source_screen_name', 'source_type']
concat = train.drop('target', axis=1).append(test.drop('id', axis=1))

for feat in dummy_feat:
    feat_dummies = pd.get_dummies(concat[feat])
    feat_dummies.columns = ['msno_%s_'%feat + '%s'%col for col in feat_dummies.columns]
    feat_dummies['msno'] = concat['msno'].values
    feat_dummies = feat_dummies.groupby('msno').mean()
    feat_dummies['msno'] = feat_dummies.index
    member = member.merge(feat_dummies, on='msno', how='left')

train_temp = train.merge(member, on='msno', how='left')
test_temp = test.merge(member, on='msno', how='left')

train['msno_source_system_tab_prob'] = train_temp[[col for col in train_temp.columns if 'source_system_tab' in col]].apply(lambda x: \
        x['msno_source_system_tab_%d'%x['source_system_tab']], axis=1)
test['msno_source_system_tab_prob'] = test_temp[[col for col in test_temp.columns if 'source_system_tab' in col]].apply(lambda x: \
        x['msno_source_system_tab_%d'%x['source_system_tab']], axis=1)

train['msno_source_screen_name_prob'] = train_temp[[col for col in train_temp.columns if 'source_screen_name' in col]].apply(lambda x: \
        x['msno_source_screen_name_%d'%x['source_screen_name']], axis=1)
test['msno_source_screen_name_prob'] = test_temp[[col for col in test_temp.columns if 'source_screen_name' in col]].apply(lambda x: \
        x['msno_source_screen_name_%d'%x['source_screen_name']], axis=1)

train['msno_source_type_prob'] = train_temp[[col for col in train_temp.columns if 'source_type' in col]].apply(lambda x: \
        x['msno_source_type_%d'%x['source_type']], axis=1)
test['msno_source_type_prob'] = test_temp[[col for col in test_temp.columns if 'source_type' in col]].apply(lambda x: \
        x['msno_source_type_%d'%x['source_type']], axis=1)

## to_csv
features = ['msno_rec_cnt']
for feat in features:
    member[feat] = np.log1p(member[feat])
member.to_csv('../temporal_data/members_id_cnt.csv', index=False)

features = ['song_length', 'song_rec_cnt', 'artist_song_cnt', 'composer_song_cnt', \
        'lyricist_song_cnt', 'genre_song_cnt', 'artist_rec_cnt', \
        'composer_rec_cnt', 'lyricist_rec_cnt', 'genre_rec_cnt']
for feat in features:
    song[feat] = np.log1p(song[feat])
#song['song_length'] = np.log1p(song['song_length'])
song.to_csv('../temporal_data/songs_id_cnt.csv', index=False)

train.to_csv('../temporal_data/train_id_cnt.csv', index=False)
test.to_csv('../temporal_data/test_id_cnt.csv', index=False)
