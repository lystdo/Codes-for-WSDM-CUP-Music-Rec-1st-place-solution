import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

## load the data
members = pd.read_csv('../source_data/members.csv')
songs = pd.read_csv('../source_data/songs.csv')
songs_extra = pd.read_csv('../source_data/song_extra_info.csv')
train = pd.read_csv('../source_data/train.csv')
test = pd.read_csv('../source_data/test.csv')

song_id_set = set(train['song_id'].append(test['song_id']))

songs['appeared'] = songs['song_id'].apply(lambda x: True if x in song_id_set else False)
songs = songs[songs.appeared]
songs.drop('appeared', axis=1, inplace=True)

songs_extra['appeared'] = songs_extra['song_id'].apply(lambda x: True if x in song_id_set else False)
songs_extra = songs_extra[songs_extra.appeared]
songs_extra.drop('appeared', axis=1, inplace=True)

msno_set = set(train['msno'].append(test['msno']))

members['appeared'] = members['msno'].apply(lambda x: True if x in msno_set else False)
members = members[members.appeared]
members.drop('appeared', axis=1, inplace=True)

print('Data loaded.')

## preprocess msno and song_id
msno_encoder = LabelEncoder()
msno_encoder.fit(members['msno'].values)
members['msno'] = msno_encoder.transform(members['msno'])
train['msno'] = msno_encoder.transform(train['msno'])
test['msno'] = msno_encoder.transform(test['msno'])

print('MSNO done.')

song_id_encoder = LabelEncoder()
song_id_encoder.fit(train['song_id'].append(test['song_id']))
songs['song_id'] = song_id_encoder.transform(songs['song_id'])
songs_extra['song_id'] = song_id_encoder.transform(songs_extra['song_id'])
train['song_id'] = song_id_encoder.transform(train['song_id'])
test['song_id'] = song_id_encoder.transform(test['song_id'])

print('Song_id done.')

## preprocess the features in train.csv & test.csv
columns = ['source_system_tab', 'source_screen_name', 'source_type']
for column in columns:
    column_encoder = LabelEncoder()
    column_encoder.fit(train[column].append(test[column]))
    train[column] = column_encoder.transform(train[column])
    test[column] = column_encoder.transform(test[column])

print('Source information done.')

## preprocess the features in members.csv
columns = ['city', 'gender', 'registered_via']
for column in columns:
    column_encoder = LabelEncoder()
    column_encoder.fit(members[column])
    members[column] = column_encoder.transform(members[column])

members['registration_init_time'] = members['registration_init_time'].apply(lambda x: \
        time.mktime(time.strptime(str(x),'%Y%m%d')))
members['expiration_date'] = members['expiration_date'].apply(lambda x: \
        time.mktime(time.strptime(str(x),'%Y%m%d')))

print('Members information done.')

## preprocess the features in songs.csv
genre_id = np.zeros((len(songs), 4))
for i in range(len(songs)):
    if not isinstance(songs['genre_ids'].values[i], basestring):
        continue
    ids = str(songs['genre_ids'].values[i]).split('|')
    if len(ids) > 2:
        genre_id[i, 0] = int(ids[0])
        genre_id[i, 1] = int(ids[1])
        genre_id[i, 2] = int(ids[2])
    elif len(ids) > 1:
        genre_id[i, 0] = int(ids[0])
        genre_id[i, 1] = int(ids[1])
    elif len(ids) == 1:
        genre_id[i, 0] = int(ids[0])
    genre_id[i, 3] = len(ids)
songs['first_genre_id'] = genre_id[:, 0]
songs['second_genre_id'] = genre_id[:, 1]
songs['third_genre_id'] = genre_id[:, 2]
songs['genre_id_cnt'] = genre_id[:, 3]

genre_encoder = LabelEncoder()
genre_encoder.fit((songs.first_genre_id.append(songs.second_genre_id)).append(songs.third_genre_id))
songs['first_genre_id'] = genre_encoder.transform(songs['first_genre_id'])
songs['second_genre_id'] = genre_encoder.transform(songs['second_genre_id'])
songs['third_genre_id'] = genre_encoder.transform(songs['third_genre_id'])
songs.drop('genre_ids', axis=1, inplace=True)

def artist_count(x):
    return x.count('and') + x.count(',') + x.count(' feat') + x.count('&') + 1

songs['artist_cnt'] = songs['artist_name'].apply(artist_count).astype(np.int8)

def get_count(x):
    try:
        return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
    except:
        return 0

songs['lyricist_cnt'] = songs['lyricist'].apply(get_count).astype(np.int8)
songs['composer_cnt'] = songs['composer'].apply(get_count).astype(np.int8)

songs['is_featured'] = songs['artist_name'].apply(lambda x: 1 if ' feat' \
        in str(x) else 0).astype(np.int8)

def get_first_artist(x):
    if x.count('and') > 0:
        x = x.split('and')[0]
    if x.count(',') > 0:
        x = x.split(',')[0]
    if x.count(' feat') > 0:
        x = x.split(' feat')[0]
    if x.count('&') > 0:
        x = x.split('&')[0]
    return x.strip()

songs['artist_name'] = songs['artist_name'].apply(get_first_artist)
    
def get_first_term(x):
    try:
        if x.count('|') > 0:
            x = x.split('|')[0]
        if x.count('/') > 0:
            x = x.split('/')[0]
        if x.count('\\') > 0:
            x = x.split('\\')[0]
        if x.count(';') > 0:
            x = x.split(';')[0]
        return x.strip()
    except:
        return x

songs['lyricist'] = songs['lyricist'].apply(get_first_term)
songs['composer'] = songs['composer'].apply(get_first_term)        

songs['language'] = songs['language'].fillna(-1)
columns = ['artist_name', 'lyricist', 'composer', 'language']
for column in columns:
    column_encoder = LabelEncoder()
    column_encoder.fit(songs[column])
    songs[column] = column_encoder.transform(songs[column])

## save files
members.to_csv('../temporal_data/members_id.csv', index=False)
songs.to_csv('../temporal_data/songs_id.csv', index=False)
songs_extra.to_csv('../temporal_data/songs_extra_id.csv', index=False)
train.to_csv('../temporal_data/train_id.csv', index=False)
test.to_csv('../temporal_data/test_id.csv', index=False)

