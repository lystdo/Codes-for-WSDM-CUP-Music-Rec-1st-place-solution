import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

## load the data
tr = pd.read_csv('../temporal_data/train_id_cnt.csv')
te = pd.read_csv('../temporal_data/test_id_cnt.csv')
member = pd.read_csv('../temporal_data/members_id_cnt.csv')
song = pd.read_csv('../temporal_data/songs_id_cnt_isrc.csv')

concat = tr[['msno', 'song_id']].append(te[['msno', 'song_id']])
member_cnt = concat['msno'].max() + 1
song_cnt = concat['song_id'].max() + 1
artist_cnt = int(song['artist_name'].max() + 1)

## svd for user-song pairs
n_component = 48

print(len(concat))

data = np.ones(len(concat))
msno = concat['msno'].values
song_id = concat['song_id'].values

rating = sparse.coo_matrix((data, (msno, song_id)))
rating = (rating > 0) * 1.0

[u, s, vt] = svds(rating, k=n_component)
print(s[::-1])
s_song = np.diag(s[::-1])

members_topics = pd.DataFrame(u[:, ::-1])
members_topics.columns = ['member_component_%d'%i for i in range(n_component)]
members_topics['msno'] = range(member_cnt)
member = member.merge(members_topics, on='msno', how='right')

song_topics = pd.DataFrame(vt.transpose()[:, ::-1])
song_topics.columns = ['song_component_%d'%i for i in range(n_component)]
song_topics['song_id'] = range(song_cnt)
song = song.merge(song_topics, on='song_id', how='right')

## svd for user-artist pairs
n_component = 16

concat = concat.merge(song[['song_id', 'artist_name']], on='song_id', how='left')
concat = concat[concat['artist_name'] >= 0]
msno = concat['msno'].values
artist = concat['artist_name'].values.astype(int)

print(len(concat))
data = np.ones(len(concat))
rating_tmp = sparse.coo_matrix((data, (msno, artist)))

rating = np.log1p(rating_tmp) * 0.3 + (rating_tmp > 0) * 1.0

[u, s, vt] = svds(rating, k=n_component)
print(s[::-1])
s_artist = np.diag(s[::-1])

members_topics = pd.DataFrame(u[:, ::-1])
members_topics.columns = ['member_artist_component_%d'%i for i in range(n_component)]
members_topics['msno'] = range(member_cnt)
member = member.merge(members_topics, on='msno', how='left')


artist_topics = pd.DataFrame(vt.transpose()[:, ::-1])
artist_topics.columns = ['artist_component_%d'%i for i in range(n_component)]
artist_topics['artist_name'] = range(artist_cnt)
song = song.merge(artist_topics, on='artist_name', how='left')

## dot features
member = member.sort_values(by='msno')
song = song.sort_values(by='song_id')

mem_cols = ['member_component_%d'%i for i in range(48)]
song_cols = ['song_component_%d'%i for i in range(48)]

member_embeddings = member[mem_cols].values
song_embeddings = song[song_cols].values

mem_cols = ['member_artist_component_%d'%i for i in range(16)]
song_cols = ['artist_component_%d'%i for i in range(16)]

member_artist_embeddings = member[mem_cols].values
song_artist_embeddings = song[song_cols].values

train_dot = np.zeros((len(tr), 2))
test_dot = np.zeros((len(te), 2))

for i in range(len(tr)):
    msno_idx = tr['msno'].values[i]
    song_idx = tr['song_id'].values[i]
    
    train_dot[i, 0] = np.dot(member_embeddings[msno_idx], np.dot(s_song, song_embeddings[song_idx]))
    train_dot[i, 1] = np.dot(member_artist_embeddings[msno_idx], np.dot(s_artist, song_artist_embeddings[song_idx]))

for i in range(len(te)):
    msno_idx = te['msno'].values[i]
    song_idx = te['song_id'].values[i]
    
    test_dot[i, 0] = np.dot(member_embeddings[msno_idx], np.dot(s_song, song_embeddings[song_idx]))
    test_dot[i, 1] = np.dot(member_artist_embeddings[msno_idx], np.dot(s_artist, song_artist_embeddings[song_idx]))

tr['song_embeddings_dot'] = train_dot[:, 0]
tr['artist_embeddings_dot'] = train_dot[:, 1]

te['song_embeddings_dot'] = test_dot[:, 0]
te['artist_embeddings_dot'] = test_dot[:, 1]

## write to files
tr.to_csv('../temporal_data/train_id_cnt_svd.csv', index=False)
te.to_csv('../temporal_data/test_id_cnt_svd.csv', index=False)
member.to_csv('../temporal_data/members_id_cnt_svd.csv', index=False)
song.to_csv('../temporal_data/songs_id_cnt_isrc_svd.csv', index=False)
