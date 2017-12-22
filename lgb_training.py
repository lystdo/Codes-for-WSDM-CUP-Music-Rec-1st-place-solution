import gc
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

#####################################################
## Data Loading
#####################################################

folder = 'training'

## load data
if folder == 'training':
    train = pd.read_csv('./input/%s/train_part.csv'%folder)
    train_add = pd.read_csv('./input/%s/train_part_add.csv'%folder)
elif folder == 'validation':
    train = pd.read_csv('./input/%s/train.csv'%folder)
    train_add = pd.read_csv('./input/%s/train_add.csv'%folder)
train_y = train['target']
train.drop(['target'], inplace=True, axis=1)

test = pd.read_csv('./input/%s/test.csv'%folder)
test_add = pd.read_csv('./input/%s/test_add.csv'%folder)
test_id = test['id']
test.drop(['id'], inplace=True, axis=1)

train_add['source'] = train_add['source'].astype('category')
test_add['source'] = test_add['source'].astype('category')

cols = ['msno_artist_name_prob', 'msno_first_genre_id_prob', 'msno_xxx_prob', \
        'msno_language_prob', 'msno_yy_prob', 'source', 'msno_source_prob', \
        'song_source_system_tab_prob', 'song_source_screen_name_prob', \
        'song_source_type_prob']
for col in cols:
    train[col] = train_add[col].values
    test[col] = test_add[col].values

## merge data
member = pd.read_csv('./input/%s/members_gbdt.csv'%folder)

train = train.merge(member, on='msno', how='left')
test = test.merge(member, on='msno', how='left')

del member
gc.collect()

member_add = pd.read_csv('./input/%s/members_add.csv'%folder)

cols = ['msno', 'msno_song_length_mean', 'artist_msno_cnt']
train = train.merge(member_add[cols], on='msno', how='left')
test = test.merge(member_add[cols], on='msno', how='left')

del member_add
gc.collect()

song = pd.read_csv('./input/%s/songs_gbdt.csv'%folder)

train = train.merge(song, on='song_id', how='left')
test = test.merge(song, on='song_id', how='left')

cols = song.columns

song.columns = ['before_'+i for i in cols]
train = train.merge(song, on='before_song_id', how='left')
test = test.merge(song, on='before_song_id', how='left')

song.columns = ['after_'+i for i in cols]
train = train.merge(song, on='after_song_id', how='left')
test = test.merge(song, on='after_song_id', how='left')

del song
gc.collect()

print('Member/Song data loaded.')

#####################################################
## Additional Features
#####################################################

## contextual features
train['before_type_same'] = (train['before_source_type'] == train['source_type']) * 1.0
test['before_type_same'] = (test['before_source_type'] == test['source_type']) * 1.0

train['after_type_same'] = (train['after_source_type'] == train['source_type']) * 1.0
test['after_type_same'] = (test['after_source_type'] == test['source_type']) * 1.0

train['before_artist_same'] = (train['before_artist_name'] == train['artist_name']) * 1.0
test['before_artist_same'] = (test['before_artist_name'] == test['artist_name']) * 1.0

train['after_artist_same'] = (train['after_artist_name'] == train['artist_name']) * 1.0
test['after_artist_same'] = (test['after_artist_name'] == test['artist_name']) * 1.0
'''
train['timestamp_mean_diff'] = train['timestamp'] - train['msno_timestamp_mean']
test['timestamp_mean_diff'] = test['timestamp'] - test['msno_timestamp_mean']

train['timestamp_mean_diff_rate'] = train['timestamp_mean_diff'] / train['msno_timestamp_std']
test['timestamp_mean_diff_rate'] = test['timestamp_mean_diff'] / test['msno_timestamp_std']
'''
train['time_spent'] = train['timestamp'] - train['registration_init_time']
test['time_spent'] = test['timestamp'] - test['registration_init_time']

train['time_left'] = train['expiration_date'] - train['timestamp']
test['time_left'] = test['expiration_date'] - test['timestamp']
'''
train['msno_till_now_cnt_rate'] = train['msno_till_now_cnt'] - train['msno_rec_cnt']
test['msno_till_now_cnt_rate'] = test['msno_till_now_cnt'] - test['msno_rec_cnt']

train['msno_left_cnt'] = np.log1p(np.exp(train['msno_rec_cnt']) - \
        np.exp(train['msno_till_now_cnt']))
test['msno_left_cnt'] = np.log1p(np.exp(test['msno_rec_cnt']) - \
        np.exp(test['msno_till_now_cnt']))
'''
## user-side features
train['duration'] = train['expiration_date'] - train['registration_init_time']
test['duration'] = test['expiration_date'] - test['registration_init_time']

train['msno_upper_time'] = train['msno_timestamp_mean'] + train['msno_timestamp_std']
test['msno_upper_time'] = test['msno_timestamp_mean'] + test['msno_timestamp_std']

train['msno_lower_time'] = train['msno_timestamp_mean'] - train['msno_timestamp_std']
test['msno_lower_time'] = test['msno_timestamp_mean'] - test['msno_timestamp_std']

## song-side features
train['song_upper_time'] = train['song_timestamp_mean'] + train['song_timestamp_std']
test['song_upper_time'] = test['song_timestamp_mean'] + test['song_timestamp_std']

train['song_lower_time'] = train['song_timestamp_mean'] - train['song_timestamp_std']
test['song_lower_time'] = test['song_timestamp_mean'] - test['song_timestamp_std']

#####################################################
## Feature Processing
#####################################################

## set features to category
embedding_features = ['msno', 'city', 'gender', 'registered_via', \
        'song_id', 'artist_name', 'composer', 'lyricist', 'language', \
        'first_genre_id', 'second_genre_id', 'third_genre_id', 'cc', 'xxx', \
        'isrc_missing', 'source_system_tab', 'source_screen_name', 'source_type']
song_id_feat = ['artist_name', 'composer', 'lyricist', 'language', \
        'first_genre_id', 'second_genre_id', 'third_genre_id', 'cc', 'xxx', \
        'isrc_missing']
embedding_features += ['before_'+i for i in song_id_feat]
embedding_features += ['after_'+i for i in song_id_feat]
embedding_features += ['before_song_id', 'after_song_id', 'before_source_type', \
        'after_source_type', 'before_type_same', 'after_type_same', \
        'before_artist_same', 'after_artist_same']

for feat in embedding_features:
    train[feat] = train[feat].astype('category')
    test[feat] = test[feat].astype('category')

## feature selection
feat_importance = pd.read_csv('./lgb_feature_importance.csv')
feature_name = feat_importance['name'].values
feature_importance = feat_importance['importance'].values

drop_col = feature_name[feature_importance<85]
def transfer(x):
    if x == 'msno_source_screen_name_15':
        return 'msno_source_screen_name_17'
    elif  x == 'msno_source_screen_name_16':
        return 'msno_source_screen_name_18'
    elif  x == 'msno_source_screen_name_17':
        return 'msno_source_screen_name_19'
    elif  x == 'msno_source_screen_name_18':
        return 'msno_source_screen_name_20'
    elif  x == 'msno_source_screen_name_19':
        return 'msno_source_screen_name_21'
    elif  x == 'msno_source_screen_name_20':
        return 'msno_source_screen_name_22'
    else:
        return x
drop_col = [transfer(i) for i in drop_col]

train.drop(drop_col, axis=1, inplace=True)
test.drop(drop_col, axis=1, inplace=True)

## print data information
print('Data preparation done.')
print('Training data shape:')
print(train.shape)
print('Testing data shape:')
print(test.shape)
print('Features invlove:')
print(train.columns)

#####################################################
## Model Training
#####################################################

## model training
train_data = lgb.Dataset(train, label=train_y, max_bin=255, \
        free_raw_data=True)

del train
gc.collect()

para = pd.read_csv('./lgb_record.csv').sort_values(by='val_auc', ascending=False)
for i in range(1):
    params = {
        'boosting_type': para['type'].values[i],
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'], 
        
        'learning_rate': para['lr'].values[i],
        
        'num_leaves': para['n_leaf'].values[i],
        'max_depth': para['n_depth'].values[i],
        'min_data_in_leaf': para['min_data'].values[i],
        
        'feature_fraction': para['feature_frac'].values[i],
        'bagging_fraction': para['bagging_frac'].values[i],
        'bagging_freq': para['bagging_freq'].values[i],
        
        'lambda_l1': para['l1'].values[i],
        'lambda_l2': para['l2'].values[i],
        'min_gain_to_split': para['min_gain'].values[i],
        'min_sum_hessian_in_leaf': para['hessian'].values[i],
        
        'num_threads': 16,
        'verbose': -1,
        'is_training_metric': 'True'
    }
    
    print('Hyper-parameters:')
    print(params)

    num_round = para['bst_rnd'].values[i]
    print('Round number: %d'%num_round)

    gbm = lgb.train(params, train_data, num_round, valid_sets=train_data, verbose_eval=100)

    val_auc = para['val_auc'].values[i]
    print('Model training done. Validation AUC: %.5f'%val_auc)

    feature_importance = pd.DataFrame({'name':gbm.feature_name(), 'importance':gbm.feature_importance()}).sort_values(by='importance', ascending=False)
    feature_importance.to_csv('./feat_importance_for_test.csv', index=False)
    
    flag = np.random.randint(0, 65536)    
       
    test_pred = gbm.predict(test)
    test_sub = pd.DataFrame({'id': test_id, 'target': test_pred})
    test_sub.to_csv('./submission/lgb_%.5f_%d.csv.gz'%(val_auc, flag), index=False, \
            compression='gzip')
    
