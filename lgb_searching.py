import gc
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

#####################################################
## Data Loading
#####################################################

## load data
train = pd.read_csv('./input/validation/train.csv')
train_y = train['target']
train.drop(['target'], inplace=True, axis=1)

test = pd.read_csv('./input/validation/test.csv')
test_y = pd.read_csv('./input/validation/test_label.csv')['target']
test.drop(['id'], inplace=True, axis=1)

train_add = pd.read_csv('./input/validation/train_add.csv')
test_add = pd.read_csv('./input/validation/test_add.csv')

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
member = pd.read_csv('./input/validation/members_gbdt.csv')

train = train.merge(member, on='msno', how='left')
test = test.merge(member, on='msno', how='left')

del member
gc.collect()

member_add = pd.read_csv('./input/validation/members_add.csv')

cols = ['msno', 'msno_song_length_mean', 'artist_msno_cnt']
train = train.merge(member_add[cols], on='msno', how='left')
test = test.merge(member_add[cols], on='msno', how='left')

del member_add
gc.collect()

song = pd.read_csv('./input/validation/songs_gbdt.csv')

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

## user-song pair features
#train['length_diff'] = train['song_length'] - train['msno_song_length_mean']
#test['length_diff'] = test['song_length'] - test['msno_song_length_mean']

#####################################################
## Additional Features - Experiments
#####################################################
'''
train['context_prob'] = train['msno_source_type_prob'] * train['msno_source_screen_name_prob'] * train['msno_source_system_tab_prob'] 
test['context_prob'] = test['msno_source_type_prob'] * test['msno_source_screen_name_prob'] * test['msno_source_system_tab_prob']

for i in range(8):
    train['mult_component_%d'%i] = train['member_component_%d'%i] * train['song_component_%d'%i]
    test['mult_component_%d'%i] = test['member_component_%d'%i] * test['song_component_%d'%i]

song_add = pd.read_csv('./input/validation/song_add.csv')
train = train.merge(song_add, on='song_id', how='left')
test = test.merge(song_add, on='song_id', how='left')

col = ['song_component_%d'%i for i in range(48)]

col_before = ['before_song_component_%d'%i for i in range(48)]
train['before_song_dot'] = train.apply(lambda x: np.dot(x[col].values, x[col_before].values), axis=1)
test['before_song_dot'] = test.apply(lambda x: np.dot(x[col].values, x[col_before].values), axis=1)

col_after = ['after_song_component_%d'%i for i in range(48)]
train['after_song_dot'] = train.apply(lambda x: np.dot(x[col].values, x[col_after].values), axis=1)
test['after_song_dot'] = test.apply(lambda x: np.dot(x[col].values, x[col_after].values), axis=1)

train_add = pd.read_csv('./input/validation/train_add.csv')
test_add = pd.read_csv('./input/validation/test_add.csv')

for col in train_add.columns:
    train[col] = train_add[col].values
    test[col] = test_add[col].values

del train_add
del test_add
gc.collect()
'''
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

col_to_drop_by_importance = feature_name[feature_importance<85]
train.drop(col_to_drop_by_importance, axis=1, inplace=True)
test.drop(col_to_drop_by_importance, axis=1, inplace=True)

## print data information
print('Data preparation done.')
print('Training data shape:')
print(train.shape)
print('Testing data shape:')
print(test.shape)
print('Features invlove:')
print(train.columns)

feat_cnt = train.shape[1]

#####################################################
## Model Training
#####################################################

## model training
train_data = lgb.Dataset(train, label=train_y, max_bin=255, \
        free_raw_data=True)
test_data = lgb.Dataset(test, label=test_y, reference=train_data, max_bin=255, \
        free_raw_data=True)

del train
del test
gc.collect()

para = pd.read_csv('./lgb_record.csv').sort_values(by='val_auc', ascending=False)
for i in range(1):
    
    params = {
        'boosting_type': para['type'].values[i],
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'], 
        
        'learning_rate': 0.5,  # para['lr'].values[i],
        
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
        'verbose': 0,
        'is_training_metric': 'True'
    }
    '''
    params = {
        'boosting_type': 'gbdt',  # np.random.choice(['dart', 'gbdt']),
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'], 
        
        'learning_rate': 0.35,
        
        'num_leaves': np.random.randint(64, 128),
        'max_depth': np.random.randint(6, 12),
        'min_data_in_leaf': int(2 ** (np.random.rand()*3.5 + 9)),
        
        'feature_fraction': np.random.rand()*0.35+0.65,
        'bagging_fraction': np.random.rand()*0.35+0.65,
        'bagging_freq': 1,
        
        'lambda_l1': 10 ** (np.random.rand() * 4),
        'lambda_l2': 10 ** (np.random.rand() * 3 + 2),
        'min_gain_to_split': 0.0,
        'min_sum_hessian_in_leaf': 0.1,
        
        'num_threads': 16,
        'verbose': 0,
        'is_training_metric': 'True'
    }
    '''
    print('Hyper-parameters:')
    print(params)

    evals_result = {}

    gbm = lgb.train(params, train_data, 5000, valid_sets=[train_data, test_data], \
            valid_names = ['train', 'valid'], evals_result=evals_result, \
            early_stopping_rounds=250, verbose_eval=100)

    bst_round = np.argmax(evals_result['valid']['auc'])
    trn_auc = evals_result['train']['auc'][bst_round]
    trn_loss = evals_result['train']['binary_logloss'][bst_round]
    val_auc = evals_result['valid']['auc'][bst_round]
    val_loss = evals_result['valid']['binary_logloss'][bst_round]

    print('Best Round: %d'%bst_round)
    print('Training loss: %.5f, Validation loss: %.5f'%(trn_loss, val_loss))
    print('Training AUC : %.5f, Validation AUC : %.5f'%(trn_auc, val_auc))
    
    feature_importance = pd.DataFrame({'name':gbm.feature_name(), 'importance':gbm.feature_importance()}).sort_values(by='importance', ascending=False)
    feature_importance.to_csv('./feat_importance.csv', index=False)
    
    res = '%s,%s,%d,%s,%.4f,%d,%d,%d,%.4f,%.4f,%d,%.4e,%.4e,%.4e,%.4e,%.4e,%d,%.5f,%.5f,%.5f,%.5f\n'%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), \
            'LightGBM_baseline_song_context_prob', feat_cnt, params['boosting_type'], params['learning_rate'], params['num_leaves'], params['max_depth'], \
            params['min_data_in_leaf'], params['feature_fraction'], params['bagging_fraction'], \
            params['bagging_freq'], params['lambda_l1'], params['lambda_l2'], params['min_gain_to_split'], \
            params['min_sum_hessian_in_leaf'], 0.0, bst_round+1, trn_loss, trn_auc, val_loss, val_auc)
    f = open('./lgb_record.csv', 'a')
    f.write(res)
    f.close()

