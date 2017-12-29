import os
import gc
import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, Activation, Reshape, Flatten
from keras.layers.merge import concatenate, dot, add, multiply
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop, Adam, SGD

from nn_generator import DataGenerator
'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.475
set_session(tf.Session(config=config))
'''
######################################################
## Data Loading
######################################################

folder = 'training'

## load train data
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

for col in train_add.columns:
    train[col] = train_add[col].values
    test[col] = test_add[col].values

print('Train data loaded.')

## load other data
member = pd.read_csv('./input/%s/members_nn.csv'%folder).sort_values('msno')
song = pd.read_csv('./input/%s/songs_nn.csv'%folder).sort_values('song_id')

member_add = pd.read_csv('./input/%s/members_add.csv'%folder).sort_values('msno')
member_add.fillna(0, inplace=True)
member = member.merge(member_add, on='msno', how='left')

print('Member/Song data loaded.')

######################################################
## Feature Preparation
######################################################

## data preparation
train = train.merge(member[['msno', 'city', 'gender', 'registered_via', \
        'registration_init_time', 'expiration_date']], on='msno', how='left')
test = test.merge(member[['msno', 'city', 'gender', 'registered_via', \
        'registration_init_time', 'expiration_date']], on='msno', how='left')

train = train.merge(song[['song_id', 'artist_name', 'composer', 'lyricist', \
        'language', 'first_genre_id', 'second_genre_id', 'third_genre_id', \
        'cc', 'xxx']], on='song_id', how='left')
test = test.merge(song[['song_id', 'artist_name', 'composer', 'lyricist', \
        'language', 'first_genre_id', 'second_genre_id', 'third_genre_id', \
        'cc', 'xxx']], on='song_id', how='left')

cols = ['song_id', 'artist_name', 'language', 'first_genre_id', 'song_rec_cnt']
tmp = song[cols]

tmp.columns = ['before_'+i for i in cols]
train = train.merge(tmp, on='before_song_id', how='left')
test = test.merge(tmp, on='before_song_id', how='left')

tmp.columns = ['after_'+i for i in cols]
train = train.merge(tmp, on='after_song_id', how='left')
test = test.merge(tmp, on='after_song_id', how='left')

train['before_type_same'] = (train['source_type'] == train['before_source_type']).astype(int)
test['before_type_same'] = (test['source_type'] == test['before_source_type']).astype(int)

train['after_type_same'] = (train['source_type'] == train['after_source_type']).astype(int)
test['after_type_same'] = (test['source_type'] == test['after_source_type']).astype(int)

train['before_artist_same'] = (train['artist_name'] == train['before_artist_name']).astype(int)
test['before_artist_same'] = (test['artist_name'] == test['before_artist_name']).astype(int)

train['after_artist_same'] = (train['artist_name'] == train['after_artist_name']).astype(int)
test['after_artist_same'] = (test['artist_name'] == test['after_artist_name']).astype(int)

train['before_genre_same'] = (train['first_genre_id'] == train['before_first_genre_id']).astype(int)
test['before_genre_same'] = (test['first_genre_id'] == test['before_first_genre_id']).astype(int)

train['after_genre_same'] = (train['first_genre_id'] == train['after_first_genre_id']).astype(int)
test['after_genre_same'] = (test['first_genre_id'] == test['after_first_genre_id']).astype(int)

del tmp
gc.collect()

print('Data preparation done.')

## generate data for training
embedding_features = ['msno', 'city', 'gender', 'registered_via', \
        'artist_name', 'language', 'cc', \
        'source_type', 'source_screen_name', 'source_system_tab', \
        'before_source_type', 'after_source_type', 'before_source_screen_name', \
        'after_source_screen_name', 'before_language', 'after_language', \
        'song_id', 'before_song_id', 'after_song_id']

train_embeddings = []
test_embeddings = []
for feat in embedding_features:
    train_embeddings.append(train[feat].values)
    test_embeddings.append(test[feat].values)

genre_features = ['first_genre_id', 'second_genre_id']

train_genre = []
test_genre = []
for feat in genre_features:
    train_genre.append(train[feat].values)
    test_genre.append(test[feat].values)

context_features = ['after_artist_same', 'after_song_rec_cnt', 'after_timestamp', \
        'after_type_same', 'before_artist_same', 'before_song_rec_cnt', \
        'before_timestamp', 'before_type_same', 'msno_10000_after_cnt', \
        'msno_10000_before_cnt', 'msno_10_after_cnt', 'msno_10_before_cnt', \
        'msno_25_after_cnt', 'msno_25_before_cnt', 'msno_50000_after_cnt', \
        'msno_50000_before_cnt', 'msno_5000_after_cnt', 'msno_5000_before_cnt', \
        'msno_500_after_cnt', 'msno_500_before_cnt', 'msno_source_screen_name_prob', \
        'msno_source_system_tab_prob', 'msno_source_type_prob', 'msno_till_now_cnt', \
        'registration_init_time', 'song_50000_after_cnt', 'song_50000_before_cnt', \
        'song_till_now_cnt', 'timestamp', 'msno_artist_name_prob', 'msno_first_genre_id_prob', \
        'msno_xxx_prob', 'msno_language_prob', 'msno_yy_prob', 'msno_source_prob', \
        'song_source_system_tab_prob', 'song_source_screen_name_prob', 'song_source_type_prob']

train_context = train[context_features].values
test_context = test[context_features].values

ss_context = StandardScaler()
train_context = ss_context.fit_transform(train_context)
test_context = ss_context.transform(test_context)

del train
del test
gc.collect()

usr_features = ['bd', 'expiration_date', 'msno_rec_cnt', 'msno_source_screen_name_0', \
        'msno_source_screen_name_1', 'msno_source_screen_name_10', 'msno_source_screen_name_11', \
        'msno_source_screen_name_12', 'msno_source_screen_name_13', 'msno_source_screen_name_14', \
        'msno_source_screen_name_17', \
        'msno_source_screen_name_18', 'msno_source_screen_name_19', 'msno_source_screen_name_2', \
        'msno_source_screen_name_20', 'msno_source_screen_name_21', 'msno_source_screen_name_3', 'msno_source_screen_name_4', \
        'msno_source_screen_name_5', 'msno_source_screen_name_6', 'msno_source_screen_name_7', \
        'msno_source_screen_name_8', 'msno_source_screen_name_9', 'msno_source_system_tab_0', \
        'msno_source_system_tab_1', 'msno_source_system_tab_2', 'msno_source_system_tab_3', \
        'msno_source_system_tab_4', 'msno_source_system_tab_5', 'msno_source_system_tab_6', \
        'msno_source_system_tab_7', 'msno_source_system_tab_8', \
        'msno_source_type_0', 'msno_source_type_1', 'msno_source_type_10', \
        'msno_source_type_11', 'msno_source_type_2', \
        'msno_source_type_3', 'msno_source_type_4', 'msno_source_type_5', \
        'msno_source_type_6', \
        'msno_source_type_7', 'msno_source_type_8', 'msno_source_type_9', \
        'msno_timestamp_mean', 'msno_timestamp_std', 'registration_init_time', \
        'msno_song_length_mean', 'msno_artist_song_cnt_mean', 'msno_artist_rec_cnt_mean', \
        'msno_song_rec_cnt_mean', 'msno_yy_mean', 'msno_song_length_std', \
        'msno_artist_song_cnt_std', 'msno_artist_rec_cnt_std', 'msno_song_rec_cnt_std', \
        'msno_yy_std', 'artist_msno_cnt']

usr_feat = member[usr_features].values
usr_feat = StandardScaler().fit_transform(usr_feat)

song_features = ['artist_rec_cnt', 'artist_song_cnt', 'composer_song_cnt', \
        'genre_rec_cnt', 'genre_song_cnt', 'song_length', \
        'song_rec_cnt', 'song_timestamp_mean', 'song_timestamp_std', \
        'xxx_rec_cnt', 'xxx_song_cnt', 'yy', 'yy_song_cnt']

song_feat = song[song_features].values
song_feat = StandardScaler().fit_transform(song_feat)

n_factors = 48

usr_component_features = ['member_component_%d'%i for i in range(n_factors)]
song_component_features = ['song_component_%d'%i for i in range(n_factors)]

usr_component = member[usr_component_features].values
song_component = song[song_component_features].values

n_artist = 16

usr_artist_features = ['member_artist_component_%d'%i for i in range(n_artist)]
song_artist_features = ['artist_component_%d'%i for i in range(n_artist)]

usr_artist_component = member[usr_artist_features].values
song_artist_component = song[song_artist_features].values

del member
del song
gc.collect()

dataGenerator = DataGenerator()

train_flow = dataGenerator.flow(train_embeddings+train_genre, [train_context], \
        train_y, batch_size=8192, shuffle=True)

######################################################
## Model Structure
######################################################

## define the model
def FunctionalDense(n, x, batchnorm=False, act='relu', lw1=0.0, dropout=None, name=''):
    if lw1 == 0.0:
        x = Dense(n, name=name+'_dense')(x)
    else:
        x = Dense(n, kernel_regularizer=l1(lw1), name=name+'_dense')(x)
    
    if batchnorm:
        x = BatchNormalization(name=name+'_batchnorm')(x)
        
    if act in {'relu', 'tanh', 'sigmoid'}:
        x = Activation(act, name=name+'_activation')(x)
    elif act =='prelu':
        x = PReLU(name=name+'_activation')(x)
    elif act == 'leakyrelu':
        x = LeakyReLU(name=name+'_activation')(x)
    elif act == 'elu':
        x = ELU(name=name+'_activation')(x)
    
    if dropout > 0:
        x = Dropout(dropout, name=name+'_dropout')(x)
        
    return x

def get_model(K, K0, lw=1e-4, lw1=1e-4, lr=1e-3, act='relu', batchnorm=False):
    embedding_inputs = []
    embedding_outputs = []
    for i in range(len(embedding_features) - 3):
        val_bound = 0.0 if i == 0 else 0.005
        tmp_input = Input(shape=(1,), dtype='int32', name=embedding_features[i]+'_input')
        tmp_embeddings = Embedding(int(train_embeddings[i].max()+1),
                K if i == 0 else K0,
                embeddings_initializer=RandomUniform(minval=-val_bound, maxval=val_bound),
                embeddings_regularizer=l2(lw),
                input_length=1,
                trainable=True,
                name=embedding_features[i]+'_embeddings')(tmp_input)
        tmp_embeddings = Flatten(name=embedding_features[i]+'_flatten')(tmp_embeddings)
        
        embedding_inputs.append(tmp_input)
        embedding_outputs.append(tmp_embeddings)

    song_id_input = Input(shape=(1,), dtype='int32', name='song_id_input')
    before_song_id_input = Input(shape=(1,), dtype='int32', name='before_song_id_input')
    after_song_id_input = Input(shape=(1,), dtype='int32', name='after_song_id_input')

    embedding_inputs += [song_id_input, before_song_id_input, after_song_id_input]

    genre_inputs = []
    genre_outputs = []
    genre_embeddings = Embedding(int(np.max(train_genre)+1),
            K0,
            embeddings_initializer=RandomUniform(minval=-0.05, maxval=0.05),
            embeddings_regularizer=l2(lw),
            input_length=1,
            trainable=True,
            name='genre_embeddings')
    for i in range(len(genre_features)):
        tmp_input = Input(shape=(1,), dtype='int32', name=genre_features[i]+'_input')
        tmp_embeddings = genre_embeddings(tmp_input)
        tmp_embeddings = Flatten(name=genre_features[i]+'_flatten')(tmp_embeddings)
        
        genre_inputs.append(tmp_input)
        genre_outputs.append(tmp_embeddings)

    usr_input = Embedding(usr_feat.shape[0],
            usr_feat.shape[1],
            weights=[usr_feat],
            input_length=1,
            trainable=False,
            name='usr_feat')(embedding_inputs[0])
    usr_input = Flatten(name='usr_feat_flatten')(usr_input)
    
    song_input = Embedding(song_feat.shape[0],
            song_feat.shape[1],
            weights=[song_feat],
            input_length=1,
            trainable=False,
            name='song_feat')(song_id_input)
    song_input = Flatten(name='song_feat_flatten')(song_input)
    
    usr_component_input = Embedding(usr_component.shape[0],
            usr_component.shape[1],
            weights=[usr_component],
            input_length=1,
            trainable=False,
            name='usr_component')(embedding_inputs[0])
    usr_component_input = Flatten(name='usr_component_flatten')(usr_component_input)
    
    song_component_embeddings = Embedding(song_component.shape[0],
            song_component.shape[1],
            weights=[song_component],
            input_length=1,
            trainable=False,
            name='song_component')
    song_component_input = song_component_embeddings(song_id_input)
    song_component_input = Flatten(name='song_component_flatten')(song_component_input)
    before_song_component_input = song_component_embeddings(before_song_id_input)
    before_song_component_input = Flatten(name='before_song_component_flatten')(before_song_component_input)
    after_song_component_input = song_component_embeddings(after_song_id_input)
    after_song_component_input = Flatten(name='after_song_component_flatten')(after_song_component_input)
    
    usr_artist_component_input = Embedding(usr_artist_component.shape[0],
            usr_artist_component.shape[1],
            weights=[usr_artist_component],
            input_length=1,
            trainable=False,
            name='usr_artist_component')(embedding_inputs[0])
    usr_artist_component_input = Flatten(name='usr_artist_component_flatten')(usr_artist_component_input)
    
    song_artist_component_embeddings = Embedding(song_artist_component.shape[0],
            song_artist_component.shape[1],
            weights=[song_artist_component],
            input_length=1,
            trainable=False,
            name='song_artist_component')
    song_artist_component_input = song_artist_component_embeddings(song_id_input)
    song_artist_component_input = Flatten(name='song_artist_component_flatten')(song_artist_component_input)
    before_song_artist_component_input = song_artist_component_embeddings(before_song_id_input)
    before_song_artist_component_input = Flatten(name='before_song_artist_component_flatten')(before_song_artist_component_input)
    after_song_artist_component_input = song_artist_component_embeddings(after_song_id_input)
    after_song_artist_component_input = Flatten(name='after_song_artist_component_flatten')(after_song_artist_component_input)
    
    context_input = Input(shape=(len(context_features),), name='context_feat')
    
    # basic profiles
    usr_profile = concatenate(embedding_outputs[1:4]+[usr_input, \
            usr_component_input, usr_artist_component_input], name='usr_profile')
    song_profile = concatenate(embedding_outputs[4:7]+genre_outputs+[song_input, \
            song_component_input, song_artist_component_input], name='song_profile')

    multiply_component = dot([usr_component_input, song_component_input], axes=1, name='component_dot')
    multiply_artist_component = dot([usr_artist_component_input, \
            song_artist_component_input], axes=1, name='artist_component_dot')
    multiply_before_song = dot([before_song_component_input, song_component_input], \
            normalize=True, axes=1, name='before_component_dot')
    multiply_after_song = dot([after_song_component_input, song_component_input], \
            normalize=True, axes=1, name='after_component_dot')
    multiply_before_artist = dot([before_song_artist_component_input, song_artist_component_input], \
            normalize=True, axes=1, name='before_artist_component_dot')
    multiply_after_artist = dot([after_song_artist_component_input, song_artist_component_input], \
            normalize=True, axes=1, name='after_artist_component_dot')
    context_profile = concatenate(embedding_outputs[7:]+[context_input, \
            multiply_component, multiply_artist_component, before_song_component_input, \
            after_song_component_input, before_song_artist_component_input, \
            after_song_artist_component_input, multiply_before_song, multiply_after_song, \
            multiply_before_artist, multiply_after_artist], name='context_profile')
    
    # user field
    usr_embeddings = FunctionalDense(K*2, usr_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='usr_profile')
    usr_embeddings = Dense(K, name='usr_profile_output')(usr_embeddings)
    usr_embeddings = add([usr_embeddings, embedding_outputs[0]], name='usr_embeddings')
    
    # song field
    song_embeddings = FunctionalDense(K*2, song_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='song_profile')
    song_embeddings = Dense(K, name='song_profile_output')(song_embeddings)
    #song_embeddings = add([song_embeddings, embedding_outputs[4]], name='song_embeddings')
    
    # context field
    context_embeddings = FunctionalDense(K, context_profile, lw1=lw1, batchnorm=batchnorm, act=act, name='context_profile')

    # joint embeddings
    joint = dot([usr_embeddings, song_embeddings], axes=1, normalize=False, name='pred_cross')
    joint_embeddings = concatenate([usr_embeddings, song_embeddings, context_embeddings, joint], name='joint_embeddings')
    
    # top model
    preds0 = FunctionalDense(K*2, joint_embeddings, batchnorm=batchnorm, act=act, name='preds_0')
    preds1 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0]), batchnorm=batchnorm, act=act, name='preds_1')
    preds2 = FunctionalDense(K*2, concatenate([joint_embeddings, preds0, preds1]), batchnorm=batchnorm, act=act, name='preds_2')
    
    preds = concatenate([joint_embeddings, preds0, preds1, preds2], name='prediction_aggr')
    preds = Dropout(0.5, name='prediction_dropout')(preds)
    preds = Dense(1, activation='sigmoid', name='prediction')(preds)
        
    model = Model(inputs=embedding_inputs+genre_inputs+[context_input], outputs=preds)
    
    opt = RMSprop(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    return model

######################################################
## Model Training
######################################################

## train the model
para = pd.read_csv('./nn_record.csv').sort_values(by='val_auc', ascending=False)
for i in range(5):

    K = para['K'].values[i]
    K0 = para['K0'].values[i]
    lw = para['lw'].values[i]
    lw1 = para['lw1'].values[i]
    lr = para['lr'].values[i]
    lr_decay = para['lr_decay'].values[i]
    activation = para['activation'].values[i]
    batchnorm = para['batchnorm'].values[i]
    bst_epoch = para['bst_epoch'].values[i]
    train_loss0 = para['trn_loss'].values[i]
    val_auc = para['val_auc'].values[i]

    while(True):
        print('K: %d, K0: %d, lw: %e, lw1: %e, lr: %e, lr_decay: %f, act: %s, batchnorm: %s'%(K, K0, lw, \
                lw1, lr, lr_decay, activation, batchnorm))
        
        model = get_model(K, K0, lw, lw1, lr, activation, batchnorm)
        
        #model.summary()
        #from keras.utils import plot_model
        #plot_model(model, to_file='model.png')

        lr_reducer = LearningRateScheduler(lambda x: lr*(lr_decay**x))
        
        hist = model.fit_generator(train_flow, train_flow.__len__(), \
                 epochs=bst_epoch, workers=4, callbacks=[lr_reducer])
        
        train_loss = hist.history['loss'][-1]
        
        if(train_loss < train_loss0 * 1.1):
            break
        
    val_auc = para['val_auc'].values[i]
    print('Model training done. Validation AUC: %.5f'%val_auc)

    flag = np.random.randint(0, 65536)
    
    test_flow = dataGenerator.flow(test_embeddings + test_genre, [test_context], \
            batch_size=16384, shuffle=False)
    test_pred = model.predict_generator(test_flow, test_flow.__len__(), workers=1)
    
    test_sub = pd.DataFrame({'id': test_id, 'target': test_pred.ravel()})
    test_sub.to_csv('./temp_nn/nn_%.5f_%.5f_%d.csv'%(val_auc, train_loss, flag), index=False)
    
