import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

ratings = pd.read_csv('dataset/ratings.dat', sep='::', header=None, engine='python',
                      names=['uid', 'mid', 'rating', 'timestamp'],encoding='ISO-8859-1' )
movies = pd.read_csv('dataset/movies.dat', sep='::', header=None, engine='python',
                     names=['mid', 'movie_name', 'movie_genre'],encoding='ISO-8859-1' )
users = pd.read_csv('dataset/users.dat', sep='::', header=None, engine='python',
                    names=['uid', 'user_fea1', 'user_fea2', 'user_fea3', 'user_fea4'],encoding='ISO-8859-1' )

tokenizer = Tokenizer(lower=True, split='|', filters='', num_words=15)
tokenizer.fit_on_texts(movies.movie_genre.values)
seq = tokenizer.texts_to_sequences(movies.movie_genre.values)
movies['movie_genre'] = pad_sequences(seq, maxlen=3, padding='post').tolist()
ratings = ratings.join(movies.set_index('mid'), on='mid', how='left')
ratings = ratings.join(users.set_index('uid'), on='uid', how='left')

# ----------------------

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


def Tensor_Mean_Pooling(name='mean_pooling', keepdims=False):
    return Lambda(lambda x: K.mean(x, axis=1, keepdims=keepdims), name=name)


def fm_1d(inputs, n_uid, n_mid, n_genre):
    fea3_input, uid_input, mid_input, genre_input = inputs

    # all tensors are reshape to (None, 1)
    num_dense_1d = [Dense(1, name='num_dense_1d_fea4')(fea3_input)]
    cat_sl_embed_1d = [Embedding(n_uid + 1, 1, name='cat_embed_1d_uid')(uid_input),
                       Embedding(n_mid + 1, 1, name='cat_embed_1d_mid')(mid_input)]
    cat_ml_embed_1d = [Embedding(n_genre + 1, 1, mask_zero=True, name='cat_embed_1d_genre')(genre_input)]

    cat_sl_embed_1d = [Reshape((1,))(i) for i in cat_sl_embed_1d]
    cat_ml_embed_1d = [Tensor_Mean_Pooling(name='embed_1d_mean')(i) for i in cat_ml_embed_1d]

    # add all tensors
    y_fm_1d = Add(name='fm_1d_output')(num_dense_1d + cat_sl_embed_1d + cat_ml_embed_1d)

    return y_fm_1d


def fm_2d(inputs, n_uid, n_mid, n_genre, k):
    fea3_input, uid_input, mid_input, genre_input = inputs

    num_dense_2d = [Dense(k, name='num_dense_2d_fea3')(fea3_input)]  # shape (None, k)
    num_dense_2d = [Reshape((1, k))(i) for i in num_dense_2d]  # shape (None, 1, k)

    cat_sl_embed_2d = [Embedding(n_uid + 1, k, name='cat_embed_2d_uid')(uid_input),
                       Embedding(n_mid + 1, k, name='cat_embed_2d_mid')(mid_input)]  # shape (None, 1, k)

    cat_ml_embed_2d = [Embedding(n_genre + 1, k, name='cat_embed_2d_genre')(genre_input)]  # shape (None, 3, k)
    cat_ml_embed_2d = [Tensor_Mean_Pooling(name='cat_embed_2d_genure_mean', keepdims=True)(i) for i in
                       cat_ml_embed_2d]  # shape (None, 1, k)

    # concatenate all 2d embed layers => (None, ?, k)
    embed_2d = Concatenate(axis=1, name='concat_embed_2d')(num_dense_2d + cat_sl_embed_2d + cat_ml_embed_2d)

    # calcuate the interactions by simplication
    # sum of (x1*x2) = sum of (0.5*[(xi)^2 - (xi^2)])
    tensor_sum = Lambda(lambda x: K.sum(x, axis=1), name='sum_of_tensors')
    tensor_square = Lambda(lambda x: K.square(x), name='square_of_tensors')

    sum_of_embed = tensor_sum(embed_2d)
    square_of_embed = tensor_square(embed_2d)

    square_of_sum = Multiply()([sum_of_embed, sum_of_embed])
    sum_of_square = tensor_sum(square_of_embed)

    sub = Subtract()([square_of_sum, sum_of_square])
    sub = Lambda(lambda x: x * 0.5)(sub)
    y_fm_2d = Reshape((1,), name='fm_2d_output')(tensor_sum(sub))

    return y_fm_2d, embed_2d


def deep_part(embed_2d, dnn_dim, dnn_dr):
    # flat embed layers from 3D to 2D tensors
    y_dnn = Flatten(name='flat_embed_2d')(embed_2d)
    for h in dnn_dim:
        y_dnn = Dropout(dnn_dr)(y_dnn)
        y_dnn = Dense(h, activation='relu')(y_dnn)
    y_dnn = Dense(1, activation='relu', name='deep_output')(y_dnn)

    return y_dnn


# Model Parameters
n_uid = ratings.uid.max()
n_mid = ratings.mid.max()
n_genre = 14
k = 20
dnn_dim = [64, 64]
dnn_dr = 0.5

# numerica features
fea3_input = Input((1,), name='input_fea3')
num_inputs = [fea3_input]
# single level categorical features
uid_input = Input((1,), name='input_uid')
mid_input = Input((1,), name='input_mid')
cat_sl_inputs = [uid_input, mid_input]

# multi level categorical features (with 3 genres at most)
genre_input = Input((3,), name='input_genre')
cat_ml_inputs = [genre_input]

inputs = num_inputs + cat_sl_inputs + cat_ml_inputs

# Define subnets
y_fm_1d = fm_1d(inputs, n_uid, n_mid, n_genre)
y_fm_2d, embed_2d = fm_2d(inputs, n_uid, n_mid, n_genre, k)
y_dnn = deep_part(embed_2d, dnn_dim, dnn_dr)

# combinded deep and fm parts
y = Concatenate()([y_fm_1d, y_fm_2d, y_dnn])
y = Dense(1, name='deepfm_output')(y)

fm_model_1d = Model(inputs, y_fm_1d)
fm_model_2d = Model(inputs, y_fm_2d)
deep_model = Model(inputs, y_dnn)
deep_fm_model = Model(inputs, y)


# -------------------------------------------------------


# Format Dataset
def df2xy(ratings):
    x = [ratings.user_fea3.values,
         ratings.uid.values,
         ratings.mid.values,
         np.concatenate(ratings.movie_genre.values).reshape(-1, 3)]
    y = ratings.rating.values
    return x, y


in_train_flag = np.random.random(len(ratings)) <= 0.9
train_data = ratings.loc[in_train_flag,]
valid_data = ratings.loc[~in_train_flag,]
train_x, train_y = df2xy(train_data)
valid_x, valid_y = df2xy(valid_data)

# train  model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

deep_fm_model.compile(loss='MSE', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=3)
model_ckp = ModelCheckpoint(filepath='deepfm_weights_v2.h5',
                            monitor='val_loss',
                            save_weights_only=False,
                            save_best_only=True)
callbacks = [model_ckp, early_stop]
train_history = deep_fm_model.fit(train_x, train_y,
                                  epochs=30, batch_size=2048,
                                  validation_split=0.1,
                                  callbacks=callbacks)

# import tensorflow.keras as keras
# model = keras.models.load_model("deepfm_weights_v2.h5")
# model.summary()