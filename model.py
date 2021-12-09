import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from input_layer import define_input_layers

def Tensor_Mean_Pooling(name='mean_pooling', keepdims=False):
    return Lambda(lambda x: K.mean(x, axis=1, keepdims=keepdims), name=name)


def fm_1d(inputs, n_mlsfc, n_mcate_nm, n_sex, n_month, n_time, n_day, n_fav_plc):

    '''
    1st embedding layers
    numeric features with shape (None, 1) => dense layer => map to shape (None, 1)
    categorical features (single level) with shape (None,1) => embedding layer (latent_dim = 1) => map to shape (None, 1)
    categorical features (multi level) with shape (None,L) => embedding layer (latent_dim = 1) => map to shape (None, L)
    output will summation of all embeded features, result in a tensor with shape (None, 1)
    '''

    age_input, mlsfc_input, mcate_input, month_input, sex_input, time_input, day_input, fav_plc_input= inputs

    # all tensors are reshape to (None, 1)
    num_dense_1d = [Dense(1, name='num_dense_1d_age')(age_input)]
    cat_sl_embed_1d = [Embedding(n_mlsfc + 1, 1, name='cat_embed_1d_mlsfc')(mlsfc_input),
                       Embedding(n_mcate_nm + 1, 1, name='cat_embed_1d_mcate')(mcate_input),
                       Embedding(n_month + 1, 1, name='cat_embed_1d_month')(month_input),
                       Embedding(n_sex + 1, 1, name='cat_embed_1d_sex')(sex_input),
                       Embedding(n_time + 1, 1, name='cat_embed_1d_time')(time_input),
                       Embedding(n_day + 1, 1, name='cat_embed_1d_day')(day_input),
                       Embedding(n_fav_plc + 1, 1, name='cat_embed_1d_fav_plc')(fav_plc_input)]

    cat_sl_embed_1d = [Reshape((1,))(i) for i in cat_sl_embed_1d]
    # cat_ml_embed_1d = [Tensor_Mean_Pooling(name='embed_1d_mean')(i) for i in cat_ml_embed_1d]

    # add all tensors
    y_fm_1d = Add(name='fm_1d_output')(num_dense_1d + cat_sl_embed_1d)

    return y_fm_1d


def fm_2d(inputs, n_mlsfc, n_mcate_nm, n_sex, n_month, n_time, n_day, n_fav_plc, k):
    '''
    2nd shared embeded layers
    numeric features => dense layer => map to shape (None, 1, k)
    categorical features (single level) => embedding layer (latent_dim = k) => map to shape (None, 1, k)
    categorical features (multi level) with shape (None,L) => embedding layer (latent_dim = k) => map to shape (None, L, k)
    shared embed layer will be the concatenated layers of all embeded features
    shared embed layer => dot layer => 2nd order of fm part
    '''

    age_input, mlsfc_input, mcate_input, month_input, sex_input, time_input, day_input, fav_plc_input= inputs

    num_dense_2d = [Dense(k, name='num_dense_2d_age')(age_input)]  # shape (None, k)
    num_dense_2d = [Reshape((1, k))(i) for i in num_dense_2d]  # shape (None, 1, k)

    cat_sl_embed_2d = [Embedding(n_mlsfc + 1, k, name='cat_embed_2d_mlsfc')(mlsfc_input),
                       Embedding(n_mcate_nm + 1, k, name='cat_embed_2d_mcate')(mcate_input),
                       Embedding(n_month + 1, k, name='cat_embed_2d_month')(month_input),
                       Embedding(n_sex + 1, k, name='cat_embed_2d_sex')(sex_input),
                       Embedding(n_time + 1, k, name='cat_embed_2d_time')(time_input),
                       Embedding(n_day + 1, k, name='cat_embed_2d_day')(day_input),
                       Embedding(n_fav_plc + 1, k, name='cat_embed_2d_fav_plc')(fav_plc_input)] # shape (None, 1, k)

    # concatenate all 2d embed layers => (None, ?, k)
    embed_2d = Concatenate(axis=1, name='concat_embed_2d')(num_dense_2d + cat_sl_embed_2d)

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
    '''
    DNN model on shared embed layers
    shared embed layer => series of dense layers => deep part
    '''
    # flat embed layers from 3D to 2D tensors
    y_dnn = Flatten(name='flat_embed_2d')(embed_2d)
    for h in dnn_dim:
        y_dnn = Dropout(dnn_dr)(y_dnn)
        y_dnn = Dense(h, activation='relu')(y_dnn)
    y_dnn = Dense(1, activation='relu', name='deep_output')(y_dnn)
    # y_dnn = Dense(1, activation='sigmoid', name='deep_output')(y_dnn)

    return y_dnn

def deep_fm_model(n_mlsfc, n_mcate_nm, n_sex, n_month, n_time, n_day, n_fav_plc, k, dnn_dim, dnn_dr):
    inputs = define_input_layers()

    y_fm_1d = fm_1d(inputs, n_mlsfc, n_mcate_nm, n_sex, n_month, n_time, n_day, n_fav_plc)
    y_fm_2d, embed_2d = fm_2d(inputs, n_mlsfc, n_mcate_nm, n_sex, n_month, n_time, n_day, n_fav_plc, k)
    y_dnn = deep_part(embed_2d, dnn_dim, dnn_dr)

    # combinded deep and fm parts
    y = Concatenate()([y_fm_1d, y_fm_2d, y_dnn])
    # y = Dense(1, name='deepfm_output')(y)
    y = Dense(1, name='deepfm_output', activation='sigmoid')(y) # sigmoid 추가

    fm_model_1d = Model(inputs, y_fm_1d)
    fm_model_2d = Model(inputs, y_fm_2d)
    deep_model = Model(inputs, y_dnn)
    deep_fm_model = Model(inputs, y)

    return fm_model_1d, fm_model_2d, deep_model, deep_fm_model