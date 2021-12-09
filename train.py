
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from preprocess_data import load_input
from model import deep_fm_model
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import os
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split

def df2xy(in_df):
    x = [in_df.Age.values,
         in_df.mlsfc.values,
         in_df.mcate_nm.values,
         in_df.Month.values,
         in_df.Sex.values,
         in_df.time.values,
         in_df.day.values,
         in_df.fav_plc.values]
    y = in_df.click.values
    return x, y

def make_vocab(train, col):
    vocab = {}
    vocab_path = "vocab/"+col+"_vocab.txt"
    if os.path.isfile(vocab_path):
        file = open(vocab_path, 'r', encoding='utf-8')
        for line in file.readlines():
            line = line.rstrip()
            key, value = line.split('\t')
            vocab[key] = value
        file.close()
    else:
        count_dict = defaultdict(int)
        classes = train[col]
        for class_ in classes:
            count_dict[class_] += 1

        file = open(vocab_path, 'w', encoding='utf-8')
        # file.write('[UNK]\t0\n')
        # vocab = {'[UNK]': 0}

        for index, (class_, count) in enumerate(sorted(count_dict.items(), reverse=True, key=lambda item: item[1])):
            vocab[class_] = index
            file.write(class_ + '\t' + str(index) + '\n')
        file.close()

    return vocab

def read_data(input_data, col_name, str_list):   # 문자를 vocab에 저장된 숫자로 변환하는 메소드
    for i, train_col in enumerate(col_name):
        if train_col in str_list:
            if train_col != 'fav_plc':
                vocab = make_vocab(input_data, train_col)
            else:
                vocab = make_vocab(input_data, 'mcate_nm')
            for j, class_ in enumerate(input_data[train_col]):
                input_data[train_col][j] = int(vocab[class_])
    # x_input = np.ones(shape=(len(train), len(col_name)))
    # for i, train_col in enumerate(col_name):
    #     if train_col in str_list:
    #         if train_col != 'fav_plc':
    #             vocab = make_vocab(train, train_col)
    #         else:
    #             vocab = make_vocab(train, 'mlsfc')
    #         for j, class_ in enumerate(train[train_col]):
    #             if class_ not in vocab.keys():  # test 데이터에는 unknown 토큰이 존재
    #                 x_input[j][i] = 0
    #             else:
    #                 x_input[j][i] = vocab[class_]
    #     else:
    #         x_input[:,i] = train[train_col]
    #
    # y_input = x_input[:, -1]
    # x_input = x_input[:, :-1]

    return input_data

# def get_params(x_input, col_name, categorical_list):
#     params = {}
#     for i, col in enumerate(col_name):
#         if col in categorical_list:
#             params[f"n_{col.lower()}"] = len(np.unique(x_input[:, i]))
#
#     params['k'] = 30
#     params['dnn_dim'] = [64, 64]
#     params['dnn_dr'] = 0.5
#
#     return params


if __name__ == '__main__':
    input_data, col_name = load_input()


    str_list = ['mlsfc', 'mcate_nm', 'Sex', 'day', 'fav_plc']
    # categorical_list = ['mlsfc', 'mcate_nm', 'Sex', 'Month', 'time', 'day', 'fav_plc']

    input_data = read_data(input_data, col_name, str_list)

    in_train_flag = np.random.random(len(input_data))

    in_train_flag = np.random.random(len(input_data)) <= 0.9
    train_data = input_data.loc[in_train_flag,]
    valid_data = input_data.loc[~in_train_flag,]
    train_x, train_y = df2xy(train_data)
    valid_x, valid_y = df2xy(valid_data)

    n_mlsfc = len(input_data['mlsfc'].unique())
    n_mcate_nm = len(input_data['mcate_nm'].unique())
    n_fav_plc = len(input_data['fav_plc'].unique())
    n_sex = len(input_data['Sex'].unique())
    n_month = len(input_data['Month'].unique())
    n_time = len(input_data['time'].unique())
    n_day = len(input_data['day'].unique())

    params = {
        'n_mlsfc': n_mlsfc,
        'n_mcate_nm': n_mcate_nm,
        'n_sex': n_sex,
        'n_month': n_month,
        'n_time': n_time,
        'n_day': n_day,
        'n_fav_plc': n_fav_plc,
        'k': 30,
        'dnn_dim': [64, 64],
        'dnn_dr': 0.5
    }

    fm_model_1d, fm_model_2d, deep_model, deep_fm_model = deep_fm_model(**params)

    # train  model
    # deep_fm_model.compile(loss='mse', optimizer='adam')
    # loss, opt 수정
    deep_fm_model.compile(loss='binary_crossentropy', optimizer='sgd')
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    model_ckp = ModelCheckpoint(filepath='deepfm_weights_fav.h5',
                                monitor='val_loss',
                                save_weights_only=False,
                                save_best_only=True)
    # callbacks = [model_ckp,early_stop]
    callbacks = [model_ckp]
    train_history = deep_fm_model.fit(train_x, train_y,
                                      epochs=200, batch_size=2048,
                                      validation_split=0.1,
                                      callbacks=callbacks)

    pred_y = deep_fm_model.predict(valid_x)
    print(pred_y)


    # le = LabelEncoder()
    # for col in str_list:
    #     input_data[col] = le.fit_transform(input_data[col])
    #     print(le.classes_)
    #     print(le.transform(le.classes_))


    # in_train_flag = np.random.random(len(input_data))
    #
    # in_train_flag = np.random.random(len(input_data)) <= 0.9
    # train_data = input_data.loc[in_train_flag,]
    # valid_data = input_data.loc[~in_train_flag,]
    # train_x, train_y = df2xy(train_data)
    # valid_x, valid_y = df2xy(valid_data)
    #
    # n_mlsfc = len(input_data['mlsfc'].unique())
    # n_mcate_nm = len(input_data['mcate_nm'].unique())
    # n_fav_plc = len(input_data['fav_plc'].unique())
    # n_sex = len(input_data['Sex'].unique())
    # n_month = len(input_data['Month'].unique())
    # n_time = len(input_data['time'].unique())
    # n_day = len(input_data['day'].unique())
    #
    # params = {
    #     'n_mlsfc': n_mlsfc,
    #     'n_mcate_nm': n_mcate_nm,
    #     'n_sex': n_sex,
    #     'n_month': n_month,
    #     'n_time': n_time,
    #     'n_day' : n_day,
    #     'n_fav_plc': n_fav_plc,
    #     'k':30,
    #     'dnn_dim':[64,64],
    #     'dnn_dr': 0.5
    # }
    #
    # deep_fm_model = deep_fm_model(**params)
    #
    # # train  model
    # # deep_fm_model.compile(loss='mse', optimizer='adam')
    # # loss, opt 수정
    # deep_fm_model.compile(loss='binary_crossentropy', optimizer='sgd')
    # early_stop = EarlyStopping(monitor='val_loss', patience=3)
    # model_ckp = ModelCheckpoint(filepath='deepfm_weights_fav.h5',
    #                             monitor='val_loss',
    #                             save_weights_only=False,
    #                             save_best_only=True)
    # # callbacks = [model_ckp,early_stop]
    # callbacks = [model_ckp]
    # train_history = deep_fm_model.fit(train_x, train_y,
    #                                   epochs=200, batch_size=2048,
    #                                   validation_split=0.1,
    #                                   callbacks=callbacks)
    #
    # pred_y = deep_fm_model.predict(valid_x)
    # print(pred_y)
