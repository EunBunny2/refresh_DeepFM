import config
from preprocess import get_modified_data
from model import DeepFM
from layers import FM_layer

import numpy as np
import pandas as pd
from time import perf_counter
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, AUC

import json


def get_data():
    file = pd.read_csv('input.csv', header=None, encoding='utf-8-sig')
    X = file.loc[1:, 0:7]
    # Y = file.iloc[1:, [7]]
    Y = file.iloc[1:, 8]
    Y = pd.to_numeric(Y)

    X.columns = config.ALL_FIELDS
    field_dict, field_index, X_modified = \
        get_modified_data(X, config.ALL_FIELDS, config.CONT_FIELDS, config.CAT_FIELDS, is_bin=False)

    X_train, X_test, Y_train, Y_test = train_test_split(X_modified, Y, test_size=0.2, stratify=Y, shuffle=True)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train.values, tf.float32), tf.cast(Y_train, tf.float32))).shuffle(30000).batch(config.BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_test.values, tf.float32), tf.cast(Y_test, tf.float32))).shuffle(10000).batch(config.BATCH_SIZE)

    return train_ds, test_ds, field_dict, field_index


# Batch 단위 학습
def train_on_batch(model, optimizer, acc, auc, inputs, targets):
    with tf.GradientTape() as tape:
        y_pred = model(inputs)
        loss = tf.keras.losses.binary_crossentropy(from_logits=False, y_true=targets, y_pred=y_pred)

    grads = tape.gradient(target=loss, sources=model.trainable_variables)

    # apply_gradients()를 통해 processed gradients를 적용함
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # accuracy & auc
    acc.update_state(targets, y_pred)
    auc.update_state(targets, y_pred)

    return loss


# 반복 학습 함수
def train(epochs, ckpt=None):
    train_ds, test_ds, field_dict, field_index = get_data()

    if ckpt is None:
        model = DeepFM(embedding_size=config.EMBEDDING_SIZE, num_feature=len(field_index), num_field=len(field_dict), field_index=field_index)
    else:
        model = tf.keras.models.load_model(ckpt)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    print("Start Training: Batch Size: {}, Embedding Size: {}".format(config.BATCH_SIZE, config.EMBEDDING_SIZE))
    start = perf_counter()
    for i in range(epochs):
        acc = BinaryAccuracy(threshold=0.5)
        auc = AUC()
        loss_history = []

        for x, y in train_ds:
            loss = train_on_batch(model, optimizer, acc, auc, x, y)
            loss_history.append(loss)

        print("Epoch {:03d}: 누적 Loss: {:.4f}, Acc: {:.4f}, AUC: {:.4f}".format(
            i, np.mean(loss_history), acc.result().numpy(), auc.result().numpy()))

    test_acc = BinaryAccuracy(threshold=0.5)
    test_auc = AUC()
    for x, y in test_ds:
        y_pred = model(x)
        test_acc.update_state(y, y_pred)
        test_auc.update_state(y, y_pred)

    print("테스트 ACC: {:.4f}, AUC: {:.4f}".format(test_acc.result().numpy(), test_auc.result().numpy()))
    print("Batch Size: {}, Embedding Size: {}".format(config.BATCH_SIZE, config.EMBEDDING_SIZE))
    print("걸린 시간: {:.3f}".format(perf_counter() - start))
    # model.save_weights('weights/weights-epoch({})-batch({})-embedding({}).h5'.format(epochs, config.BATCH_SIZE, config.EMBEDDING_SIZE))
    model.save_weights('save_weights_test.h5'.format(epochs, config.BATCH_SIZE, config.EMBEDDING_SIZE))

    # model.save('pb_test.pb')
    json_config = model.to_json()

    with open("non_layer_getConfig_json_test.json", "w") as f:
        json.dump(json_config, f)

    conf = model.get_config()
    print(conf)

def test(ckpt, json_path):
    train_ds, test_ds, field_dict, field_index = get_data()
    # model = tf.keras.models.load_model(ckpt) # save로 저장한 모델 돌릴 때

    # json
    with open(json_path, "r") as st_json:
        model_json = json.load(st_json)

    model = tf.keras.models.model_from_json(model_json, custom_objects={'DeepFM': DeepFM})

    model.load_weights(ckpt)
    model.summary()



    for x, y in test_ds:
        y_pred = model(x)
        print(y_pred)

if __name__ == '__main__':
    # train(epochs=1)
    ckpt_path = "C:\\Users\\ChoEunBin\\PycharmProjects\\DeepFM\\save_weights_test.h5"
    json_path = "C:\\Users\\ChoEunBin\\PycharmProjects\\DeepFM\\non_layer_getConfig_json_test.json"
    test(ckpt_path, json_path)