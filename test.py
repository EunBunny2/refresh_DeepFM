import tensorflow as tf
from tensorflow.keras.models import load_model
from temp_fav import load_input, df2xy
from sklearn.preprocessing import LabelEncoder
import numpy as np

new_model = load_model('deepfm_weights_fav_with_save.h5')

input_data = load_input()

str_list = ['mlsfc', 'mcate_nm', 'Sex', 'day', 'fav_plc']
le = LabelEncoder()
for col in str_list:
    input_data[col] = le.fit_transform(input_data[col])

in_train_flag = np.random.random(len(input_data))

in_train_flag = np.random.random(len(input_data)) <= 0.9
train_data = input_data.loc[in_train_flag,]
valid_data = input_data.loc[~in_train_flag,]
train_x, train_y = df2xy(train_data)
valid_x, valid_y = df2xy(valid_data)

# for idx, value in enumerate(train_x):
#     np.save(f"{idx}_npy.npy", value)
#     print(value)
# 
# np.save("output_npy.npy", train_y)
# print(train_y)

# new_model.fit(train_x, train_y)

print(new_model.predict(valid_x))

# 모델 구조 확인
new_model.summary()