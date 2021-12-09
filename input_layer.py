import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


def define_input_layers():

    age_fea = Input((1,), name='age_fea')
    num_inputs = [age_fea]

    mlsfc_fea = Input((1,), name='mlsfc_fea')
    mcate_fea = Input((1,), name='mcate_fea')
    month_fea = Input((1,), name='month_fea')
    sex_fea = Input((1,), name='sex_fea')
    time_fea = Input((1,), name='time_fea')
    day_fea = Input((1,), name='day_fea')
    fav_plc_fea = Input((1,), name='fav_plc_fea')
    cat_sl_inputs = [mlsfc_fea, mcate_fea, sex_fea, month_fea, time_fea, day_fea, fav_plc_fea]

    inputs = num_inputs + cat_sl_inputs

    return inputs
