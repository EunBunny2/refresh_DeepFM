import tensorflow as tf

# model = tf.keras.models.load('test', compile=False)
#
# export_path = './pb'
# model.save(export_path, save_format="tf")
#
# saved_model_dir = './pb'
converter = tf.lite.TFLiteConverter.from_saved_model("test")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('./tf/converted_model.tflite', 'wb').write(tflite_model)