import numpy as np
import pandas as pd
from tensorflow import keras
from metrics_tracking import F1Score
import tensorflow as tf

def load_data():
    data = np.load("Preprocessed_Data/roads_canids_windows_200hz_3s.npz")
    # Access arrays by their keys
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]
    X_train = X_train[..., :-1] #TEMP FIX REMOVE LATER - FIX PREPROCESSING TO GET RID OF STRING COLUMN
    X_test  = X_test[..., :-1] #TEMP FIX REMOVE LATER
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    data.close()
    return X_train, y_train, X_test, y_test

def load_model(): 
    model_save_name = "stm32_ROAD_model32_2Output.keras"
    keras_model = keras.models.load_model( #import model for quantization
        # "saved_models/best_model_road_128cnn.keras",
        f"saved_models/{model_save_name}",
        compile=True,
        custom_objects={"F1Score": F1Score},
        safe_mode=False
    )
    print(keras_model.summary())
    return keras_model, model_save_name

def compress_model(keras_model, model_save_name, quantizefp16=False, quantizeint8=False, representative_data_gen=None):
    model_save_name = model_save_name.replace(".keras", "")
    if quantizeint8:
        tflite_save_name = f"saved_models/TFLite/{model_save_name}_int8.tflite"
    elif quantizefp16:
        tflite_save_name = f"saved_models/TFLite/{model_save_name}_fp16.tflite"
    else:
        tflite_save_name = f"saved_models/TFLite/{model_save_name}.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if quantizefp16 and not quantizeint8:
        converter.target_spec.supported_types = [tf.float16]
    if quantizeint8:
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(tflite_save_name, "wb") as f:
        f.write(tflite_model)
    return tflite_model, tflite_save_name

def make_representative_data_gen(X_train):
    def generator():
        for i in range(200):
            sample = X_train[i:i+1].astype(np.float32)
            yield [sample]
    return generator

def main(): 
    X_train, y_train, X_test, y_test = load_data()
    keras_model, model_save_name = load_model()
    tflite_model, tflite_save_name = compress_model(keras_model, model_save_name, quantizefp16=False, quantizeint8=False, representative_data_gen=None)
    rep_gen = make_representative_data_gen(X_train)
    tflite_path = compress_model(
        keras_model,
        model_save_name=model_save_name,
        quantizefp16=False,
        quantizeint8=True,
        representative_data_gen=rep_gen
    )
    fp32_model, fp32_path = compress_model(
        keras_model, model_save_name,
        quantizefp16=False, quantizeint8=False
    )
    # int8_model , int8_path = compress_model(keras_model, model_save_name,quantizefp16=False, quantizeint8=True,  representative_data_gen=rep_gen)
    return fp32_model

if __name__ == '__main__':
    main()