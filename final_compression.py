import numpy as np
import pandas as pd
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from keras import layers
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from metrics_tracking import SoftmaxAUC, F1Score, plot_metrics
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

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

def representative_data_gen(X_train):
    for i in range(200):                    
        sample = X_train[i:i+1].astype(np.float32)
        yield [sample]
        
def evaluate_tflite_auc(model_path, x_data, y_labels):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_det = interpreter.get_input_details()
    output_det = interpreter.get_output_details()
    in_dtype = input_det[0]['dtype']
    y_probs = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1].astype(np.float32)
        if in_dtype == np.int8:
            scale, zp = input_det[0]['quantization']
            sample = (sample / scale + zp).round().astype(np.int8)
        interpreter.set_tensor(input_det[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_det[0]['index'])
        if output_det[0]['dtype'] == np.int8:
            s, z = output_det[0]['quantization']
            output = (output.astype(np.float32) - z) * s
        y_probs.append(output[0][0])
    return roc_auc_score(y_test, y_probs)


def main(): 
    keras_model, model_save_name = load_model()
    tflite_model, tflite_save_name = compress_model(keras_model, model_save_name, quantizefp16=False, quantizeint8=False, representative_data_gen=None)
    tflite_path = compress_model(
        keras_model,
        model_save_name=model_save_name,
        quantizefp16=False,
        quantizeint8=True,
        representative_data_gen=representative_data_gen
    )

if __name__ == '__main__':
    main()