import numpy as np
import pandas as pd
import serial
import serial
import tensorflow as tf
import os
import random
# import tensorflow_model_optimization as tfmot
from metrics_tracking import F1Score, plot_metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import os
import struct
from keras.models import load_model
import struct

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

# SCALE = 1000.0  # float → int16 multiplier
def create_balanced_csv_samples(X_test, y_test, per_class=1):
    attack_idx = np.where(y_test == 1)[0]
    normal_idx = np.where(y_test == 0)[0]
    chosen = random.sample(list(attack_idx), per_class) + \
             random.sample(list(normal_idx), per_class)
    random.shuffle(chosen)
    X_sel = X_test[chosen]
    y_sel = y_test[chosen]
    return X_sel, y_sel

def write_samples_files_fp32(X_sel, y_sel, scaler_mean, scaler_std, prefix="samples_fp32"):
    os.makedirs("Preprocessed_Data/STM_SAMPLES", exist_ok=True)
    h_file = f"Preprocessed_Data/STM_SAMPLES/{prefix}.h"
    c_file = f"Preprocessed_Data/STM_SAMPLES/{prefix}.c"
    S, T, F = X_sel.shape
    # --------- HEADER FILE ---------
    with open(h_file, "w") as h:
        h.write("#pragma once\n")
        h.write("#include <stdint.h>\n\n")
        h.write(f"#define SAMPLE_COUNT {S}\n")
        h.write(f"#define SAMPLE_TIME   {T}\n")
        h.write(f"#define SAMPLE_FEATS  {F}\n\n")
        h.write(f"extern const float samples_X[SAMPLE_COUNT][SAMPLE_TIME][SAMPLE_FEATS];\n")
        h.write(f"extern const int   samples_y[SAMPLE_COUNT];\n")
        h.write(f"extern const float scaler_mean[SAMPLE_FEATS];\n")
        h.write(f"extern const float scaler_std[SAMPLE_FEATS];\n")

    # --------- C FILE ---------
    with open(c_file, "w") as c:
        c.write(f'#include "{prefix}.h"\n\n')
        c.write("const float samples_X[SAMPLE_COUNT][SAMPLE_TIME][SAMPLE_FEATS] = {\n")
        for s in range(S):
            c.write("  {\n")
            for t in range(T):
                row = ", ".join(f"{v:.6f}f" for v in X_sel[s, t])
                c.write(f"    {{ {row} }},\n")
            c.write("  },\n")
        c.write("};\n\n")
        c.write("const int samples_y[SAMPLE_COUNT] = { ")
        c.write(", ".join(str(int(v)) for v in y_sel))
        c.write(" };\n\n")
        c.write("const float scaler_mean[SAMPLE_FEATS] = { ")
        c.write(", ".join(f"{v:.6f}f" for v in scaler_mean))
        c.write(" };\n\n")
        c.write("const float scaler_std[SAMPLE_FEATS] = { ")
        c.write(", ".join(f"{v:.6f}f" for v in scaler_std))
        c.write(" };\n")
    print(f"Generated {h_file} and {c_file} successfully.")


def send_sample(window, ser):
    flat = window.astype(np.float32).flatten()
    ser.write(flat.tobytes())
    print(f"Sent {len(flat.tobytes())} bytes")  # confirm bytes left Python

import pdb 
def main(): 
    # OUTPUT_DIR = "Preprocessed_Data/STM_SAMPLES"
    # PREFIX = "samples_fp32"
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    SAMPLE_COUNT = 2  # number of samples to export
    X_train, y_train, X_test, y_test = load_data()
    X_sel, y_sel  = create_balanced_csv_samples(X_test, y_test, per_class=SAMPLE_COUNT)
    
    ser = serial.Serial('COM6', 115200)  # or /dev/ttyUSB0 on Linux
    for i in range(len(X_test)):
        pred = send_sample(X_test[i], ser)
        print(f"Sample {i}: {'ATTACK' if pred else 'AMBIENT'}")

    ser.close()
    # write_samples_files_fp32(X_sel, y_sel, scaler_mean, scaler_std, prefix="samples_fp32")
    # model = load_model("saved_models/stm32_ROAD_model32_final.keras", compile=True)
    # print("Attack sample prediction:",
    #     model.predict(X_sel[y_sel == 1]))
    # print("Normal sample prediction:",
    #     model.predict(X_sel[y_sel == 0]))
    # print(y_sel)

if __name__ == "__main__":
    main()