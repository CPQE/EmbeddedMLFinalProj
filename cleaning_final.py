# %%
import os, glob, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
import random
#https://github.com/ohoaha/can-ids-benchmarking-road/blob/main/ML_Anomaly_experiments.ipynb 

# %%

def combine_data(): 
    AMBIENT_DATA = [
    "road/signal_extractions/ambient/accelerator_attack_drive_1.csv",
    "road/signal_extractions/ambient/accelerator_attack_drive_2.csv",
    "road/signal_extractions/ambient/accelerator_attack_reverse_1.csv",
    "road/signal_extractions/ambient/accelerator_attack_reverse_2.csv",
    "road/signal_extractions/ambient/ambient_dyno_drive_basic_long.csv",
    "road/signal_extractions/ambient/ambient_dyno_drive_basic_short.csv",
    "road/signal_extractions/ambient/ambient_dyno_drive_benign_anomaly.csv",
    "road/signal_extractions/ambient/ambient_dyno_drive_extended_long.csv",
    "road/signal_extractions/ambient/ambient_dyno_drive_extended_short.csv",
    "road/signal_extractions/ambient/ambient_dyno_drive_radio_infotainment.csv",
    "road/signal_extractions/ambient/ambient_dyno_drive_winter.csv",
    "road/signal_extractions/ambient/ambient_dyno_exercise_all_bits.csv",
    "road/signal_extractions/ambient/ambient_dyno_idle_radio_infotainment.csv",
    "road/signal_extractions/ambient/ambient_dyno_reverse.csv",
    "road/signal_extractions/ambient/ambient_highway_street_driving_diagnostics.csv",
    "road/signal_extractions/ambient/ambient_highway_street_driving_long.csv",
    "road/signal_extractions/ambient/correlated_signal_attack_1_masquerade.csv",
    "road/signal_extractions/ambient/correlated_signal_attack_2_masquerade.csv",
    "road/signal_extractions/ambient/correlated_signal_attack_3_masquerade.csv",
    "road/signal_extractions/ambient/max_engine_coolant_temp_attack_masquerade.csv",
    "road/signal_extractions/ambient/max_speedometer_attack_1_masquerade.csv",
    "road/signal_extractions/ambient/max_speedometer_attack_2_masquerade.csv",
    "road/signal_extractions/ambient/max_speedometer_attack_3_masquerade.csv",
    "road/signal_extractions/ambient/reverse_light_off_attack_1_masquerade.csv",
    "road/signal_extractions/ambient/reverse_light_off_attack_2_masquerade.csv",
    "road/signal_extractions/ambient/reverse_light_off_attack_3_masquerade.csv",
    "road/signal_extractions/ambient/reverse_light_on_attack_1_masquerade.csv",
    "road/signal_extractions/ambient/reverse_light_on_attack_2_masquerade.csv",
    "road/signal_extractions/ambient/reverse_light_on_attack_3_masquerade.csv",
    ] 

    ATTACK_DATA = [
    "road/signal_extractions/attacks/accelerator_attack_drive_1.csv",
    "road/signal_extractions/attacks/accelerator_attack_drive_2.csv",
    "road/signal_extractions/attacks/accelerator_attack_reverse_1.csv",
    "road/signal_extractions/attacks/accelerator_attack_reverse_2.csv",
    "road/signal_extractions/attacks/correlated_signal_attack_1_masquerade.csv",
    "road/signal_extractions/attacks/correlated_signal_attack_2_masquerade.csv",
    "road/signal_extractions/attacks/correlated_signal_attack_3_masquerade.csv",
    "road/signal_extractions/attacks/max_engine_coolant_temp_attack_masquerade.csv",
    "road/signal_extractions/attacks/max_speedometer_attack_1_masquerade.csv",
    "road/signal_extractions/attacks/max_speedometer_attack_2_masquerade.csv",
    "road/signal_extractions/attacks/max_speedometer_attack_3_masquerade.csv",
    "road/signal_extractions/attacks/reverse_light_off_attack_1_masquerade.csv",
    "road/signal_extractions/attacks/reverse_light_off_attack_2_masquerade.csv",
    "road/signal_extractions/attacks/reverse_light_off_attack_3_masquerade.csv",
    "road/signal_extractions/attacks/reverse_light_on_attack_1_masquerade.csv",
    "road/signal_extractions/attacks/reverse_light_on_attack_2_masquerade.csv",
    "road/signal_extractions/attacks/reverse_light_on_attack_3_masquerade.csv",
    ]
    print(len(ATTACK_DATA), len(AMBIENT_DATA))
    # combined list used by the rest of the pipeline
    DATA_FILES = AMBIENT_DATA + ATTACK_DATA
    return DATA_FILES
#https://github.com/ohoaha/can-ids-benchmarking-road/blob/main/ML_Anomaly_experiments.ipynb

# %%
def load_capture(path, test=False):
    if test:
        path = 'road/' + path
    print("Reading in: ", path)
    df = pd.read_csv(path)
    # enforce canonical columns
    df = df.rename(columns=lambda c: c.strip())
    # ensure Time is float seconds
    df["Time"] = df["Time"].astype(float)
    # sort by time (important for resampling)
    df = df.sort_values("Time").reset_index(drop=True)
    # forward-fill signal NaNs
    signal_cols = [c for c in df.columns if c.startswith("Signal_")]
    df[signal_cols] = df[signal_cols].ffill().fillna(0) #remove all NAs
    return df

# df_test = load_capture(ATTACK_DATA[15], test = True)
# print(df_test['Label'].value_counts())

# %%
def interpolate_capture(df, target_hz=200):
    t_min, t_max = df["Time"].min(), df["Time"].max()
    # construct uniform time grids
    new_time = np.arange(t_min, t_max, 1.0 / target_hz)
    # reindex to grid
    interp_df = pd.DataFrame({"Time": new_time})
    # only interpolate the signal columns
    signal_cols = [c for c in df.columns if c.startswith("Signal_")]
    for col in signal_cols:
        interp_df[col] = np.interp(new_time, df["Time"].values, df[col].astype(float).values)
    interp_df = pd.merge_asof(interp_df, df[["Time","ID"]], on="Time") #handle IDs by setting them to nearest neighbor
    label_value = int(df["Label"].iloc[0])  # assign 0 or 1 to entire dataframe, TODO may want to reconsider if the data is mixed within datasets
    interp_df["Label"] = label_value
    return interp_df
# df_test = interpolate_capture(df_test, target_hz=200) #may need to tweak after testing with models 

# %%
def parse_metadata():
    something = pd.read_json("road/road/signal_extractions/attacks/metadata.json") 
    flipped = something.T
    flipped.reset_index(inplace=True)
    flipped.columns = ['type', 'description', 'elapsed_sec', 'injection_data_str', 'injection_id',
        'injection_interval', 'modified', 'on_dyno']
    meta_df = flipped[['type', 'elapsed_sec', 'injection_interval', 'injection_id']]
    return meta_df

# %%
def get_key_for_metadata(filename):
    return Path(filename).stem # Convert full path to base name WITHOUT extension.

def create_windows(df, filename, window_sec=3, stride_sec=0.5, target_hz=200):
        # Extract signal columns
    meta_df = parse_metadata()
    attack_meta = {}
    for _, row in meta_df.iterrows(): #turn it into a dictionary again 
        attack_meta[row["type"]] = {
            "interval": row["injection_interval"],
            "elapsed_sec": row["elapsed_sec"],
            "injection_id": row["injection_id"]
        }
    key = get_key_for_metadata(filename)  # <— join key
    meta = attack_meta.get(key, None)
    # Set default labeling = all benign
    injection_interval = None
    if meta and isinstance(meta["interval"], list):
        injection_interval = meta["interval"]  # [start, end]
    win_size = int(window_sec * target_hz)
    stride = int(stride_sec * target_hz)
    X, y = [], []
    for start in range(0, len(df) - win_size, stride):
        end = start + win_size
        window = df.iloc[start:end]
        # Window timestamp range
        t0 = window["Time"].iloc[0]
        t1 = window["Time"].iloc[-1]
        if injection_interval is None:
            label = 0   # ambient or "modified": false
        else:
            inj_start, inj_end = injection_interval
            if (t1 >= inj_start) and (t0 <= inj_end):
                label = 1
            else:
                label = 0
        X.append(window.drop(columns=["Label"]).values)
        y.append(label)
    return np.array(X), np.array(y)
   
# X, y = create_windows(df_test, ATTACK_DATA[0], window_sec = 3, target_hz = 200) #can play with window_sec across some models

# %%
def preprocess_file(path, target_hz=200, window_sec=3):
    df = load_capture(path)
    df = interpolate_capture(df, target_hz=target_hz)
    X, y = create_windows( df, path, window_sec=window_sec, target_hz=target_hz)
    return X, y

def preprocess_all_by_file(files, base_path="road/"):
    all_data = {}  # filename -> (X, y)
    for f in files:
        # print("Processing:", f)
        X, y = preprocess_file(Path(base_path) / f, target_hz = 200, window_sec = 3)
        all_data[f] = (X, y)

    return all_data
# dfs = preprocess_all_by_file(DATA_FILES)

# %%
def file_has_attack(xy_tuple): #dets if session has any attack labels at all
    X, y = xy_tuple
    return np.any(y == 1)

def split_files_by_session(all_data, test_ratio=0.2, seed=42):
    random.seed(seed)
    #shuffle the attack/ambient files as a whole randomly, NOT shuffling time series within each
    files = list(all_data.keys())
    attack_files = [f for f in files if file_has_attack(all_data[f])]
    ambient_files = [f for f in files if not file_has_attack(all_data[f])]
    random.shuffle(attack_files)
    random.shuffle(ambient_files)
    n_attack_test = max(1, int(len(attack_files) * test_ratio))
    n_ambient_test = max(1, int(len(ambient_files) * test_ratio))
    test_files = set(attack_files[:n_attack_test] + ambient_files[:n_ambient_test])
    train_files = set(attack_files[n_attack_test:] + ambient_files[n_ambient_test:])
    print("Train sessions:", len(train_files))
    print("  Attack:", sum(np.any(all_data[f][1] == 1) for f in train_files))
    print("  Benign:", sum(not np.any(all_data[f][1] == 1) for f in train_files))

    print("\nTest sessions:", len(test_files))
    print("  Attack:", sum(np.any(all_data[f][1] == 1) for f in test_files))
    print("  Benign:", sum(not np.any(all_data[f][1] == 1) for f in test_files))
    return train_files, test_files

# train_files, test_files = split_files_by_session(dfs)
#print(test_files)

# %%
def build_dataset_from_split(all_data, train_files, test_files):
    X_train, y_train = [], []
    X_test, y_test   = [], []
    for f, (X, y) in all_data.items():
        if f in train_files:
            X_train.append(X)
            y_train.append(y)
        else:
            X_test.append(X)
            y_test.append(y)
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test  = np.concatenate(X_test, axis=0)
    y_test  = np.concatenate(y_test, axis=0)

    return X_train, y_train, X_test, y_test

# %%
def main():
    window_frequency = 200
    window_sec = 3
    DATA_FILES = combine_data()
    all_data = preprocess_all_by_file(DATA_FILES, base_path="road/")
    train_files, test_files = split_files_by_session(all_data, test_ratio=0.20)
    print("Train sessions:", train_files)
    print("Test sessions:", test_files)
    # 3. Build final arrays
    X_train, y_train, X_test, y_test = build_dataset_from_split(
        all_data, train_files, test_files
    )
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    #save the dataset 
    np.savez_compressed(
        f"roads_canids_windows_{window_frequency}hz_{window_sec}s.npz",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )  

# %%
if __name__ == '__main__':
    main()


