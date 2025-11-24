# %%
import os, glob, json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
import pdb
import json
from pathlib import Path
import pandas as pd

# %%
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
#gen by GitHub Copilot
# combined list used by the rest of the pipeline
DATA_FILES = AMBIENT_DATA + ATTACK_DATA
# %%
DATA_DIR = "road/signal_extractions"
METADATA_PATH = "road/metadata.csv"
WINDOW_SIZE = 256
STRIDE = 128
TOP_K_IDS = 256
ATTACK_THRESHOLD = 1
MIN_FRAMES = 50
SAVE_DIR = "preprocessed_npz"
RND = 42

def read_metadata_df(base_path: str):
    base = Path(base_path)
    ambient_path = base / "signal_extractions" / "ambient" / "metadata.json"
    attacks_path = base / "signal_extractions" / "attacks" / "metadata.json"
    rows = []
    with ambient_path.open("r") as f:
        ambient_json = json.load(f)
    for name, meta in ambient_json.items():
        rows.append({
            "name": name,
            "category": "ambient",
            "description": meta.get("description"),
            "elapsed_sec": meta.get("elapsed_sec"),
            "modified": meta.get("modified"),
            "on_dyno": meta.get("on_dyno"),
            "injection_data_str": None,
            "injection_id": None,
            "injection_interval": None,
        })
    with attacks_path.open("r") as f:
        attacks_json = json.load(f)
    for name, meta in attacks_json.items():
        rows.append({
            "name": name,
            "category": "attack",
            "description": meta.get("description"),
            "elapsed_sec": meta.get("elapsed_sec"),
            "modified": meta.get("modified"),
            "on_dyno": meta.get("on_dyno"),
            "injection_data_str": meta.get("injection_data_str"),
            "injection_id": meta.get("injection_id"),
            "injection_interval": meta.get("injection_interval"),
        })
    df = pd.DataFrame(rows)
    df = df[
        [
            "name",
            "category",
            "description",
            "elapsed_sec",
            "modified",
            "on_dyno",
            "injection_data_str",
            "injection_id",
            "injection_interval",
        ]
    ]
    return df

def read_csv_files():
    df = pd.read_csv("road/signal_extractions/ambient/ambient_dyno_drive_basic_long.csv")
    pdb.set_trace()
    print(df.head())

def main():
    read_csv_files()
    df = read_metadata_df("road")
    print(df.head())
    print("\nTotal rows:", len(df))

if __name__ == "__main__":
    main()
# %%
