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

import os
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit

METADATA_PATH = "road/metadata.csv"
WINDOW_SIZE = 256
STRIDE = 128
TOP_K_IDS = 256
ATTACK_THRESHOLD = 1
MIN_FRAMES = 50
SAVE_DIR = "preprocessed_npz"
RND = 42

def read_metadata(path):
    meta = defaultdict(list)
    if not os.path.exists(path):
        return meta
    try:
        df = pd.read_csv(path)
        file_col = next((c for c in df.columns if 'file' in c.lower() or 'source' in c.lower()), None)
        start_col = next((c for c in df.columns if 'start' in c.lower() or 'from' in c.lower()), None)
        end_col = next((c for c in df.columns if 'end' in c.lower() or 'to' in c.lower()), None)
        if file_col and start_col and end_col:
            for _, r in df.iterrows():
                try:
                    fname = os.path.basename(str(r[file_col]))
                    s = float(r[start_col]); e = float(r[end_col])
                    meta[fname].append((s,e))
                except:
                    continue
            return meta
    except:
        pass
    try:
        with open(path) as f:
            j = json.load(f)
        for k,v in j.items():
            for s,e in v:
                meta[os.path.basename(k)].append((float(s), float(e)))
    except:
        pass
    return meta

def parse_payload(row):
    if 'data' in row and pd.notnull(row['data']):
        raw = str(row['data']).replace("0x","").replace(" ","").replace(":","").replace("-","")
        if len(raw)%2!=0: raw = '0'+raw
        b = []
        for i in range(0, len(raw), 2):
            try: b.append(int(raw[i:i+2],16))
            except: b.append(0)
        b += [0]*8
        return b[:8]
    bcols = [c for c in row.index if c.lower().startswith('b') and c[1:].isdigit()]
    if bcols:
        out = []
        for i in range(8):
            c = f"b{i}"
            try: out.append(int(row.get(c,0)))
            except: out.append(0)
        return out
    return [0]*8

def load_csv(p):
    df = pd.read_csv(p, low_memory=False)
    ts_cols = [c for c in df.columns if 'time' in c.lower() or 'timestamp' in c.lower()]
    if not ts_cols:
        raise ValueError("no timestamp column")
    ts = ts_cols[0]
    df = df.copy()
    df['timestamp'] = pd.to_numeric(df[ts], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    id_cols = [c for c in df.columns if c.lower()=='id' or 'arbitration' in c.lower() or 'arb' in c.lower()]
    if id_cols:
        def pid(x):
            s = str(x)
            try:
                if s.lower().startswith('0x'): return int(s,16)
                return int(s)
            except:
                digs = ''.join(ch for ch in s if ch.isdigit())
                return int(digs) if digs else 0
        df['id'] = df[id_cols[0]].apply(pid)
    else:
        df['id'] = 0
    payloads = df.apply(parse_payload, axis=1, result_type='expand')
    payloads.columns = [f"b{i}" for i in range(8)]
    for i in range(8):
        df[f"b{i}"] = pd.to_numeric(payloads[i], errors='coerce').fillna(0).astype(int)
    df = df[['timestamp','id'] + [f"b{i}" for i in range(8)]]
    df['src_file'] = os.path.basename(p)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def label_mask(df, intervals):
    if not intervals: return np.zeros(len(df), dtype=bool)
    ts = df['timestamp'].values
    mask = np.zeros(len(df), dtype=bool)
    for s,e in intervals:
        mask |= ((ts >= s) & (ts <= e))
    return mask

def build_id_map(dfs, k):
    cnt = Counter()
    for d in dfs: cnt.update(d['id'].astype(int).tolist())
    top = [iid for iid,_ in cnt.most_common(k)]
    return {iid: idx+1 for idx,iid in enumerate(top)}

def df_to_windows(df, id_map, wsize, stride, attack_mask):
    bs = np.stack([df[f"b{i}"].values.astype(np.uint8) for i in range(8)], axis=1).astype(np.float32)/255.0
    ids = np.array([id_map.get(int(x),0) for x in df['id'].values], dtype=np.float32)
    if id_map:
        ids = ids/(max(1,max(id_map.values()))+1.0)
    ts = df['timestamp'].values
    dt = np.diff(ts, prepend=ts[0])
    dt = np.clip(dt, 0.0, 1.0).astype(np.float32)
    feats = np.concatenate([bs, ids.reshape(-1,1), dt.reshape(-1,1)], axis=1)
    N = len(df)
    Xs=[]; ys=[]
    for s in range(0, max(1, N - wsize + 1), stride):
        e = s + wsize
        if e > N: break
        win = feats[s:e]
        lbl = 1 if int(attack_mask[s:e].sum()) >= ATTACK_THRESHOLD else 0
        Xs.append(win)
        ys.append(lbl)
    if not Xs:
        return np.zeros((0,wsize,feats.shape[1]),dtype=np.float32), np.zeros((0,),dtype=np.int64)
    return np.stack(Xs,axis=0).astype(np.float32), np.array(ys,dtype=np.int64)

def group_split_save(X, y, groups, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RND)
    tr_idx, te_idx = next(gss.split(X,y,groups))
    X_trall, y_trall = X[tr_idx], y[tr_idx]
    groups_tr = np.array(groups)[tr_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=RND)
    tri, vali = next(gss2.split(X_trall, y_trall, groups_tr))
    tr_sel = tr_idx[tri]; val_sel = tr_idx[vali]
    np.savez_compressed(os.path.join(save_dir,"train.npz"), X=X[tr_sel], y=y[tr_sel])
    np.savez_compressed(os.path.join(save_dir,"val.npz"), X=X[val_sel], y=y[val_sel])
    np.savez_compressed(os.path.join(save_dir,"test.npz"), X=X[te_idx], y=y[te_idx])
    print("saved", save_dir)

def main():
    files = DATA_FILES
    meta = read_metadata(METADATA_PATH)
    dfs=[]; groups=[]
    for p in files:
        try:
            df = load_csv(p)
        except Exception as e:
            print("skip", p, e); continue
        if len(df) < MIN_FRAMES: continue
        dfs.append(df)
    if not dfs: raise SystemExit("no road CSVs found")
    id_map = build_id_map(dfs, TOP_K_IDS)
    Xs=[]; ys=[]
    for d in dfs:
        src = d['src_file'].iloc[0]
        mask = label_mask(d, meta.get(src,[]))
        X, y = df_to_windows(d, id_map, WINDOW_SIZE, STRIDE, mask)
        if X.shape[0]==0: continue
        Xs.append(X); ys.append(y); groups.extend([src]*len(y))
    X = np.concatenate(Xs, axis=0); y = np.concatenate(ys, axis=0)
    scaler = MinMaxScaler()
    n, t, c = X.shape
    X_scaled = scaler.fit_transform(X.reshape(-1,c)).reshape(n,t,c)
    group_split_save(X_scaled, y, groups, SAVE_DIR)
    print("done", X.shape, "labels", int(y.sum()))

if __name__ == "__main__":
    main()