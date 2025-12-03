import os
import json
import numpy as np
import pandas as pd
import wfdb

# output directory
os.makedirs("results", exist_ok=True)

# helpers
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# loading per-beat features
delineation = load_json("results/delineation.json")
per_beat = delineation["per_beat"]
fs = delineation["sampling_rate_hz"]

PR = np.array(per_beat["PR_peak_interval_ms"], dtype=float)
QT = np.array(per_beat["QT_peak_interval_ms"], dtype=float)
QTc = np.array(per_beat["QTc_ms"], dtype=float)
P_dur = np.array(per_beat["P_duration_ms"], dtype=float)
T_dur = np.array(per_beat["T_duration_ms"], dtype=float)

# R-peak positions
R_peaks = np.array(per_beat["R_peaks"], dtype=int)

# RR intervals (ms) – length is len(R_peaks) - 1
RR_ms = np.diff(R_peaks) / fs * 1000.0

# align lengths (drop last beat so everything has same n)
n = min(len(RR_ms), len(PR), len(QT), len(QTc), len(P_dur), len(T_dur))
RR_ms = RR_ms[:n]
PR = PR[:n]
QT = QT[:n]
QTc = QTc[:n]
P_dur = P_dur[:n]
T_dur = T_dur[:n]
R_peaks_used = R_peaks[:n]

# loading MIT-BIH annotation and creating labels
ann = wfdb.rdann("100", "atr", pn_dir="mitdb", sampfrom=0)

# ann.sample: sample indices, ann.symbol: beat type codes
ann_samples = np.array(ann.sample, dtype=int)
ann_symbols = np.array(ann.symbol, dtype=str)

# map WFDB beat symbols to coarse labels
# N, L, R, e.g. normal; V = ventricular; everything else = other
label_map = {
    "N": "normal",
    "L": "normal",
    "R": "normal",
    "V": "ventricular",
}

labels = []
for rp in R_peaks_used:
    # find nearest annotation sample
    idx = np.argmin(np.abs(ann_samples - rp))
    sym = ann_symbols[idx]
    labels.append(label_map.get(sym, "other"))

labels = np.array(labels, dtype=str)

# build DataFrame
df = pd.DataFrame({
    "RR_ms": RR_ms,
    "PR_peak_ms": PR,
    "QT_peak_ms": QT,
    "QTc_ms": QTc,
    "P_duration_ms": P_dur,
    "T_duration_ms": T_dur,
    "R_peak_sample": R_peaks_used,
    "label": labels,
})

# drop any rows with NaN in features
df = df.dropna().reset_index(drop=True)

# save dataset
out_csv = "results/beat_features.csv"
df.to_csv(out_csv, index=False)

print(f"Beat-level feature dataset saved to: {out_csv}")
print("Class counts:")
print(df["label"].value_counts())
