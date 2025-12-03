import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# figure directory -- check
os.makedirs("results/figures", exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# loading per-beat delineation features
delineation = load_json("results/delineation.json")
per_beat = delineation["per_beat"]

PR = np.array(per_beat["PR_peak_interval_ms"], dtype=float)
QT = np.array(per_beat["QT_peak_interval_ms"], dtype=float)
QTc = np.array(per_beat["QTc_ms"], dtype=float)
P_dur = np.array(per_beat["P_duration_ms"], dtype=float)
T_dur = np.array(per_beat["T_duration_ms"], dtype=float)

# R-peaks for RR computation
r_peaks = np.array(per_beat["R_peaks"], dtype=int)
# RR in ms
RR_ms = np.diff(r_peaks) / delineation["sampling_rate_hz"] * 1000.0

# align lengths: use the minimal common length across all vectors
lengths = [
    len(RR_ms),
    len(PR),
    len(QT),
    len(QTc),
    len(P_dur),
    len(T_dur),
]
n = min(lengths)

RR_ms = RR_ms[:n]
PR = PR[:n]
QT = QT[:n]
QTc = QTc[:n]
P_dur = P_dur[:n]
T_dur = T_dur[:n]

# build beat-level feature DataFrame
df = pd.DataFrame({
    "RR_ms": RR_ms,
    "PR_peak_ms": PR,
    "QT_peak_ms": QT,
    "QTc_ms": QTc,
    "P_duration_ms": P_dur,
    "T_duration_ms": T_dur,
})

# drop any rows with NaN to avoid artefacts in the correlation
df = df.dropna()

# compute correlation matrix
corr = df.corr()

# plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", square=True)
plt.title("Beat-level correlation among ECG features")
plt.tight_layout()
out_path = "results/figures/beat_level_feature_correlation.png"
plt.savefig(out_path, dpi=300)
plt.close()

print("Beat-level correlation heatmap saved to:", out_path)
