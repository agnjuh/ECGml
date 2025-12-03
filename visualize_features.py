import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# output directory check
os.makedirs("results/figures", exist_ok=True)

# helper
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# loading all inputs
hrv = load_json("results/hrv_metrics.json")
morpho = load_json("results/features.json")
delineation = load_json("results/delineation.json")

# prepare feature vectors

rr_intervals = np.diff(np.load("results/r_peaks.npy")) / 360.0 * 1000.0  # ms

del_sum = delineation["summary"]
per_beat = delineation["per_beat"]

PR = np.array(per_beat["PR_peak_interval_ms"])
QT = np.array(per_beat["QT_peak_interval_ms"])
QTc = np.array(per_beat["QTc_ms"])
P_dur = np.array(per_beat["P_duration_ms"])
T_dur = np.array(per_beat["T_duration_ms"])

QRS_core = np.array(morpho["summary"]["QRS_core_width_ms_mean"])
# Actually, the per-beat QRS array is not saved; use summary only.

# R amplitude per beat unavailable → use summary only
R_amp_mean = morpho["summary"]["R_amplitude_mean"]

# plots

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# 1. Histogram – PR_peak
plt.figure(figsize=(6,4))
sns.histplot(PR, bins=40, kde=True, color="blue")
plt.xlabel("PR_peak interval (ms)")
plt.ylabel("Count")
plt.title("Distribution of PR_peak interval")
savefig("results/figures/pr_peak_hist.png")


# 2. Histogram – QT_peak
plt.figure(figsize=(6,4))
sns.histplot(QT, bins=40, kde=True, color="purple")
plt.xlabel("QT_peak interval (ms)")
plt.ylabel("Count")
plt.title("Distribution of QT_peak interval")
savefig("results/figures/qt_peak_hist.png")


# 3. Histogram – QTc
plt.figure(figsize=(6,4))
sns.histplot(QTc, bins=40, kde=True, color="green")
plt.xlabel("QTc (ms)")
plt.ylabel("Count")
plt.title("Distribution of QTc (Bazett)")
savefig("results/figures/qtc_hist.png")


# 4. Distributions – P and T durations
plt.figure(figsize=(6,4))
sns.histplot(P_dur, bins=40, kde=True, color="orange", label="P duration")
sns.histplot(T_dur, bins=40, kde=True, color="red", label="T duration", alpha=0.6)
plt.xlabel("Duration (ms)")
plt.ylabel("Count")
plt.legend()
plt.title("P-wave and T-wave durations")
savefig("results/figures/p_t_durations.png")


# 5. Scatter – RR vs QTc
plt.figure(figsize=(6,4))
plt.scatter(rr_intervals, QTc[:len(rr_intervals)], s=10, alpha=0.6)
plt.xlabel("RR interval (ms)")
plt.ylabel("QTc (ms)")
plt.title("RR interval vs QTc")
savefig("results/figures/rr_vs_qtc.png")


# 6. Correlation heatmap -- corrected
# Build feature matrix
feature_dict = {
    "RR_mean_ms": hrv["time_domain"]["mean_RR_ms"],
    "SDNN_ms": hrv["time_domain"]["SDNN_ms"],
    "RMSSD_ms": hrv["time_domain"]["RMSSD_ms"],
    "HR_bpm": hrv["time_domain"]["HR_bpm"],
    "PR_peak_mean_ms": del_sum["PR_peak_interval_ms_mean"],
    "QT_peak_mean_ms": del_sum["QT_peak_interval_ms_mean"],
    "QTc_mean_ms": del_sum["QTc_ms_mean"],
    "P_dur_mean_ms": del_sum["P_duration_ms_mean"],
    "T_dur_mean_ms": del_sum["T_duration_ms_mean"],
    "QRS_core_width_ms": morpho["summary"]["QRS_core_width_ms_mean"],
    "R_amplitude_mean": morpho["summary"]["R_amplitude_mean"],
}

df = pd.DataFrame([feature_dict])

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="viridis", fmt=".2f")
plt.title("Correlation among ECG features")
savefig("results/figures/feature_correlation.png")


print("All plots saved under results/figures/")
