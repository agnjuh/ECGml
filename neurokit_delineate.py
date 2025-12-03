import os
import json
import numpy as np
import wfdb
import neurokit2 as nk

# output directory
os.makedirs("results", exist_ok=True)

# loading preprocessed ECG and R-peaks
ecg = np.load("results/preprocessed.npy")
r_peaks = np.load("results/r_peaks.npy")

# loading sampling frequency from MIT-BIH header
record = wfdb.rdrecord("data/100")
fs = record.fs


def interval_ms(start_idx, end_idx, fs):
    """
    Convert sample index intervals to milliseconds.
    start_idx and end_idx must be sequences of integers.
    """
    start = np.array(start_idx, dtype=float)
    end = np.array(end_idx, dtype=float)

    if len(start) == 0 or len(end) == 0:
        return np.array([])

    n = min(len(start), len(end))
    return (end[:n] - start[:n]) / fs * 1000.0


# cleaning ECG before delineation
ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs)

# NeuroKit2 requires dict format for R-peaks
rpeaks_dict = {"ECG_R_Peaks": r_peaks}

# wavelet-based delineation
# NeuroKit2 returns (signals, waves)
_, waves = nk.ecg_delineate(
    ecg_clean,
    rpeaks_dict,
    sampling_rate=fs,
    method="dwt"
)

# extract boundaries that NeuroKit2 reliably provides
P_onsets  = waves.get("ECG_P_Onsets", [])
P_offsets = waves.get("ECG_P_Offsets", [])
T_onsets  = waves.get("ECG_T_Onsets", [])
T_offsets = waves.get("ECG_T_Offsets", [])

# R-peaks: from waves if available, else from our detection
R_peaks_waves = waves.get("ECG_R_Peaks", [])
if len(R_peaks_waves) > 0:
    R_used = np.array(R_peaks_waves, dtype=int)
else:
    R_used = np.array(r_peaks, dtype=int)


# DURATIONS
P_durations_ms = interval_ms(P_onsets, P_offsets, fs)
T_durations_ms = interval_ms(T_onsets, T_offsets, fs)

# SURROGATE INTERVALS
# PR_peak = P onset → R peak
PR_peak_ms = interval_ms(P_onsets, R_used, fs)

# QT_peak = R peak → T offset
QT_peak_ms = interval_ms(R_used, T_offsets, fs)

# QTc Bazett
if len(R_used) > 1 and QT_peak_ms.size > 0:
    rr_intervals = np.diff(R_used) / fs
    rr_mean = np.mean(rr_intervals)
    QTc_ms = QT_peak_ms / np.sqrt(rr_mean)
else:
    QTc_ms = np.full_like(QT_peak_ms, np.nan)


# SUMMARY
summary = {
    "PR_peak_interval_ms_mean": float(np.nanmean(PR_peak_ms)) if PR_peak_ms.size else None,
    "QT_peak_interval_ms_mean": float(np.nanmean(QT_peak_ms)) if QT_peak_ms.size else None,
    "QTc_ms_mean":              float(np.nanmean(QTc_ms))     if QTc_ms.size else None,

    "P_duration_ms_mean":       float(np.nanmean(P_durations_ms)) if P_durations_ms.size else None,
    "T_duration_ms_mean":       float(np.nanmean(T_durations_ms)) if T_durations_ms.size else None,
}

# SAVE OUTPUT
results = {
    "sampling_rate_hz": float(fs),
    "n_beats_used": int(len(R_used)),
    "summary": summary,
    "per_beat": {
        "PR_peak_interval_ms": PR_peak_ms.tolist(),
        "QT_peak_interval_ms": QT_peak_ms.tolist(),
        "QTc_ms": QTc_ms.tolist(),
        "P_duration_ms": P_durations_ms.tolist(),
        "T_duration_ms": T_durations_ms.tolist(),

        "P_onsets": list(P_onsets),
        "P_offsets": list(P_offsets),
        "T_onsets": list(T_onsets),
        "T_offsets": list(T_offsets),
        "R_peaks": R_used.tolist(),
    },
}

with open("results/delineation.json", "w") as f:
    json.dump(results, f, indent=4)

print("NeuroKit2 ECG delineation completed.")
print("PR_peak interval mean (ms):", summary["PR_peak_interval_ms_mean"])
print("QT_peak interval mean (ms):", summary["QT_peak_interval_ms_mean"])
print("QTc mean (ms):", summary["QTc_ms_mean"])