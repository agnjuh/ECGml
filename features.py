import os
import json
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

# directory check
os.makedirs("results", exist_ok=True)

# loading preprocessed ECG signal and R-peaks
# preprocessed.npy: bandpass-filtered + normalised ECG (lead 1)
signal = np.load("results/preprocessed.npy")
r_peaks = np.load("results/r_peaks.npy")  # sample indices of R-peaks

# loading sampling frequency from header
record = wfdb.rdrecord("data/100")
fs = record.fs


def bandpass_filter_qrs(sig, fs, low=5.0, high=20.0, order=2):
    """
    Bandpass filter to emphasize QRS complexes.
    Uses a relatively narrow band around the main QRS frequency content.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)


# emphasize QRS region for morphology analysis
qrs_signal = bandpass_filter_qrs(signal, fs)

# window around each R-peak for morphology analysis
pre_ms = 150   # milliseconds before R-peak
post_ms = 150  # milliseconds after R-peak
pre_samples = int(pre_ms / 1000.0 * fs)
post_samples = int(post_ms / 1000.0 * fs)

QRS_core_width_ms = []
R_amplitudes = []
R_rise_slope = []
R_fall_slope = []

for p in r_peaks:
    # skip beats too close to signal borders
    if p - pre_samples < 0 or p + post_samples >= len(qrs_signal):
        continue

    seg = qrs_signal[p - pre_samples : p + post_samples]
    center = pre_samples  # R-peak index within segment

    # absolute amplitude in this window
    seg_abs = np.abs(seg)
    max_abs = np.max(seg_abs)
    if max_abs < 1e-6:
        continue

    # amplitude threshold as a fraction of local maximum
    # 0.1 => defines the "core" of the QRS complex (high-amplitude region)
    thr = 0.1 * max_abs

    # find QRS "core" onset: last crossing from below to above threshold before R
    onset = 0
    for i in range(center, 0, -1):
        if seg_abs[i] >= thr and seg_abs[i - 1] < thr:
            onset = i
            break

    # find QRS "core" offset: first crossing from above to below threshold after R
    offset = len(seg) - 1
    for i in range(center, len(seg) - 1):
        if seg_abs[i] >= thr and seg_abs[i + 1] < thr:
            offset = i
            break

    # core duration in milliseconds
    dur_ms = (offset - onset) / fs * 1000.0
    QRS_core_width_ms.append(dur_ms)

    # R amplitude (from the QRS-emphasised signal)
    R_amp = seg[center]
    R_amplitudes.append(float(R_amp))

    # approximate left and right minima within QRS core region for slope estimation
    left_region = seg[onset:center + 1]
    right_region = seg[center:offset + 1]

    if len(left_region) == 0 or len(right_region) == 0:
        continue

    left_min_idx_local = int(np.argmin(left_region))
    right_min_idx_local = int(np.argmin(right_region))

    left_min_idx = onset + left_min_idx_local
    right_min_idx = center + right_min_idx_local

    left_min_amp = float(seg[left_min_idx])
    right_min_amp = float(seg[right_min_idx])

    # rising slope from left minimum to R-peak: ΔV / Δt
    dt_rise = (center - left_min_idx) / fs
    if dt_rise > 0:
        slope_rise = (R_amp - left_min_amp) / dt_rise
    else:
        slope_rise = np.nan

    # falling slope from R-peak to right minimum
    dt_fall = (right_min_idx - center) / fs
    if dt_fall > 0:
        slope_fall = (right_min_amp - R_amp) / dt_fall
    else:
        slope_fall = np.nan

    R_rise_slope.append(float(slope_rise))
    R_fall_slope.append(float(slope_fall))

# convert lists to numpy arrays
QRS_core_width_ms = np.array(QRS_core_width_ms)
R_amplitudes = np.array(R_amplitudes)
R_rise_slope = np.array(R_rise_slope)
R_fall_slope = np.array(R_fall_slope)

features = {
    "sampling_rate_hz": float(fs),
    "n_beats_used": int(len(QRS_core_width_ms)),
    "summary": {
        "QRS_core_width_ms_mean": float(np.nanmean(QRS_core_width_ms)),
        "QRS_core_width_ms_std": float(np.nanstd(QRS_core_width_ms)),
        "R_amplitude_mean": float(np.nanmean(R_amplitudes)),
        "R_amplitude_std": float(np.nanstd(R_amplitudes)),
        "R_rise_slope_mean": float(np.nanmean(R_rise_slope)),
        "R_rise_slope_std": float(np.nanstd(R_rise_slope)),
        "R_fall_slope_mean": float(np.nanmean(R_fall_slope)),
        "R_fall_slope_std": float(np.nanstd(R_fall_slope)),
    },
}

with open("results/features.json", "w") as f:
    json.dump(features, f, indent=4)

print("QRS morphology extraction finished.")
print("Beats used:", int(len(QRS_core_width_ms)))
print("Mean QRS core width (ms):", float(np.nanmean(QRS_core_width_ms)))
