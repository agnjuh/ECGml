import os
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# load preprocessed signal
signal = np.load("results/preprocessed.npy")

# sampling frequency from header
record = wfdb.rdrecord("data/100")
fs = record.fs

def bandpass_filter_qrs(sig, fs, low=5.0, high=15.0, order=2):
    """Narrow bandpass filter to emphasize QRS complex."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)

# further filter to emphasize QRS
filtered = bandpass_filter_qrs(signal, fs)

#normalize
filtered = (filtered - np.mean(filtered)) / np.std(filtered)

# adaptive threshold based on peak amplitude
threshold = 0.35 * np.max(filtered)

# Detect peaks (R-peaks)
peaks, props = find_peaks(filtered, height=threshold, distance=0.25 * fs)

# Save R-peaks
np.save("results/r_peaks.npy", peaks)

# Basic RR intervals
rr_intervals = np.diff(peaks) / fs
mean_rr = float(np.mean(rr_intervals))

print(f"Detected {len(peaks)} R-peaks")
print(f"Mean RR interval: {mean_rr:.3f} s")

# Diagnostic plot: first 10 seconds
window = 10  # seconds
samples = int(window * fs)
t = np.arange(samples) / fs

plt.figure(figsize=(12, 4))
plt.plot(t, filtered[:samples], label="Filtered ECG")
mask = peaks < samples
plt.plot(peaks[mask] / fs, filtered[peaks[mask]], "ro", label="R-peaks")
plt.title("Filtered ECG with detected R-peaks (first 10 seconds)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("results/ecg_plot.png", dpi=150)
plt.close()

