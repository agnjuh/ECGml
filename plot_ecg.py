import wfdb
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

# read record
record = wfdb.rdrecord('data/100')
signal = record.p_signal[:, 0]  # lead 1
fs = record.fs

# normalise
normalized_signal = (signal - np.mean(signal)) / np.std(signal)

# detect peaks
peaks, _ = find_peaks(normalized_signal, height=0.5, distance=0.2*fs)

# plot
t = np.arange(len(normalized_signal)) / fs
plt.figure(figsize=(12, 4))
plt.plot(t, normalized_signal, label='ECG Lead 1')
plt.plot(peaks/fs, normalized_signal[peaks], 'ro', label='R-peaks')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (normalized)")
plt.title("ECG Lead 1 with detected R-peaks")
plt.legend()
plt.show()

