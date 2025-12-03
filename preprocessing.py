import os
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt

os.makedirs("results", exist_ok=True)

# loading ECG record (MIT-BIH record 100)
record = wfdb.rdrecord("data/100")
signal = record.p_signal[:, 0]  # use lead 1
fs = record.fs

def bandpass_filter(sig, fs, low=0.5, high=40.0, order=2):
    """Bandpass filter to remove baseline wander and high-frequency noise."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)

# apply bandpass filtering
filtered = bandpass_filter(signal, fs)

# normalize (zero mean, unit variance)
normalized = (filtered - np.mean(filtered)) / np.std(filtered)

# save preprocessed signal
np.save("results/preprocessed.npy", normalized)

print("Preprocessing completed.")
print(f"Preprocessed signal length: {len(normalized)} samples, fs = {fs} Hz")
