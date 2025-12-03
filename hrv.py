import wfdb
import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import json
import os

# results directory
os.makedirs("results", exist_ok=True)

# loading R-peaks (sample indices)
r_peaks = np.load("results/r_peaks.npy")

# loading sampling frequency from record header
record = wfdb.rdrecord("data/100")
fs = record.fs

# computing RR intervals in seconds
rr_intervals = np.diff(r_peaks) / fs

# saving RR intervals
np.save("results/rr_intervals.npy", rr_intervals)

rr_ms = rr_intervals * 1000.0  # convert to ms

mean_rr = float(np.mean(rr_ms))
sdnn = float(np.std(rr_ms, ddof=1))               # standard deviation of NN intervals
rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) # root mean square of successive differences
hr_bpm = float(60000.0 / mean_rr)                    # mean heart rate in bpm

time_domain = {
    "mean_RR_ms": mean_rr,
    "SDNN_ms": sdnn,
    "RMSSD_ms": rmssd,
    "HR_bpm": hr_bpm,
}


# frequency-domain HRV metrics

# building time axis for RR intervals (tachogram)
t_rr = np.cumsum(np.hstack(([0.0], rr_intervals)))[:-1]

# resample RR tachogram to evenly spaced time grid
fs_resample = 4.0  # Hz, standard choice for HRV analysis
t_uniform = np.arange(0, t_rr[-1], 1.0 / fs_resample)
interp_func = interp1d(t_rr, rr_ms, kind="cubic", fill_value="extrapolate")
rr_resampled = interp_func(t_uniform)

# Welch power spectral density
f, pxx = welch(rr_resampled, fs=fs_resample, nperseg=min(256, len(rr_resampled)))

def band_power(f, pxx, f_low, f_high):
    mask = (f >= f_low) & (f < f_high)
    if not np.any(mask):
        return 0.0
    return float(trapezoid(pxx[mask], f[mask]))

vlf = band_power(f, pxx, 0.003, 0.04)
lf  = band_power(f, pxx, 0.04, 0.15)
hf  = band_power(f, pxx, 0.15, 0.40)
total_power = vlf + lf + hf
lf_hf_ratio = float(lf / hf) if hf > 0 else float("nan")

frequency_domain = {
    "VLF_power": vlf,
    "LF_power": lf,
    "HF_power": hf,
    "Total_power": total_power,
    "LF_HF_ratio": lf_hf_ratio,
}

# saving metrics to JSON
metrics = {
    "time_domain": time_domain,
    "frequency_domain": frequency_domain,
}

with open("results/hrv_metrics.json", "w") as f_out:
    json.dump(metrics, f_out, indent=4)

print("HRV analysis completed.")
print("Time-domain metrics:", time_domain)
print("Frequency-domain metrics:", frequency_domain)
