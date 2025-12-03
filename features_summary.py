import os
import json
import pandas as pd

# output directory check
os.makedirs("results", exist_ok=True)

# helper function
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# loading all feature sources
hrv = load_json("results/hrv_metrics.json")
morpho = load_json("results/features.json")
delineation = load_json("results/delineation.json")

summary_table = []

# HRV metrics
time_domain = hrv.get("time_domain", {})
freq_domain = hrv.get("frequency_domain", {})

summary_table.append(["Mean RR (ms)", time_domain.get("mean_RR_ms")])
summary_table.append(["SDNN (ms)", time_domain.get("SDNN_ms")])
summary_table.append(["RMSSD (ms)", time_domain.get("RMSSD_ms")])
summary_table.append(["Heart rate (bpm)", time_domain.get("HR_bpm")])

summary_table.append(["VLF power", freq_domain.get("VLF_power")])
summary_table.append(["LF power", freq_domain.get("LF_power")])
summary_table.append(["HF power", freq_domain.get("HF_power")])
summary_table.append(["LF/HF ratio", freq_domain.get("LF_HF_ratio")])

# QRS Core Morphology ----custom algorithm
morpho_sum = morpho.get("summary", {})

summary_table.append(["QRS core width mean (ms)", morpho_sum.get("QRS_core_width_ms_mean")])
summary_table.append(["QRS core width SD (ms)", morpho_sum.get("QRS_core_width_ms_std")])
summary_table.append(["R amplitude mean", morpho_sum.get("R_amplitude_mean")])
summary_table.append(["R amplitude SD", morpho_sum.get("R_amplitude_std")])
summary_table.append(["R rise slope mean", morpho_sum.get("R_rise_slope_mean")])
summary_table.append(["R fall slope mean", morpho_sum.get("R_fall_slope_mean")])

# NeuroKit2 morphology
del_sum = delineation.get("summary", {})

summary_table.append(["P duration mean (ms)", del_sum.get("P_duration_ms_mean")])
summary_table.append(["T duration mean (ms)", del_sum.get("T_duration_ms_mean")])
summary_table.append(["PR_peak interval mean (ms)", del_sum.get("PR_peak_interval_ms_mean")])
summary_table.append(["QT_peak interval mean (ms)", del_sum.get("QT_peak_interval_ms_mean")])
summary_table.append(["QTc mean (ms)", del_sum.get("QTc_ms_mean")])

# convert to DataFrame
df = pd.DataFrame(summary_table, columns=["Feature", "Value"])

# outputs --- saving
df.to_csv("results/feature_summary.csv", index=False)
df.to_json("results/feature_summary.json", orient="records", indent=4)

# print nicely to terminal yay :)
print("\nECG Feature Summary\n" + "-"*60)
print(df.to_string(index=False))
print("\nSaved to results/feature_summary.csv and .json\n")
