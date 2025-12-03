# Snakefile for ECGml project
# Defines the workflow for ECG data download, preprocessing, R-peak detection,
# HRV analysis, ECG visualisation, morphology extraction, delineation, 
# feature summarisation, beat-level dataset generation and baseline ML training.



#  all main outputs
rule all:
    input:
        "results/preprocessed.npy",
        "results/r_peaks.npy",
        "results/rr_intervals.npy",
        "results/hrv_metrics.json",
        "results/ecg_plot.png",
        "results/features.json",
        "results/delineation.json",
        "results/feature_summary.csv",
        "results/feature_summary.json",
        "results/beat_features.csv",
        "results/model_metrics.json",
        "results/rf_feature_importance.csv",
        "results/figures/beat_level_feature_correlation.png"



# Step 1: downloading ECG data
rule download_data:
    output:
        "data/100.dat",
        "data/100.hea",
        "data/100.atr"
    shell:
        "python download_data.py"

# Step 2: preprocessing ECG signal
rule preprocess:
    input:
        "data/100.dat",
        "data/100.hea"
    output:
        "results/preprocessed.npy"
    shell:
        "python preprocessing.py"

# Step 3: detect R-peaks
rule r_peak_detection:
    input:
        "results/preprocessed.npy"
    output:
        "results/r_peaks.npy"
    shell:
        "python r_peak_detection.py"

# Step 4: HRV analysis
rule hrv_analysis:
    input:
        "results/r_peaks.npy"
    output:
        "results/rr_intervals.npy",
        "results/hrv_metrics.json"
    shell:
        "python hrv.py"

# Step 5: plot ECG with R-peaks
rule plot_ecg:
    input:
        "results/preprocessed.npy",
        "results/r_peaks.npy"
    output:
        "results/ecg_plot.png"
    shell:
        "python plot_ecg.py"

# Step 6: QRS morphology extraction
rule qrs_features:
    input:
        "results/preprocessed.npy",
        "results/r_peaks.npy"
    output:
        "results/features.json"
    shell:
        "python features.py"

# Step 7: neuroKit2 ECG delineation
rule delineate_ecg:
    input:
        "results/preprocessed.npy",
        "results/r_peaks.npy"
    output:
        "results/delineation.json"
    shell:
        "python neurokit_delineate.py"


# Step 8: feature summary table
rule feature_summary:
    input:
        "results/hrv_metrics.json",
        "results/features.json",
        "results/delineation.json"
    output:
        "results/feature_summary.csv",
        "results/feature_summary.json"
    shell:
        "python features_summary.py"

# Step 9: beat-level correlation heatmap -- - corrected
rule beat_level_correlation:
    input:
        "results/delineation.json"
    output:
        "results/figures/beat_level_feature_correlation.png"
    shell:
        "python beat_level_correlation.py"

# Step 10: building beat-level ML dataset
rule build_beat_dataset:
    input:
        "results/delineation.json"
    output:
        "results/beat_features.csv"
    shell:
        "python build_beat_dataset.py"

# Step 11: train baseline ML classifier
rule train_classifier:
    input:
        "results/beat_features.csv"
    output:
        "results/model_metrics.json",
        "results/rf_feature_importance.csv"
    shell:
        "python train_classifier.py"