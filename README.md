# ECGml — Modular ECG signal processing & machine learning pipeline

ECGml is a fully reproducible, Snakemake-driven pipeline for ECG signal processing, wave delineation, morphological feature extraction, and baseline machine-learning classification. It operates directly on MIT-BIH ECG records and produces visualisations, and model outputs.

---

## Features

### Signal Processing
- MIT-BIH record download (WFDB)
- Bandpass filtering and normalisation
- Robust R-peak detection
- HRV (time-domain and frequency-domain)

### Morphology & Delineation
- Custom QRS core-region morphology extraction  
- NeuroKit2 wavelet-based ECG delineation  
  - P-wave duration  
  - T-wave duration  
  - surrogate PR_peak interval  
  - surrogate QT_peak interval  
  - QTc (Bazett)

### Beat-Level Dataset
- RR interval per beat  
- P/T durations  
- PR_peak, QT_peak, QTc  
- Integration of MIT-BIH beat annotations (normal / ventricular / other)

### Machine Learning
- Automatic beat-level feature matrix
- Baseline binary classifier (normal vs abnormal)
- Logistic Regression and Random Forest models
- Metrics: accuracy, ROC AUC, confusion matrix, feature importance

### Visualisations
- ECG plots with R-peaks  
- PR, QT, QTc histograms  
- P & T duration distributions  
- RR–QTc relationship  
- Beat-level correlation heatmap  

---

## Installation

Create a virtual environment:

```bash
python3 -m venv ecg_env
source ecg_env/bin/activate

Outputs written to:
results/
│
├── preprocessed.npy
├── r_peaks.npy
├── rr_intervals.npy
├── hrv_metrics.json
├── features.json
├── delineation.json
├── feature_summary.csv
├── beat_features.csv
├── model_metrics.json
└── figures/


Machine Learning Output
the trained models produce:
- Binary classification performance
- ROC AUC scores
- Confusion matrices
- Random Forest feature importances
All results stored in results/model_metrics.json

Project structure: 
ECGml/
│
├── Snakefile
├── requirements.txt
├── download_data.py
├── preprocessing.py
├── r_peak_detection.py
├── hrv.py
├── features.py
├── neurokit_delineate.py
├── features_summary.py
├── beat_level_correlation.py
├── build_beat_dataset.py
└── train_classifier.py