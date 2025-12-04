# ECGml — Modular ECG signal processing & machine learning pipeline ![status](https://img.shields.io/badge/Status-Experimental_Research_Project-purple) ![docs](https://img.shields.io/badge/Docs-Continuously_Updated-6f42c1)

ECGml is a fully reproducible, Snakemake-driven pipeline for ECG signal processing, wave delineation, morphological feature extraction, and baseline machine-learning classification. It operates directly on MIT-BIH ECG records and produces visualisations, and model outputs.

---

## Features

### Signal processing
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

### Beat-level dataset
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

--------------------
### Technical notes
- QRS onset/offset estimation is handled by NeuroKit2; custom derivative-based methods are included only for exploratory morphology and are not intended as clinical QRS duration measures.
- PR and QT values are surrogate peak-to-peak intervals, used because MIT-BIH signals often lack reliable isoelectric baselines for true onset/offset detection.
- Beat labels are reduced to three classes (normal / ventricular / other) to stabilise training under strong class imbalance.
- Delineation-derived features and morphology-derived descriptors are kept separate to avoid mixing method-specific assumptions in downstream models.

Create a virtual environment:
Create a virtual environment:

```bash
python3 -m venv ecg_env
source ecg_env/bin/activate
```

```
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
```

```
Machine Learning Output:
- Binary classification performance
- ROC AUC scores
- Confusion matrices
- Random Forest feature importances
All results stored in results/model_metrics.json
```

This project is distributed under the terms described in the LICENSE file.
