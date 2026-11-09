# EEG HC-vs-MCI — three-contrast EEGNet pipeline

Binary classification of Healthy Control (HC) vs Mild Cognitive Impairment (MCI) from 32-channel
EEG recorded during a VR/map spatial perspective-taking task. The model is an EEGNet trained on three
within-subject contrasts and fused by mean probability.

**Headline result (74 subjects, 5×5 CV):** three contrasts `cs1 + cisi + s1isi` = **72.0 ± 10.4 BACC / 76.9 ± 11.0 ROC**.

## Pipeline (run in order)
```
preprocessing/
  Step1_Convert.py        raw sub*.hdf5 -> EEGLAB .set              -> data/converted/{HC,MCI,SCD}
  Step2_Preprocessing.m   data/converted -> bandpass 0.53-30 Hz, resample 200 Hz, notch 60, average
                          reference, ICA infomax + ICLabel (muscle/eye/heart >= 0.9)
                                                                    -> data/preprocessed/{HC,MCI,SCD}   (MATLAB / EEGLAB)
  Step3_Segmentation.py   data/preprocessed -> segment 1-s trials (S2 / S1 / ISI), build the trial cache
                                                                    -> results/segments.pkl
                          (helpers: s1_segment.py, isi_segment.py)
model/
  EEGNet.py               EEGNet architecture (reference; Step4_Train embeds an equivalent)
run/
  Step4_Train.py          train the 3 contrasts -> per-fold/per-epoch predictions CSV
  Step5_Report.py         fuse + report BACC/ROC (e20 & e30) for every contrast combination
```

## The three contrasts (built inside `Step4_Train.py`)
- **cs1**   = Dir0 supertrial − subject's mean S1 supertrial
- **cisi**  = Dir0 supertrial − subject's mean ISI supertrial
- **s1isi** = S1 supertrial − subject's mean ISI supertrial

Supertrials: up to 500 unique random k=10-trial averages per subject/view, kept only if the average stays
within ±80 µV. Subjects with ≥100 supertrials in Dir0, S1 and ISI (minus sub39). 5×5 subject CV, EEGNet
(F1=8, D=2, dropout 0.5), 50 epochs, AdamW 1e-3/1e-2, per-subject-balanced weighting. A subject's probability
is the mean sigmoid over its supertrials; the three contrasts are fused by mean probability.

## Data
Not included (clinical EEG). Raw `sub*.hdf5` go in the source set `Step1_Convert.py` points at
(`DATASET_DIR`); everything after that lands under the repo:
```
data/converted/{HC,MCI,SCD}/sub*.set      # produced by Step 1
data/preprocessed/{HC,MCI,SCD}/*.set      # produced by Step 2
results/segments.pkl                      # produced by Step 3
```

## How to run (from the repo root)
```bash
pip install -r requirements.txt

# Step 1: raw HDF5 -> data/converted/
python preprocessing/Step1_Convert.py

# Step 2: filter + ICA (MATLAB / EEGLAB), data/converted -> data/preprocessed/
#   run preprocessing/Step2_Preprocessing.m

# Step 3: segment + build the trial cache -> results/segments.pkl
python preprocessing/Step3_Segmentation.py

# Step 4: train the three contrasts -> predictions CSV
python run/Step4_Train.py --models cs1,cisi,s1isi --out results/three_contrasts_preds.csv

# Step 5: report fused BACC/ROC (e20 & e30), all contrast combinations
python run/Step5_Report.py results/three_contrasts_preds.csv
```

## Notes
- Preprocessing is **0.53–30 Hz, 200 Hz** (`Step2_Preprocessing.m`).
- `Step3_Segmentation.py` merges the S2 segmenter and the cache builder; `s1_segment.py` and `isi_segment.py`
  are its S1 / ISI helper segmenters.
- `model/EEGNet.py` is the standalone architecture; `run/Step4_Train.py` is self-contained and embeds an
  equivalent EEGNet, so it runs without importing it.
- `results/` and all data folders are git-ignored.
