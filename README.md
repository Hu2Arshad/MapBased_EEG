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
within ±80 µV. Subjects with ≥100 supertrials in Dir0, S1 and ISI. 5×5 subject CV

## Data
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

# Step 5: report BACC/ROC, all contrast combinations
python run/Step5_Report.py results/three_contrasts_preds.csv
```
