"""Step 3 - Segmentation + trial cache.
Segments the preprocessed .set files (output of Step2_Preprocessing.m) into 1-s trials and builds the
cache used by Step4_Train.py:
  - segment_all()      : S2 1-s windows per direction (this file)
  - segment_all_s1()   : S1 cue-locked windows (s1_segment.py)
  - segment_isi()      : ISI windows, 1 s before S2 (isi_segment.py)
Run this file to build  <repo>/results/segments.pkl  {sid:{'label','dir0','s1','isi'}}.
Set BP_DIR to the preprocessed .set folder (default <repo>/data/preprocessed).
"""
import os, re, pickle, warnings
import numpy as np, mne
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")

CANON=['Fp1','Fp2','AF3','AF4','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8',
       'CP5','CP1','CP2','CP6','P7','P3','Pz','P4','P8','PO7','PO3','PO4','PO8','Oz']
CUE={10,11,20,21,30,31,40,41}; STIM={21,22,23,24,25,26,27,28,31,32,33,34,35,36,37,38}
END={200,201}; AMB={21,31}
DIRBINS={'Dir0':{21,25,31,35},'Dir90':{22,26,32,36},'Dir180':{23,27,33,37},'Dir270':{24,28,34,38}}
WIN_SEC=1.0; SFREQ=200; WIN=200

def parse_val(d):
    d=str(d).strip(); return int(d[4:]) if d.startswith('val_') else int(float(d))

def onsets(raw, correct_only=True):
    """cue -> S2 stim -> response. correct_only=True keeps ONLY 201 (correct) trials;
    False keeps both 201 and 200. Either way an END event (200 or 201) closes the trial."""
    ev=[]
    for a in raw.annotations:
        try: ev.append((float(a['onset']), parse_val(a['description'])))
        except: pass
    ev.sort()
    IDLE,WS,WE=0,1,2; st=IDLE; stim_t=stim_v=None; res=[]
    for t,v in ev:
        if st==IDLE:
            if v in CUE: st=WS
        elif st==WS:
            if v in STIM: stim_t,stim_v=t,v; st=WE
            elif v in END: st=IDLE
        elif st==WE:
            if v in END:
                keep=(v==201) or (not correct_only)        # 201=correct; 200=incorrect
                if keep and t-stim_t>=WIN_SEC: res.append((stim_t,stim_v))
                st=IDLE
            elif v in CUE and v not in AMB: stim_t=None; st=WS
            elif v in AMB: stim_t=None; st=WS
            elif v in STIM: stim_t,stim_v=t,v
    return res

def pick32(raw):
    up={c.upper():c for c in raw.ch_names}
    picks=[up[c.upper()] for c in CANON if c.upper() in up]
    if not picks: return None
    raw.pick(picks)
    ren={up[c.upper()]:c for c in CANON if c.upper() in up and up[c.upper()]!=c}
    if ren: raw.rename_channels(ren)
    present=[c for c in CANON if c in raw.ch_names]; raw.reorder_channels(present); return present

def segment_all(raw_dir, no_filter, correct_only=True):
    """-> {sid:{'label':lab,'X':[(32,200)uV...],'stim':[...]}} across all directions."""
    montage=mne.channels.make_standard_montage('standard_1005'); out={}; nfiles=0
    for grp,lab in (('HC',0),('MCI',1)):
        gdir=os.path.join(raw_dir,grp)
        if not os.path.isdir(gdir): print(f"  missing {gdir}"); continue
        def getsid(fn): m=re.search(r'sub(\d+)',fn); return int(m.group(1)) if m else None
        files=sorted(f for f in os.listdir(gdir) if getsid(f) is not None and f.endswith('.set'))
        for f in files:
            sid=getsid(f)
            try:
                raw=mne.io.read_raw_eeglab(os.path.join(gdir,f),preload=True,verbose=False)
                present=pick32(raw)
                if present is None or len(present)<28: continue
                if not no_filter:
                    raw.notch_filter(60,verbose=False)
                    raw.filter(0.1,60,method='fir',fir_window='hamming',verbose=False)
                if raw.info['sfreq']!=SFREQ: raw.resample(SFREQ,verbose=False)
                if len(present)<32:                         # complete to 32 by interpolation
                    raw.set_montage(montage,on_missing='ignore',match_case=False)
                    full=mne.io.RawArray(np.zeros((32,raw.n_times)),mne.create_info(CANON,SFREQ,'eeg'),verbose=False)
                    full.set_montage(montage,on_missing='ignore',match_case=False)
                    for ch in present: full._data[CANON.index(ch)]=raw.get_data(picks=ch)[0]
                    full.info['bads']=[c for c in CANON if c not in present]
                    full.interpolate_bads(reset_bads=True,verbose=False); raw=full
                else:
                    raw.set_montage(montage,on_missing='ignore',match_case=False)
                data=raw.get_data(units='uV'); n=data.shape[1]
                for stim_t,stim_v in onsets(raw, correct_only):
                    if stim_v not in STIM: continue
                    s0=int(round(stim_t*SFREQ)); s1=s0+WIN
                    if s0<0 or s1>n: continue
                    seg=data[:,s0:s1]
                    if seg.shape[1]!=WIN or not np.isfinite(seg).all(): continue
                    out.setdefault(sid,{'label':lab,'X':[],'stim':[]})
                    out[sid]['X'].append(seg.astype(np.float32)); out[sid]['stim'].append(int(stim_v))
                nfiles+=1
            except Exception as e:
                print(f"  FAIL {f}: {type(e).__name__} {e}",flush=True)
    print(f"segmented {nfiles} file(s), {len(out)} subject(s)",flush=True)
    return out

if __name__ == '__main__':
    from s1_segment import segment_all_s1
    from isi_segment import segment_isi
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                    # repo root
    BP   = os.environ.get('BP_DIR', os.path.join(ROOT, 'data', 'preprocessed'))   # input: Step 2 output (0.53-30 Hz, 200 Hz)
    OUT  = os.path.join(ROOT, 'results', 'segments.pkl')                          # output: Step 4 input
    S2 = segment_all(BP, True, correct_only=True)
    S1 = segment_all_s1(BP, True, correct_only=True)
    SI = segment_isi(BP)
    out = {}
    for sid in sorted(S2):
        X = np.asarray(S2[sid]['X'], np.float32); st = np.asarray(S2[sid]['stim'])
        out[sid] = {'label': int(S2[sid]['label']),
                    'dir0': X[np.isin(st, list(DIRBINS['Dir0']))],
                    's1':  np.asarray(S1[sid]['X'], np.float32) if sid in S1 else None,
                    'isi': np.asarray(SI[sid]['X'], np.float32) if sid in SI else None}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pickle.dump(out, open(OUT, 'wb'), protocol=4)
    print(len(out), "subjects saved ->", OUT)
