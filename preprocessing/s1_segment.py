"""
Stage 1 segmenter
"""
import os, re, glob, numpy as np, mne, warnings
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
from Step3_Segmentation import parse_val, pick32, CANON, CUE, STIM, SFREQ, WIN

def segment_all_s1(raw_dir, no_filter, correct_only=True, win_start_ms=0, win_len_ms=1000, bandpass=None):
    montage=mne.channels.make_standard_montage('standard_1005'); out={}; nfiles=0
    off=int(win_start_ms/1000*SFREQ); L=int(win_len_ms/1000*SFREQ)
    for grp,lab in (('HC',0),('MCI',1)):
        gdir=os.path.join(raw_dir,grp)
        if not os.path.isdir(gdir): print(f"  missing {gdir}"); continue
        for f in sorted(glob.glob(os.path.join(gdir,'*.set'))):
            m=re.search(r'sub(\d+)',os.path.basename(f))
            if not m: continue
            sid=int(m.group(1))
            try:
                raw=mne.io.read_raw_eeglab(f,preload=True,verbose=False)
                present=pick32(raw)
                if present is None or len(present)<28: continue
                if not no_filter:
                    raw.notch_filter(60,verbose=False); raw.filter(0.1,60,method='fir',fir_window='hamming',verbose=False)
                if bandpass is not None:
                    raw.filter(bandpass[0],bandpass[1],method='fir',fir_window='hamming',verbose=False)
                if raw.info['sfreq']!=SFREQ: raw.resample(SFREQ,verbose=False)
                if len(present)<32:
                    raw.set_montage(montage,on_missing='ignore',match_case=False)
                    full=mne.io.RawArray(np.zeros((32,raw.n_times)),mne.create_info(CANON,SFREQ,'eeg'),verbose=False)
                    full.set_montage(montage,on_missing='ignore',match_case=False)
                    for ch in present: full._data[CANON.index(ch)]=raw.get_data(picks=ch)[0]
                    full.info['bads']=[c for c in CANON if c not in present]; full.interpolate_bads(reset_bads=True,verbose=False); raw=full
                else:
                    raw.set_montage(montage,on_missing='ignore',match_case=False)
                data=raw.get_data(units='uV'); n=data.shape[1]
                ev=[]
                for a in raw.annotations:
                    try: ev.append((float(a['onset']),parse_val(a['description'])))
                    except: pass
                ev.sort()
                for i in range(len(ev)-2):
                    (t0,v0),(t1,v1),(t2,v2)=ev[i],ev[i+1],ev[i+2]
                    if v0 in CUE and v1 in STIM and (v2==201 or not correct_only):   # t0 = cue/S1 onset; v1 = S2 code
                        s0=int(round(t0*SFREQ))+off; s1=s0+L
                        if s0<0 or s1>n: continue
                        seg=data[:,s0:s1]
                        if seg.shape[1]==L and np.isfinite(seg).all():
                            out.setdefault(sid,{'label':lab,'X':[],'stim':[]})
                            out[sid]['X'].append(seg.astype(np.float32)); out[sid]['stim'].append(int(v1))
                nfiles+=1
            except Exception as e:
                print(f"  FAIL {os.path.basename(f)}: {type(e).__name__} {e}",flush=True)
    print(f"S1 segmented {nfiles} file(s), {len(out)} subject(s)",flush=True); return out

def segment_all_s2_win(raw_dir, no_filter, correct_only=True, win_start_ms=-200, win_len_ms=1000, bandpass=None):
    montage=mne.channels.make_standard_montage('standard_1005'); out={}; nfiles=0
    off=int(round(win_start_ms/1000*SFREQ)); L=int(win_len_ms/1000*SFREQ)
    for grp,lab in (('HC',0),('MCI',1)):
        gdir=os.path.join(raw_dir,grp)
        if not os.path.isdir(gdir): print(f"  missing {gdir}"); continue
        for f in sorted(glob.glob(os.path.join(gdir,'*.set'))):
            m=re.search(r'sub(\d+)',os.path.basename(f))
            if not m: continue
            sid=int(m.group(1))
            try:
                raw=mne.io.read_raw_eeglab(f,preload=True,verbose=False)
                present=pick32(raw)
                if present is None or len(present)<28: continue
                if not no_filter:
                    raw.notch_filter(60,verbose=False); raw.filter(0.1,60,method='fir',fir_window='hamming',verbose=False)
                if bandpass is not None:
                    raw.filter(bandpass[0],bandpass[1],method='fir',fir_window='hamming',verbose=False)
                if raw.info['sfreq']!=SFREQ: raw.resample(SFREQ,verbose=False)
                if len(present)<32:
                    raw.set_montage(montage,on_missing='ignore',match_case=False)
                    full=mne.io.RawArray(np.zeros((32,raw.n_times)),mne.create_info(CANON,SFREQ,'eeg'),verbose=False)
                    full.set_montage(montage,on_missing='ignore',match_case=False)
                    for ch in present: full._data[CANON.index(ch)]=raw.get_data(picks=ch)[0]
                    full.info['bads']=[c for c in CANON if c not in present]; full.interpolate_bads(reset_bads=True,verbose=False); raw=full
                else:
                    raw.set_montage(montage,on_missing='ignore',match_case=False)
                data=raw.get_data(units='uV'); n=data.shape[1]
                ev=[]
                for a in raw.annotations:
                    try: ev.append((float(a['onset']),parse_val(a['description'])))
                    except: pass
                ev.sort()
                for i in range(len(ev)-2):
                    (t0,v0),(t1,v1),(t2,v2)=ev[i],ev[i+1],ev[i+2]
                    if v0 in CUE and v1 in STIM and (v2==201 or not correct_only):   # t1 = S2 onset; v1 = S2 code
                        s0=int(round(t1*SFREQ))+off; s1=s0+L
                        if s0<0 or s1>n: continue
                        seg=data[:,s0:s1]
                        if seg.shape[1]==L and np.isfinite(seg).all():
                            out.setdefault(sid,{'label':lab,'X':[],'stim':[]})
                            out[sid]['X'].append(seg.astype(np.float32)); out[sid]['stim'].append(int(v1))
                nfiles+=1
            except Exception as e:
                print(f"  FAIL {os.path.basename(f)}: {type(e).__name__} {e}",flush=True)
    print(f"S2 win[{win_start_ms},{win_start_ms+win_len_ms}]ms segmented {nfiles} file(s), {len(out)} subject(s)",flush=True); return out
