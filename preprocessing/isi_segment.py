"""ISI segmenter.
ISI = the 1 s immediately before S2 onset, over all correct trials, pooled over directions
"""
import os, re, glob, numpy as np, mne, warnings
warnings.filterwarnings("ignore"); mne.set_log_level("ERROR")
from Step3_Segmentation import parse_val, pick32, CUE, STIM, SFREQ, WIN
MINGAP = 4.0   # minimum S1->S2 gap (s) for a trial's ISI window to be kept

def segment_isi(raw_dir):
    out={}; dropped=0; kept=0
    for grp,lab in (('HC',0),('MCI',1)):
        for f in sorted(glob.glob(os.path.join(raw_dir,grp,'*.set'))):
            m=re.search(r'sub(\d+)',os.path.basename(f))
            if not m: continue
            sid=int(m.group(1)); raw=mne.io.read_raw_eeglab(f,preload=True,verbose=False); pres=pick32(raw)
            assert pres is not None and len(pres)==32 and raw.info['sfreq']==SFREQ
            data=raw.get_data(units='uV'); n=data.shape[1]; ev=[]
            for a in raw.annotations:
                try: ev.append((float(a['onset']),parse_val(a['description'])))
                except: pass
            ev.sort()
            for i in range(len(ev)-2):
                (t0,v0),(t1,v1),(t2,v2)=ev[i],ev[i+1],ev[i+2]
                if v0 in CUE and v1 in STIM and v2==201:
                    if t1-t0<MINGAP: dropped+=1; continue
                    b=int(round(t1*SFREQ)); a=b-WIN
                    if a<0 or b>n: continue
                    seg=data[:,a:b]
                    if seg.shape[1]==WIN and np.isfinite(seg).all():
                        out.setdefault(sid,{'label':lab,'X':[]})['X'].append(seg.astype(np.float32)); kept+=1
    print(f"ISI segmented: {len(out)} subjects, {kept} trials kept, {dropped} dropped (S1->S2 gap < {MINGAP} s)",flush=True)
    return out
