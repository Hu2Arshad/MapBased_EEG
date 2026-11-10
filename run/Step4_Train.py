"""HC vs MCI with EEGNet

Input: segmented trials, a pickle {sid: {'label': 0 HC / 1 MCI,
         'dir0': (n,32,200) uV  Dir0 S2 trials 
         's1':   (n,32,200) uV  S1 trials 
         'isi':  (n,32,200) uV  ISI trials 
Pipeline:
  1. supertrials: per subject and view, up to 500 unique random combinations of k=10 trials, averaged.
  2. subjects: >= 100 supertrials in Dir0, S1 and ISI, minus --exclude
  3. model inputs:  dir0  = Dir0 supertrials                      (baseline)
                    cs1   = Dir0 supertrials - subject's mean S1 supertrial
                    cisi  = Dir0 supertrials - subject's mean ISI supertrial
                    s1isi = S1 supertrials   - subject's mean ISI supertrial
                    s1raw / isiraw = raw S1 / raw ISI supertrials (optional, --models)
  4. 5x5 CV over subjects (StratifiedKFold, random_state split_seed+rep). Per fold and model, EEGNet (F1=8, D=2, dropout 0.5)
     trained 50 epochs: AdamW lr 1e-3 wd 1e-2, batch 64, grad clip 1.0, BCE weighted 1/(supertrials of the subject) x class
     weight, seed 0 (--seed drives supertrial sampling, the CV split, and EEGNet init/shuffle). A test subject's probability = mean sigmoid over its supertrials.
  5. readout: probabilities at 20 epochs; fusion = mean of the models' probabilities; BACC (threshold 0.5) and
     ROC-AUC per test fold, mean +/- std over the 25 folds
Best result (74 subjects): three contrasts cs1+cisi+s1isi = 72.0 +/- 10.4 BACC

Usage:  python train_best_bacc.py [--data results/segments_dir0_s1_isi.pkl] [--models dir0,cs1,cisi,s1isi,s1raw,isiraw]
        [--folds 0,1] (subset of folds, for a quick check) [--out results/train_best_bacc_preds.csv]

"""
import os, sys, subprocess
if os.environ.get('PYTHONHASHSEED')!='0':              
    sys.exit(subprocess.call([sys.executable]+sys.argv, env=dict(os.environ, PYTHONHASHSEED='0')))
import csv, pickle, random, argparse, itertools, warnings
from collections import Counter, defaultdict
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy import stats
warnings.filterwarnings("ignore")

K=10; REJECT=80.0; NMAX=500; NMIN=100; EPOCHS=20; SAVE_EPOCHS=[20]; AVG_EPOCHS=[20] 

# ----------------------------------------------------------------------------------------------------------- model
class EEGNet(nn.Module):
    def __init__(self, num_temporal_filts=8, num_spatial_filts=2, num_chans=32, window_length=200, p_dropout=0.5,
                 avgpool_factor=4, num_classes=1):
        super().__init__()
        F1, D, C, T = num_temporal_filts, num_spatial_filts, num_chans, window_length; F2 = D*F1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 9), padding='same'),
            nn.Conv2d(F1, D*F1, (C, 1), groups=F1),
            nn.ELU(),
            nn.AvgPool2d((1, avgpool_factor)),
            nn.Dropout(p_dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 2*(T//(avgpool_factor*2))+1), groups=F2, padding='same'),
            nn.Conv2d(F2, F2, (1, 1)),
            nn.ELU(),
            nn.Flatten(start_dim=1),
            nn.Linear((T//avgpool_factor)*F2, num_classes))
    def forward(self, x):                                  # x: (batch, 200 time, 32 channels)
        x = x.swapaxes(1, 2).unsqueeze(1)
        return self.block2(self.block1(x)).squeeze(1)

def setup_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ----------------------------------------------------------------------------------------------------------- supertrials
def supertrials_k(X, seed, k=K, reject=REJECT, nmax=NMAX, nmin=NMIN):
    """X (n,32,200) uV -> up to nmax unique k-trial averages kept if max|avg| <= reject uV; /100 -> (N,200,32). None if < nmin."""
    rng=np.random.default_rng(seed); n=len(X)
    if n<k: return None
    keep=[]; seen=set(); att=0; cap=200*nmax+4000
    while len(keep)<nmax and att<cap:
        att+=1; idx=tuple(sorted(rng.choice(n,size=k,replace=False).tolist()))
        if idx in seen: continue
        seen.add(idx); z=X[list(idx)].mean(0)
        if np.abs(z).max()<=reject: keep.append(z)
    if len(keep)<nmin: return None
    return np.transpose(np.stack(keep),(0,2,1)).astype(np.float32)/100.0

# ----------------------------------------------------------------------------------------------------------- training
@torch.no_grad()
def subj_probs(model, dd, S, dev):
    model.eval(); out={}
    for s in S:
        X=torch.tensor(dd[s],dtype=torch.float32,device=dev); z=[]
        for i in range(0,len(X),1024): z.append(model(X[i:i+1024]).reshape(-1))
        out[s]=float(torch.sigmoid(torch.cat(z)).mean())
    return out

def train_preds(dd, trS, teS, lab, dev, seed=0):
    """Train one EEGNet on the training subjects' samples; return {epoch: {test subject: probability}}."""
    Xtr=[];ytr=[];sid=[]
    for s in trS:
        for i in range(len(dd[s])): Xtr.append(dd[s][i]); ytr.append(lab[s]); sid.append(int(s))
    Xtr=np.stack(Xtr); ytr=np.array(ytr,np.float32); sid=np.array(sid)
    cc=np.bincount(ytr.astype(int),minlength=2).astype(float); clsw=cc.sum()/(2*np.maximum(cc,1))
    scn=Counter(sid.tolist()); w=np.array([(1.0/scn[sid[j]])*clsw[int(ytr[j])] for j in range(len(ytr))]); w=w/w.mean()
    setup_seed(seed); model=EEGNet(8,2,32,200,0.5,4,1).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2); bce=nn.BCEWithLogitsLoss(reduction='none')
    Xt=torch.tensor(Xtr,device=dev); yt=torch.tensor(ytr,device=dev); wt=torch.tensor(w,dtype=torch.float32,device=dev); n=len(Xt); out={}
    for ep in range(1,EPOCHS+1):
        model.train(); perm=torch.randperm(n,device=dev)
        for i in range(0,n,64):
            idx=perm[i:i+64]; opt.zero_grad(); (bce(model(Xt[idx]).reshape(-1),yt[idx])*wt[idx]).mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if ep in SAVE_EPOCHS: out[ep]=subj_probs(model,dd,teS,dev)
    del model,Xt; torch.cuda.empty_cache(); return out

def score(y, p): return balanced_accuracy_score(y,[int(x>=.5) for x in p])*100, roc_auc_score(y,p)*100

# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='results/segments.pkl'); ap.add_argument('--exclude',default='39')
    ap.add_argument('--models',default='dir0,cs1,cisi,s1isi'); ap.add_argument('--split_seed',type=int,default=0)
    ap.add_argument('--folds',default=''); ap.add_argument('--out',default='results/train_best_bacc_preds.csv')
    ap.add_argument('--seed',type=int,default=0)           # seed for supertrial sampling, CV split, EEGNet init / shuffle
    ap.add_argument('--n_folds',type=int,default=5)        # folds per repeat (5 repeats); 10 -> 90% of the subjects train each model
    a=ap.parse_args(); MODELS=a.models.split(','); EXC={int(x) for x in a.exclude.split(',') if x}
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device {dev} | models {MODELS} | 5x{a.n_folds}-fold | epoch {EPOCHS}",flush=True)

    # 1-2. supertrials and subject selection
    seg=pickle.load(open(a.data,'rb'))
    X0={}; X1={}; XI={}; lab={}
    for sid in sorted(seg):
        d=seg[sid]
        if d['s1'] is None or d['isi'] is None: continue
        if sid in EXC: continue
        s0=supertrials_k(d['dir0'],seed=hash(('Dir0',sid,K,a.seed))%(2**31))
        s1=supertrials_k(d['s1'],seed=hash(('S1',sid,K,a.seed))%(2**31))
        si=supertrials_k(d['isi'],seed=hash(('ISI',sid,K,a.seed))%(2**31))
        if s0 is None or s1 is None or si is None: continue            # < 100 clean supertrials in a view
        X0[sid]=s0; X1[sid]=s1; XI[sid]=si; lab[sid]=d['label']
    subs=np.array(sorted(lab)); y=np.array([lab[s] for s in subs])
    print(f"{len(subs)} subjects (HC {int((y==0).sum())} MCI {int((y==1).sum())})",flush=True)

    # 3. model inputs
    M1={s:X1[s].mean(0,keepdims=True) for s in subs}; MI={s:XI[s].mean(0,keepdims=True) for s in subs}
    def data(m, S):
        if m=='dir0':   return {s:X0[s] for s in S}
        if m=='cs1':    return {s:(X0[s]-M1[s]).astype(np.float32) for s in S}
        if m=='cisi':   return {s:(X0[s]-MI[s]).astype(np.float32) for s in S}
        if m=='s1isi':  return {s:(X1[s]-MI[s]).astype(np.float32) for s in S}
        if m=='s1raw':  return {s:X1[s] for s in S}
        if m=='isiraw': return {s:XI[s] for s in S}
        raise ValueError(m)

    # 4. 5x5 cross-validation
    SPLITS=[]
    for rep in range(5): SPLITS+=list(StratifiedKFold(a.n_folds,shuffle=True,random_state=a.split_seed+rep).split(subs,y))
    run=[int(f) for f in a.folds.split(',')] if a.folds else list(range(len(SPLITS)))
    PP=defaultdict(dict); rows=[]
    for fi in run:
        tr,te=SPLITS[fi]; trS=list(subs[tr]); teS=list(subs[te])
        for m in MODELS:
            dd=data(m,trS+teS); PP[fi][m]=train_preds(dd,trS,teS,lab,dev,a.seed); del dd
            for e in SAVE_EPOCHS:
                for s in teS: rows.append([m,fi,e,int(s),lab[s],f"{PP[fi][m][e][s]:.6f}"])
        with open(a.out,'w',newline='') as fh:
            wr=csv.writer(fh); wr.writerow(['model','fold','epoch','sid','label','p']); wr.writerows(rows)

    # 5. readout: every model and every fusion (epoch 20)
    def fold_scores(ms, eps):
        R=[]
        for fi in run:
            teS=list(subs[SPLITS[fi][1]])
            R.append(score([lab[s] for s in teS],[np.mean([np.mean([PP[fi][m][e][s] for e in eps]) for m in ms]) for s in teS]))
        return np.array(R)
    combos=[list(c) for r in range(1,len(MODELS)+1) for c in itertools.combinations(MODELS,r)]
    for c in sorted(combos,key=lambda c:-fold_scores(c,AVG_EPOCHS)[:,0].mean()):
        R=fold_scores(c,AVG_EPOCHS)
        print(f"   {'+'.join(c)}:  BACC = {R[:,0].mean():.1f} +/- {R[:,0].std():.1f}   ROC = {R[:,1].mean():.1f} +/- {R[:,1].std():.1f}",flush=True)
    print(f"saved -> {a.out}",flush=True)

if __name__=='__main__': main()
