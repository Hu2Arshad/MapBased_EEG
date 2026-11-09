"""Every combination report for a variant run: all non-empty subsets of {Dir0-S1, Dir0-ISI, S1-ISI} from the variant, and the
same subsets plus Dir0-Dir180 (original recipe, results/pairs_preds.csv, 73 subjects), each paired with the same combination
of the original models (results/allsubj_k10_preds.csv + the same Dir0-Dir180). Mean-probability fusion at 0.5, readout avg
e10-e50, fold-avg BACC / ROC +/- std over the folds that are finished for every model; exact-tie paired Wilcoxon over folds.
Rows without Dir0-Dir180 use all subjects of the variant run; rows with it use the subjects that have Dir0-Dir180.
If the variant file (or --vd180 file) has its own 'dir0-dir180' model (train_variants.py --with_d180 / --only_d180), the variant
side uses it; the original side always uses the original-recipe Dir0-Dir180.
Usage: python combo_report.py results/variant_tmask_preds.csv [--prefix b_] [--vd180 file] [--ref ...] [--d180 ...]"""
import csv, argparse, itertools, numpy as np
from fractions import Fraction
from collections import defaultdict
from scipy import stats
AV = [20]; C3 = ['cs1', 'cisi', 's1isi']; D = 'dir0-dir180'   # epoch 20 only
NAME = {'cs1': 'Dir0-S1', 'cisi': 'Dir0-ISI', 's1isi': 'S1-ISI', D: 'Dir0-Dir180'}

def load(fn, keep, rename=None, epochs=None):
    P = defaultdict(lambda: defaultdict(list)); lab = {}
    for r in csv.DictReader(open(fn)):
        m = (rename or {}).get(r['model'], r['model'])
        if m in keep and int(r['epoch']) in (epochs or AV):
            P[(m, int(r['fold']))][int(r['sid'])].append(float(r['p'])); lab[int(r['sid'])] = int(r['label'])
    return P, lab
def exact(P, lab, models, f, S):
    p = np.array([np.mean([np.mean(P[(m, f)][s]) for m in models]) for s in S]); y = np.array([lab[s] for s in S])
    npos = int(y.sum()); nneg = len(y) - npos; pr = p >= 0.5
    tp = int((pr & (y == 1)).sum()); tn = int((~pr & (y == 0)).sum())
    c2 = sum(2 * int(a > b) + int(a == b) for a in p[y == 1] for b in p[y == 0])
    return 50 * (Fraction(tp, npos) + Fraction(tn, nneg)), 100 * Fraction(c2, 2 * npos * nneg)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('variant'); ap.add_argument('--prefix', default='')
    ap.add_argument('--ref', default='results/allsubj_k10_preds.csv'); ap.add_argument('--d180', default='results/pairs_preds.csv')
    ap.add_argument('--vd180', default=None); ap.add_argument('--test_subset', default=''); ap.add_argument('--epochs', default='20'); a = ap.parse_args()   # report epoch 20 only
    for ep_ in [int(x) for x in a.epochs.split(',')]: report_epoch(a, [ep_])

def report_epoch(a, EP):                                   # user rule 2026-10-11: epoch 20 and epoch 30 reported separately, no epoch average
    TS = {int(x) for x in open(a.test_subset).read().replace(chr(10), ',').split(',') if x.strip()} if a.test_subset else None
    V, lab = load(a.variant, C3 + [D], {a.prefix + m: m for m in C3}, EP); O, l2 = load(a.ref, C3, None, EP); Dd, l3 = load(a.d180, [D], None, EP); lab.update(l2); lab.update(l3)
    if a.vd180: VD, l4 = load(a.vd180, [D], None, EP); lab.update(l4); V.update({k_: v_ for k_, v_ in VD.items()})
    folds = [f for f in range(25) if all(len(V[(m, f)]) for m in C3)]
    own = all(len(V[(D, f)]) for f in folds)
    print(f"EPOCH {EP[0]} ({len(folds)} folds)", flush=True)
    for with_d in (False, True):
        rng = range(0, 4) if with_d else range(1, 4)
        for k in rng:
            for comb in itertools.combinations(C3, k):
                RN = []
                for f in folds:
                    S = sorted(set.intersection(*[set(V[(m, f)]) for m in C3]) & ((set(Dd[(D, f)]) & (set(V[(D, f)]) if own else set(Dd[(D, f)]))) if with_d else set(V[('cs1', f)])))
                    if TS: S = [s for s in S if s in TS]
                    if len({lab[s] for s in S}) < 2: continue        # fold without one of the classes after the restriction
                    PV = {**{(m, f): V[(m, f)] for m in C3}, (D, f): (V[(D, f)] if own else Dd[(D, f)])}
                    ms = list(comb) + ([D] if with_d else [])
                    RN.append(exact(PV, lab, ms, f, S))
                if not RN: continue
                nb = np.array([float(x[0]) for x in RN]); na = np.array([float(x[1]) for x in RN])
                lbl = ' + '.join([NAME[m] for m in comb] + (['Dir0-Dir180'] if with_d else []))
                print(f"   {lbl}:  BACC = {nb.mean():.1f} +/- {nb.std():.1f}   ROC = {na.mean():.1f} +/- {na.std():.1f}", flush=True)

if __name__ == '__main__': main()
