import sys; sys.path.insert(0,'research')
from common import *
import itertools
def ctxmodel(qs,seqs,lags,mode,hl,strength,prior):
    out={}
    for r in qs.itertuples():
        s=seqs[r.machine_id]; c=r.cutoff
        f=(lambda v: v) if mode=='5' else (lambda v:int(v>0)) if mode=='w' else (lambda v:min(v,2))
        def ctx(t):
            vs=[s.get(t-l) for l in lags]
            return None if None in vs else tuple(f(v) for v in vs)
        want=ctx(c); cnt=np.array(prior[r.machine_type])*strength
        allc=np.zeros(5)
        for t,v in s.items():
            if t>=c: continue
            w=0.5**((c-1-t)/hl) if hl else 1
            allc[v]+=w
            if want is not None and ctx(t)==want: cnt[v]+=w
        out[r.query_id]=cnt/cnt.sum()
    return pd.DataFrame.from_dict(out,orient='index',columns=PC)
TQ,TS=load('train')
prior={t:np.bincount(g.outcome.map(P.index),minlength=5)/len(g) for t,g in TQ.groupby('machine_type')}
for lags in [(1,),(2,),(4,),(7,),(2,4),(2,7),(4,7),(1,2),(2,4,7)]:
  for mode in ['w','3']:
    for st in [2]:
        p=ctxmodel(VQ,VS,lags,mode,None,st,prior); print(lags,mode,st,round(score(p)[0],2))
