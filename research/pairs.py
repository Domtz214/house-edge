import sys; sys.path.insert(0,'research')
from common import *
import itertools, math
def condH(seq,cut,lags,win=True,start=None):
    tab=defaultdict(lambda: np.zeros(2))
    for t,v in seq.items():
        if t>=cut or (start and t<start): continue
        ctx=tuple(seq.get(t-l) for l in lags)
        if None in ctx: continue
        ctx=tuple(int(x>0) for x in ctx)
        tab[ctx][int(v>0)]+=1
    n=sum(c.sum() for c in tab.values()); H=0
    for c in tab.values():
        s=c.sum(); H+=sum(-x/n*math.log(x/s) for x in c if x)
    return H/math.log(2), tab, n
d=VQ.merge(T,left_on='query_id',right_index=True)
pairs=list(itertools.combinations(range(1,9),2))
for r in d.head(16).itertuples():
    s=VS[r.machine_id]
    res=sorted((condH(s,r.cutoff,p)[0],p) for p in pairs)[:3]
    single=sorted((condH(s,r.cutoff,(l,))[0],l) for l in range(1,9))[:2]
    H,tab,n=condH(s,r.cutoff,res[0][1])
    want=tuple(int(s.get(r.cutoff-l,-1)>0) for l in res[0][1])
    print(r.machine_type,r.condition,'p_win=%.3f'%(1-r.p_0),'best pairs',[(round(a,2),b) for a,b in res],'single',[(round(a,2),b) for a,b in single],'ctx',want,{k:v.tolist() for k,v in tab.items()})
