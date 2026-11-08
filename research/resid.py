import sys; sys.path.insert(0,'research')
from v4 import *
p=predict(VQ,VS,hls=(15,),size_hl=40,size_temp=0.5)
d=VQ.set_index('query_id').copy(); d['pw']=1-p.p_0; d['tw']=1-T.p_0
d['err']=(d.pw-d.tw).abs()
print(d.groupby('condition').err.describe().round(3))
print((d.err>0.4).groupby(d.condition).mean())
# is best pair constant within an episode? Use H over last 40 ticks
import itertools,math
def H(s,c,p,W=40):
    N=defaultdict(lambda: np.zeros(2))
    for t in s:
        if c-W<=t<c:
            ctx=tuple(s.get(t-l) for l in p)
            if None in ctx: continue
            N[tuple(int(x>0) for x in ctx)][int(s[t]>0)]+=1
    n=sum(x.sum() for x in N.values())
    return sum(-v/n*math.log2(v/x.sum()) for x in N.values() for v in x if v)
d['bp']=[min(PAIRS,key=lambda p:H(VS[r.machine_id],r.cutoff,p)) for r in d.itertuples()]
print(d.groupby('episode_id').bp.nunique().value_counts())
print(pd.crosstab(d.machine_type,d.bp))
print(pd.crosstab(d.machine_id.str[-3:],d.bp))
