import sys; sys.path.insert(0,'research')
from common import *
from pairs import condH, pairs as ALLP
from collections import Counter
p=pd.read_csv('v6-report/validation_predictions.csv').set_index('query_id')
d=VQ.set_index('query_id').join(p.p_0.rename('pp0')).join(T.p_0.rename('tp0'))
d['err']=(d.pp0-d.tp0).abs()
big=d[d.err>0.25]; small=d[d.err<0.05]
print(len(big),'big errors by condition',big.condition.value_counts().to_dict())
def bestp(r,W=60): return min(ALLP,key=lambda p:condH(VS[r.machine_id],r.cutoff,p,start=r.cutoff-W)[0])
for name,g in [('big',big),('small',small.sample(60,random_state=0))]:
    c=Counter(bestp(r) for r in g.itertuples()); print(name,c.most_common(6))
print(big[['machine_type','condition','cutoff','pp0','tp0']].head(15))
