import pandas as pd, numpy as np, itertools, math
from collections import Counter, defaultdict
D='public/'
h=pd.concat([pd.read_csv(D+s+'/history.csv') for s in ['train','validation']])
q=pd.concat([pd.read_csv(D+s+'/queries.csv') for s in ['train','validation']])
cond=pd.concat([pd.read_csv(D+s+'/conditions.csv') for s in ['train','validation']])
q=q.merge(cond,on='query_id')
P=[0,1,2,5,20]
def cat(v,mode):
    if mode=='win': return int(v>0)
    if mode=='c3': return 0 if v==0 else 1 if v==1 else 2
    return P.index(v)
def H(c):
    n=sum(c.values()); return -sum(v/n*math.log(v/n) for v in c.values() if v)
def mi(seq,lag,mode,cut):
    joint=defaultdict(Counter); tot=Counter()
    for t,v in seq.items():
        if t>=cut: continue
        u=seq.get(t-lag)
        if u is None: continue
        joint[cat(u,mode)][P.index(v)]+=1; tot[P.index(v)]+=1
    n=sum(tot.values())
    return H(tot)-sum(sum(c.values())/n*H(c) for c in joint.values()), n
res=[]
g=h.groupby('machine_id')
for _,r in q.sample(120,random_state=0).iterrows():
    seq=dict(zip(g.get_group(r.machine_id).tick,g.get_group(r.machine_id).payout))
    mis=[mi(seq,l,'c3',r.cutoff)[0] for l in range(1,9)]
    best=int(np.argmax(mis))+1
    res.append((r.machine_type,r.condition,best,round(max(mis),3),round(sorted(mis)[-2],3),len(seq)))
df=pd.DataFrame(res,columns=['type','cond','bestlag','mi1','mi2','n'])
print(df.groupby('type')[['mi1','mi2']].mean()); print(df.bestlag.value_counts()); print(df.head(30))
