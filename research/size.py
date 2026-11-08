import sys; sys.path.insert(0,'research')
from pattern import *
t=T.copy(); tsz=t[PC[1:]].values/t[PC[1:]].values.sum(1,keepdims=True)
d=VQ.set_index('query_id').join(pd.DataFrame(tsz,index=t.index,columns=['s1','s2','s5','s20']))
d['profA']=(d.s1>d.groupby('machine_type').s1.transform('median')).astype(int)
print(pd.crosstab([d.machine_type,d.signal],d.profA))
print(pd.crosstab(d.condition,d.profA))
# same profile across episode?
print(d.groupby('episode_id').profA.agg(['mean']).value_counts().head())
print('corr of profA with win at lag k / big(>=2) at lag k')
for k in range(1,9):
    w=[int(VS[m].get(c-k,-1)>0) for m,c in zip(d.machine_id,d.cutoff)]
    b=[int(VS[m].get(c-k,-1)>=2) for m,c in zip(d.machine_id,d.cutoff)]
    print(k, round(np.corrcoef(w,d.profA)[0,1],3), round(np.corrcoef(b,d.profA)[0,1],3))
# likelihood-based inference from recent sizes
PROF={ty:[g[g.profA==i][['s1','s2','s5','s20']].mean().values for i in (0,1)] for ty,g in d.groupby('machine_type')}
print(PROF)
for hl in [5,10,20,40]:
    acc=[]
    for r in d.itertuples():
        s=VS[r.machine_id]; ll=np.zeros(2)
        for tk,v in s.items():
            if tk<r.cutoff and v>0:
                w=0.5**((r.cutoff-1-tk)/hl); ll+=w*np.log([PROF[r.machine_type][i][v-1] for i in (0,1)])
        acc.append(int(np.argmax(ll))==r.profA)
    print('hl',hl,'acc',np.mean(acc))
d['mi']=d.machine_id.str[-3:].astype(int)
eps={e:dict(zip(g.mi,g.machine_id)) for e,g in d.groupby('episode_id')}
best=[]
for off in range(1,8):
  for k in range(1,9):
    for f in ['win','big']:
      x=[]
      for r in d.itertuples():
        o=eps[r.episode_id].get((r.mi+off)%8); v=VS[o].get(r.cutoff-k,-1) if o else -1
        x.append(int(v>0) if f=='win' else int(v>=2))
      best.append((abs(np.corrcoef(x,d.profA)[0,1]),off,k,f))
print(sorted(best,reverse=True)[:6])
# same for win target: does neighbor explain residual of pwin?
