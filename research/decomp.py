import sys; sys.path.insert(0,'research')
from pattern import *
p=predict(VQ,VS,hl=20,a=0.1,size_prior=SP)
t=T.loc[p.index]
# oracle win, our sizes
pw=1-t.p_0.values; ours_sz=p[PC[1:]].values/ p[PC[1:]].sum(1).values[:,None]
o1=pd.DataFrame(np.c_[1-pw,pw[:,None]*ours_sz],index=p.index,columns=PC); print('oracle win, our size',score(o1)[0])
tsz=t[PC[1:]].values/t[PC[1:]].sum(1).values[:,None]; pw2=1-p.p_0.values
o2=pd.DataFrame(np.c_[1-pw2,pw2[:,None]*tsz],index=p.index,columns=PC); print('our win, oracle size',score(o2)[0])
# size profiles in targets
d=VQ.merge(pd.DataFrame(tsz,index=p.index,columns=['s1','s2','s5','s20']),left_on='query_id',right_index=True)
for ty,g in d.groupby('machine_type'): print(ty); print(g[['s1','s2','s5','s20']].round(2).value_counts().head(8))
