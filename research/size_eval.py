import sys; sys.path.insert(0,'research')
from v4 import *
from sizefeat import size_feats, TYPES
import lightgbm as lgb
TQ,TS=load('train')
X=[];y=[]
for r in TQ.itertuples():
    s=TS[r.machine_id]; cuts=[t for t in sorted(s) if t>=24 and t<r.cutoff and s[t]>0]
    if not cuts: continue
    F=size_feats(s,cuts,r.machine_type); X.append(np.c_[F,np.full(len(cuts),TYPES.index(r.machine_type))]); y+=[s[t]-1 for t in cuts]
X=np.vstack(X); y=np.array(y); print(X.shape)
Xv=np.array([np.r_[size_feats(VS[r.machine_id],[r.cutoff],r.machine_type)[0],TYPES.index(r.machine_type)] for r in VQ.itertuples()])
m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,min_child_samples=200,verbose=-1).fit(X,y)
szp=m.predict_proba(Xv)
pw=np.load('research/pw_val.npy')[:,0]
base=predict(VQ,VS,hls=(15,),size_hl=40,size_temp=0.5)
sz0=base[PC[1:]].values/base[PC[1:]].values.sum(1,keepdims=True)
def mk(pw,sz): return pd.DataFrame(np.c_[1-pw,pw[:,None]*sz],index=base.index,columns=PC)
print('lgb win + posterior size',round(score(mk(pw,sz0))[0],2))
print('lgb win + lgb size',round(score(mk(pw,szp))[0],2))
print('lgb win + avg size',round(score(mk(pw,(szp+sz0)/2))[0],2))
tsz=T.loc[base.index][PC[1:]].values; tsz=tsz/tsz.sum(1,keepdims=True)
print('lgb win + ORACLE size',round(score(mk(pw,tsz))[0],2))
np.save('research/sz_val.npy',szp)
