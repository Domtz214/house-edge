import sys, os; sys.path.insert(0,'research'); sys.path.insert(0,'my-entry-v6')
import features as F, train_v6 as T6, numpy as np, lightgbm as lgb, pandas as pd
from common import score, PC, VQ
from train_v5 import WIN_PARAMS, SIZE_PARAMS
tr=T6.pseudo(F,'public/train'); vq,vf=T6.query_feats(F,'public/validation')
ids=[q['query_id'] for q in vq]; Vw=np.array([vf[i][0] for i in ids]); Vs=np.array([vf[i][1] for i in ids])
size=lgb.LGBMClassifier(**SIZE_PARAMS).fit(tr[2],tr[3]); SZ=size.predict_proba(Vs)
def sc(pw,sz=SZ):
    return round(score(pd.DataFrame(np.c_[1-pw,pw[:,None]*sz],index=ids,columns=PC))[0],3)
for kw in [dict(),dict(n_estimators=800,learning_rate=0.015),dict(num_leaves=63,min_child_samples=50),dict(num_leaves=15,min_child_samples=300),dict(n_estimators=1200,learning_rate=0.01,num_leaves=31,min_child_samples=200)]:
    p=dict(WIN_PARAMS); p.update(kw); m=lgb.LGBMClassifier(**p).fit(tr[0],tr[1]); print('win',kw,sc(m.predict_proba(Vw)[:,1]))
base=lgb.LGBMClassifier(**WIN_PARAMS).fit(tr[0],tr[1]).predict_proba(Vw)[:,1]
for kw in [dict(n_estimators=600,learning_rate=0.02),dict(num_leaves=31,min_child_samples=100)]:
    p=dict(SIZE_PARAMS); p.update(kw); m=lgb.LGBMClassifier(**p).fit(tr[2],tr[3]); print('size',kw,sc(base,m.predict_proba(Vs)))
np.save('research/cache_base_pw.npy',base)
