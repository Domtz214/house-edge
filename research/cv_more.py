import sys; sys.path.insert(0,'research'); sys.path.insert(0,'my-entry-v6')
import features as F, train_v6 as T6, numpy as np, lightgbm as lgb, pandas as pd
from common import score, PC, VQ
from train_v5 import WIN_PARAMS, SIZE_PARAMS
tr=T6.pseudo(F,'public/train')
queries,eps=T6.episodes(F,'public/validation')
epl=sorted(eps); rng=np.random.RandomState(0); rng.shuffle(epl); folds=[set(epl[::2]),set(epl[1::2])]
def pseudo_eps(E,min_t=24):
    Xw,yw,Xs,ys=[],[],[],[]
    for e in E:
        ms=eps[e]; wanted={m:[t for t in sorted(v[0]) if min_t<=t<v[2]] for m,v in ms.items()}; wanted={m:c for m,c in wanted.items() if c}
        feats=F.episode_features({m:(v[0],v[1]) for m,v in ms.items()},wanted)
        for m,cuts in wanted.items():
            s=ms[m][0]
            for t in cuts:
                f,g=feats[(m,t)]; Xw.append(f); yw.append(int(s[t]>0))
                if s[t]>0: Xs.append(g); ys.append(s[t]-1)
    return np.array(Xw),np.array(yw),np.array(Xs),np.array(ys)
vq,vf=T6.query_feats(F,'public/validation')
preds_a={};preds_b={}
for k in range(2):
    hold=folds[k]; extra=pseudo_eps(folds[1-k])
    for tag,data,store in [('train only',tr,preds_a),('train+other half',tuple(np.concatenate([a,b]) if a.ndim==1 else np.vstack([a,b]) for a,b in zip(tr,extra)),preds_b)]:
        w=lgb.LGBMClassifier(**WIN_PARAMS).fit(data[0],data[1]); s=lgb.LGBMClassifier(**SIZE_PARAMS).fit(data[2],data[3])
        for q in vq:
            if q['episode_id'] in hold:
                fw,g=vf[q['query_id']]; pw=w.predict_proba(np.array([fw]))[0,1]; sz=s.predict_proba(np.array([g]))[0]
                store[q['query_id']]=np.r_[1-pw,pw*sz]
for name,st in [('train only',preds_a),('train + other val half',preds_b)]:
    print(name, score(pd.DataFrame.from_dict(st,orient='index',columns=PC)))
