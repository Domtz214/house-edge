import sys; sys.path.insert(0,'research')
import v4
from v4 import *
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')
LAGS=[1,2,3,4,5,7]
def feats(s,t):
    v={l:s.get(t-l) for l in LAGS}
    if any(x is None for x in v.values()): return None
    x={l:2*int(v[l]>0)-1 for l in LAGS}
    return [x[l] for l in LAGS]+[x[a]*x[b] for a,b in PAIRS]
def lr_pred(s,c,ticks,hl,C):
    X=[];y=[];w=[]
    for t in ticks:
        f=feats(s,t)
        if f is None: continue
        X.append(f); y.append(int(s[t]>0)); w.append(0.5**((c-1-t)/hl))
    cur=feats(s,c)
    if cur is None or len(set(y))<2: return None
    m=LogisticRegression(C=C,max_iter=500).fit(X,y,sample_weight=np.array(w)/np.mean(w))
    return m.predict_proba([cur])[0,1]
base=v4.win_pred
for hl in [15,25]:
  for C in [0.3,1,3]:
    def wp(s,c,t,h,a,pr,tp,hl=hl,C=C):
        r=lr_pred(s,c,t,hl,C); return base(s,c,t,15,0.1,pr,tp) if r is None else r
    v4.win_pred=wp
    print(hl,C,score(v4.predict(VQ,VS,hls=(15,),size_hl=40,size_temp=0.5)))
