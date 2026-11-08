import sys; sys.path.insert(0,'research')
import v4
from v4 import *
FUNCS={'win':lambda v:int(v>0),'c3':lambda v:min(v,2),'c5':lambda v:v,'big':lambda v:int(v>=2)}
def win_pred2(s,c,ticks,hl,a,pairs,temp,f,prior_w=None):
    lls=[];preds=[]
    for p in pairs:
        N=defaultdict(lambda: np.zeros(2))
        for t in ticks:
            ctx=tuple(s.get(t-l) for l in p)
            if None in ctx: continue
            N[tuple(f(x) for x in ctx)][int(s[t]>0)]+=0.5**((c-1-t)/hl)
        lls.append(sum(lbeta(n[1]+a,n[0]+a)-lbeta(a,a) for n in N.values()))
        cur=tuple(s.get(c-l) for l in p)
        if None in cur: preds.append(None)
        else:
            n=N[tuple(f(x) for x in cur)]; preds.append((n[1]+a)/(n.sum()+2*a))
    lls=np.array(lls)*temp; w=np.exp(lls-lls.max())
    ok=[i for i,x in enumerate(preds) if x is not None]
    return (sum(w[i]*preds[i] for i in ok)/sum(w[i] for i in ok)) if ok else 0.45
for name in ['c3','big','c5']:
    for a in [0.1,0.3]:
        v4.win_pred=lambda s,c,t,hl,aa,pr,tp,f=FUNCS[name],a=a: win_pred2(s,c,t,hl,a,pr,tp,f)
        print(name,a,round(score(v4.predict(VQ,VS,hls=(15,),size_hl=40,size_temp=0.5))[0],2))
