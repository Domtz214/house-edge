import sys; sys.path.insert(0,'research')
from common import *
from math import lgamma
PAIRS=[(1,2),(1,3),(2,5),(4,7)]
# two size profiles per type (cond. on win), from validation targets (organizer-released)
PROF={'balanced':[[0.29543605,0.49013145,0.18379929,0.03063322],[0.49985311,0.34792827,0.1304731,0.02174552]],
 'jackpot':[[0.14496259,0.31501378,0.40501772,0.13500591],[0.33514603,0.2449462,0.31493083,0.10497694]],
 'steady':[[0.46176351,0.4305892,0.10549435,0.00215295],[0.75398939,0.19680849,0.04821808,0.00098404]]}
LP={k:np.log(np.array(v)) for k,v in PROF.items()}
def lbeta(a,b): return lgamma(a)+lgamma(b)-lgamma(a+b)
def win_pred(s,c,ticks,hl,a,pairs,temp):
    lls=[];preds=[]
    for p in pairs:
        N=defaultdict(lambda: np.zeros(2))
        for t in ticks:
            ctx=tuple(s.get(t-l) for l in p)
            if None in ctx: continue
            N[tuple(int(x>0) for x in ctx)][int(s[t]>0)]+=0.5**((c-1-t)/hl)
        lls.append(sum(lbeta(n[1]+a,n[0]+a)-lbeta(a,a) for n in N.values()))
        cur=tuple(s.get(c-l) for l in p)
        if None in cur: preds.append(None)
        else:
            n=N[tuple(int(x>0) for x in cur)]; preds.append((n[1]+a)/(n.sum()+2*a))
    lls=np.array(lls)*temp; w=np.exp(lls-lls.max())
    ok=[i for i,x in enumerate(preds) if x is not None]
    return (sum(w[i]*preds[i] for i in ok)/sum(w[i] for i in ok)) if ok else 0.45
def predict(qs,seqs,hls=(20,),a=0.1,temp=1.0,size_hl=40,size_temp=1.0,pairs=PAIRS):
    out={}
    for r in qs.itertuples():
        s=seqs[r.machine_id]; c=r.cutoff; ticks=[t for t in s if t<c]
        pwin=np.mean([win_pred(s,c,ticks,hl,a,pairs,temp) for hl in hls])
        ll=np.zeros(2)
        for t in ticks:
            if s[t]>0: ll+=0.5**((c-1-t)/size_hl)*LP[r.machine_type][:,s[t]-1]
        ll*=size_temp; post=np.exp(ll-ll.max()); post/=post.sum()
        sz=post@np.array(PROF[r.machine_type])
        out[r.query_id]=np.r_[1-pwin,pwin*sz]
    return pd.DataFrame.from_dict(out,orient='index',columns=PC)
if __name__=='__main__':
    for shl in [20,40,80]:
        for st in [0.5,1]:
            print('size',shl,st,round(score(predict(VQ,VS,size_hl=shl,size_temp=st))[0],2))
    for hls in [(10,),(15,),(10,20),(10,20,40),(8,15,30)]:
        print('hls',hls,round(score(predict(VQ,VS,hls=hls,size_hl=40))[0],2))
