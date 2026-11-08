import sys; sys.path.insert(0,'research')
from common import *
from math import lgamma
PAIRS=[(1,2),(1,3),(2,5),(4,7)]
TQ,TS=load('train')
def lbeta(a,b): return lgamma(a)+lgamma(b)-lgamma(a+b)
def predict(qs,seqs,hl=20,a=0.3,temp=1.0,pairs=PAIRS,size_hl=40,size_k=5,size_prior=None,ret_parts=False):
    out={}; parts={}
    for r in qs.itertuples():
        s=seqs[r.machine_id]; c=r.cutoff
        ticks=[t for t in s if t<c]
        lls=[];preds=[]
        for p in pairs:
            N=defaultdict(lambda: np.zeros(2))
            for t in ticks:
                ctx=tuple(s.get(t-l) for l in p)
                if None in ctx: continue
                w=0.5**((c-1-t)/hl)
                N[tuple(int(x>0) for x in ctx)][int(s[t]>0)]+=w
            ll=sum(lbeta(n[1]+a,n[0]+a)-lbeta(a,a) for n in N.values())
            cur=tuple(s.get(c-l) for l in p)
            if None in cur: pw=None
            else:
                n=N[tuple(int(x>0) for x in cur)]; pw=(n[1]+a)/(n.sum()+2*a)
            lls.append(ll); preds.append(pw)
        lls=np.array(lls)*temp; w=np.exp(lls-lls.max())
        ok=[i for i,x in enumerate(preds) if x is not None]
        if ok: pwin=sum(w[i]*preds[i] for i in ok)/sum(w[i] for i in ok)
        else: pwin=0.45
        # sizes
        sc=np.array(size_prior[r.machine_type])*size_k
        for t in ticks:
            if s[t]>0: sc[s[t]-1]+=0.5**((c-1-t)/size_hl)
        sz=sc/sc.sum()
        out[r.query_id]=np.r_[1-pwin,pwin*sz]
        parts[r.query_id]=(pwin,w/w.sum())
    df=pd.DataFrame.from_dict(out,orient='index',columns=PC)
    return (df,parts) if ret_parts else df
# size prior per type from all train histories
SP={}
for t,g in TQ.groupby('machine_type'):
    cnt=np.zeros(4)
    for m in g.machine_id:
        for v in TS[m].values():
            if v>0: cnt[v-1]+=1
    SP[t]=cnt/cnt.sum()
print(SP)
if __name__=='__main__':
    for hl in [10,20,40]:
        for a in [0.1,0.3,1]:
            p=predict(VQ,VS,hl=hl,a=a,size_prior=SP); print(hl,a,score(p))
