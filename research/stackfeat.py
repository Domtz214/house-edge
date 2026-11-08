"""Vectorised pattern features for every cutoff of a machine (numpy)."""
import numpy as np
from math import lgamma
from scipy.special import gammaln
PAIRS=[(1,2),(1,3),(2,5),(4,7)]
HLS=[6,12,25,60]
A=0.1
TYPES=['steady','balanced','jackpot']
def machine_feats(seq, cutoffs):
    """seq: dict tick->payout_idx. returns array [len(cutoffs), F]."""
    T=max(max(seq),max(cutoffs))+2
    pres=np.zeros(T+9,bool); win=np.zeros(T+9)
    big=np.zeros((T+9,4))
    for t,v in seq.items():
        pres[t+8]=True; win[t+8]=float(v>0)
        if v>0: big[t+8,v-1]=1
    off=8
    feats_all=[]
    cut=np.array(cutoffs)
    out=[]
    for hl in HLS:
        d=0.5**(1/hl)
        # recency weighted win rate & counts: S[c]=sum_{t<c} d^(c-1-t) x_t
        def rec(x):
            S=np.zeros(T+1); acc=0.0
            for c in range(1,T+1):
                acc=acc*d+x[c-1+off]; S[c]=acc
            return S
        Sp=rec(pres.astype(float)); Sw=rec(win*pres)
        out.append(((Sw[cut]+0.5)/(Sp[cut]+1)))
        out.append(np.log1p(Sp[cut]))
        lls=[];pws=[];ns=[]
        for (i,j) in PAIRS:
            valid=np.zeros(T+9,bool); code=np.zeros(T+9,int)
            idx=np.arange(off,T+off)
            valid[idx]=pres[idx]&pres[idx-i]&pres[idx-j]
            code[idx]=2*win[idx-i].astype(int)+win[idx-j].astype(int)
            N1=np.zeros((4,T+1)); N0=np.zeros((4,T+1))
            for k in range(4):
                m=(valid&(code==k)&pres)
                N1[k]=rec(m*win); N0[k]=rec(m*(1-win))
            ll=(gammaln(N1+A)+gammaln(N0+A)-gammaln(N1+N0+2*A)-(2*gammaln(A)-gammaln(2*A))).sum(0)[cut]
            # current context
            ci=cut+off
            cv=pres[ci-i]&pres[ci-j]; cc=2*win[ci-i].astype(int)+win[ci-j].astype(int)
            n1=N1[cc,cut]; n0=N0[cc,cut]
            pw=np.where(cv,(n1+A)/(n1+n0+2*A),np.nan); n=np.where(cv,n1+n0,-1)
            lls.append(ll); pws.append(pw); ns.append(n)
        lls=np.array(lls); pws=np.array(pws); ns=np.array(ns)
        w=np.exp(lls-lls.max(0)); w=w/w.sum(0)
        ok=~np.isnan(pws)
        bma=np.where(ok.any(0),(np.nan_to_num(pws)*w*ok).sum(0)/np.maximum((w*ok).sum(0),1e-12),np.nan)
        out.append(bma)
        for k in range(4): out+= [pws[k], ns[k], w[k]]
        out.append(w.max(0))
    # size posterior features (two profiles) -> computed outside
    return np.array(out).T
