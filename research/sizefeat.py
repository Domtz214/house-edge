import numpy as np
from v4 import PROF
SHLS=[4,10,25,60,200]
TYPES=['steady','balanced','jackpot']
def size_feats(seq,cutoffs,mtype):
    T=max(max(seq),max(cutoffs))+1
    X=np.zeros((T,4))
    for t,v in seq.items():
        if v>0: X[t,v-1]=1
    LR=np.log(np.array(PROF[mtype][0]))-np.log(np.array(PROF[mtype][1]))
    cut=np.array(cutoffs); out=[]
    for hl in SHLS:
        d=0.5**(1/hl); S=np.zeros((T+1,4)); acc=np.zeros(4)
        for c in range(1,T+1):
            acc=acc*d+X[c-1]; S[c]=acc
        Sc=S[cut]; n=Sc.sum(1)
        out+= [ (Sc[:,k]+0.25)/(n+1) for k in range(4)]+[np.log1p(n), Sc@LR]
    return np.array(out).T
