import sys; sys.path.insert(0,'research')
from common import *
from stackfeat import machine_feats, TYPES
import lightgbm as lgb, time
def pseudo(split,qs,seqs,min_t=24,every=1):
    X=[];y=[];ty=[]
    for r in qs.itertuples():
        s=seqs[r.machine_id]
        cuts=[t for t in sorted(s) if t>=min_t and t<r.cutoff][::every]
        if not cuts: continue
        F=machine_feats(s,cuts)
        X.append(F); y+= [int(s[t]>0) for t in cuts]; ty+=[TYPES.index(r.machine_type)]*len(cuts)
    X=np.vstack(X); return np.c_[X,ty],np.array(y)
def qfeats(qs,seqs):
    X=[]
    for r in qs.itertuples():
        X.append(np.r_[machine_feats(seqs[r.machine_id],[r.cutoff])[0],TYPES.index(r.machine_type)])
    return np.array(X)
if __name__=='__main__':
    t0=time.time()
    TQ,TS=load('train')
    Xtr,ytr=pseudo('train',TQ,TS); print('train pseudo',Xtr.shape,time.time()-t0)
    Xva=qfeats(VQ,VS)
    np.save('research/Xtr.npy',Xtr); np.save('research/ytr.npy',ytr); np.save('research/Xva.npy',Xva)
