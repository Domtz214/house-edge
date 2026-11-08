import sys; sys.path.insert(0,'research')
from pairs import *
from collections import Counter
for W in [30,50]:
  best=Counter(); Hs=[]
  for r in d.itertuples():
    s=VS[r.machine_id]
    res=sorted((condH(s,r.cutoff,p,start=r.cutoff-W)[0],p) for p in pairs)
    best[res[0][1]]+=1; Hs.append(res[0][0])
  print(W,best.most_common(8),'median H',np.median(Hs))
