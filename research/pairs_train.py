import sys; sys.path.insert(0,'research')
from pairs import condH, pairs
from common import load
from collections import Counter
TQ,TS=load('train'); best=Counter()
for r in TQ.itertuples():
    s=TS[r.machine_id]; best[sorted((condH(s,r.cutoff,p,start=r.cutoff-30)[0],p) for p in pairs)[0][1]]+=1
print('train best pairs',best.most_common(6))
