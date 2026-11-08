import sys; sys.path.insert(0,'research')
from common import *
pd.set_option('display.width',250)
d=VQ.merge(T,left_on='query_id',right_index=True)
d['last8']=[ ''.join(str(VS[r.machine_id].get(r.cutoff-l,'.')) for l in range(1,9)) for r in d.itertuples()]
print(d[['machine_type','signal','condition','cutoff','last8']+PC].round(3).head(40).to_string())
print(d[PC].describe().round(3))
print(d.groupby(['machine_type'])[PC].mean().round(3))
print(d.groupby(['signal'])[PC].mean().round(3))
