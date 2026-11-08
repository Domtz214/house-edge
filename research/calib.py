import sys; sys.path.insert(0,'research')
from v4 import *
p=predict(VQ,VS,hls=(15,),size_hl=40,size_temp=0.5)
d=pd.DataFrame({'pw':1-p.p_0,'tw':1-T.loc[p.index].p_0})
d['bin']=pd.cut(d.pw,[0,.05,.1,.2,.35,.5,.65,.8,.9,.95,1])
print(d.groupby('bin',observed=True).agg(n=('tw','size'),pw=('pw','mean'),tw=('tw','mean')).round(3))
print('target pwin distribution'); print(pd.cut(d.tw,[0,.05,.1,.2,.35,.5,.65,.8,.9,.95,1]).value_counts().sort_index())
