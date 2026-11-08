import sys; sys.path.insert(0,'research')
from common import *
p=pd.read_csv('v5-report/validation_predictions.csv').set_index('query_id')[PC]
t=T.loc[p.index]
def mk(pw,sz): return pd.DataFrame(np.c_[1-pw,pw[:,None]*sz],index=p.index,columns=PC)
psz=p[PC[1:]].values/p[PC[1:]].values.sum(1,keepdims=True); tsz=t[PC[1:]].values/t[PC[1:]].values.sum(1,keepdims=True)
print('v5',round(score(p)[0],2)); print('oracle win',round(score(mk(1-t.p_0.values,psz))[0],2)); print('oracle size',round(score(mk(1-p.p_0.values,tsz))[0],2))
