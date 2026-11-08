import sys; sys.path.insert(0,'research')
import v4
from v4 import *
import lightgbm as lgb, xgboost as xgb, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
Xtr=np.load('research/Xtr.npy'); ytr=np.load('research/ytr.npy'); Xva=np.load('research/Xva.npy')
base=v4.predict(VQ,VS,hls=(15,),size_hl=40,size_temp=0.5)
sz=base[PC[1:]].values/base[PC[1:]].values.sum(1,keepdims=True)
def mk(pw):
    return pd.DataFrame(np.c_[1-pw,pw[:,None]*sz],index=base.index,columns=PC)
print('BMA baseline',round(score(base)[0],2))
res={}
for name,model in [
  ('lgb',lgb.LGBMClassifier(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=100,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)),
  ('xgb',xgb.XGBClassifier(n_estimators=400,learning_rate=0.03,max_depth=6,min_child_weight=50,subsample=0.8,colsample_bytree=0.8,tree_method='hist')),
  ('rf',RandomForestClassifier(n_estimators=300,min_samples_leaf=50,max_features=0.3,n_jobs=-1)),
]:
    Xf=np.nan_to_num(Xtr,nan=-1); Xv=np.nan_to_num(Xva,nan=-1)
    model.fit(Xf,ytr); pw=model.predict_proba(Xv)[:,1]; res[name]=pw
    print(name,score(mk(pw)))
from sklearn.preprocessing import StandardScaler
sc=StandardScaler().fit(np.nan_to_num(Xtr,nan=-1))
lr=LogisticRegression(C=1,max_iter=2000).fit(sc.transform(np.nan_to_num(Xtr,nan=-1)),ytr)
res['lr']=lr.predict_proba(sc.transform(np.nan_to_num(Xva,nan=-1)))[:,1]; print('lr',score(mk(res['lr'])))
bpw=1-base.p_0.values
for k in res: print('avg',k,'+bma',round(score(mk((res[k]+bpw)/2))[0],2))
print('lgb+xgb',round(score(mk((res['lgb']+res['xgb'])/2))[0],2))
np.save('research/pw_val.npy',np.c_[res['lgb'],res['xgb'],res['rf'],res['lr'],bpw])
