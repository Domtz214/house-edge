import sys,json; sys.path.insert(0,'research'); sys.path.insert(0,'my-entry-v6')
import features as F, train_v6 as T6, numpy as np, lightgbm as lgb, pandas as pd, math
from train_v5 import WIN_PARAMS, SIZE_PARAMS
tr=T6.pseudo(F,'public/train'); va=T6.pseudo(F,'public/validation')
w=lgb.LGBMClassifier(**WIN_PARAMS).fit(np.vstack([tr[0],va[0]]),np.concatenate([tr[1],va[1]]))
s=lgb.LGBMClassifier(**SIZE_PARAMS).fit(np.vstack([tr[2],va[2]]),np.concatenate([tr[3],va[3]]))
assert json.dumps(w.booster_.dump_model())==json.dumps(json.load(open('my-entry-v6/win_model.json'))), 'win trees differ'
tq,tf=T6.query_feats(F,'public/test')
X=np.array([tf[q['query_id']][0] for q in tq]); G=np.array([tf[q['query_id']][1] for q in tq])
print('NaN in features:',np.isnan(X).any(),np.isnan(G).any())
pw=w.predict_proba(X)[:,1]; sz=s.predict_proba(G); P=np.c_[1-pw,pw[:,None]*sz]; P/=P.sum(1,keepdims=True)
csv=pd.read_csv('v6-check/submission.csv').set_index('query_id').loc[[q['query_id'] for q in tq]].values
print('max |stdlib CSV - lightgbm| =',np.abs(csv-P).max())
